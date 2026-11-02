"""ChemBot HTTP API — rate-limited and cost-capped.

Layered controls, cheapest first:
  1. Per-IP rate limit (slowapi).
  2. Answer cache — repeats are free and skip every cap below.
  3. Global daily live-query cap (the master cost ceiling).
  4. Global concurrency cap (queries are heavy/slow).
  5. Per-provider daily caps enforced inside the clients (backstops).
"""

# NB: no `from __future__ import annotations` here — it stringizes the endpoint
# annotations, and FastAPI can't resolve them through slowapi's @limit wrapper
# (body would be misread as a query param). Keep annotations as real objects.

import contextlib
import logging
import os
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from ..runtime import build_runtime
from ..usage import METER, QUERY, DailyCapExceeded, configure_limits_from_env, record_provider_call
from . import cache
from .examples import get_examples
from .limits import RATE_LIMITS, ServerBusy, concurrency_slot, limiter
from .schemas import AskRequest, AskResponse, UsageResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("chembot.api")

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:8080").split(",")
    if o.strip()
]


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Build the heavy runtime once: loads embedding + reranker models and spawns
    # the MCP subprocesses, reused for the process lifetime.
    limits = configure_limits_from_env()
    logger.info("Applied daily caps: %s", limits)
    logger.info("Building ChemBot runtime...")
    runtime = build_runtime()
    app.state.runtime = runtime
    logger.info("Runtime ready.")

    if os.getenv("PREWARM_EXAMPLES", "").strip().lower() in {"1", "true", "yes", "on"}:
        logger.info("Pre-warming example questions into the answer cache...")
        for q in get_examples():
            try:
                result = runtime.answer(q)
                cache.store(q, result["answer"], result["sources"])
            except Exception as exc:  # keep booting even if a prewarm query fails
                logger.warning("Prewarm failed for %r: %s", q, exc)
        logger.info("Prewarm complete (%d cached).", cache.size())

    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            runtime.close()


app = FastAPI(title="ChemBot API", lifespan=lifespan)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests — slow down, or try an example question."},
    )


@app.exception_handler(DailyCapExceeded)
async def daily_cap_handler(request: Request, exc: DailyCapExceeded) -> JSONResponse:
    if exc.key == QUERY:
        detail = (
            f"This demo has hit its daily limit of {exc.limit} live questions. "
            "Try an example question, or come back tomorrow."
        )
    else:
        detail = (
            f"A data source ({exc.key}) hit its daily request cap. "
            "Try an example question, or come back tomorrow."
        )
    return JSONResponse(status_code=429, content={"detail": detail})


@app.exception_handler(ServerBusy)
async def server_busy_handler(request: Request, exc: ServerBusy) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "The server is busy answering other questions. Please retry shortly."},
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/examples", response_model=list[str])
def examples() -> list[str]:
    return get_examples()


@app.get("/usage", response_model=UsageResponse)
def usage() -> dict:
    return METER.snapshot()


@app.post("/ask", response_model=AskResponse)
@limiter.limit(RATE_LIMITS)
def ask(request: Request, body: AskRequest) -> AskResponse:
    question = body.question.strip()

    # 1. Cache hit: free, bypasses every cap.
    cached = cache.peek(question)
    if cached is not None:
        answer, sources = cached
        logger.info("cache hit (q=%r)", question)
        return AskResponse(answer=answer, sources=sources, cached=True)

    # 2. Count this as a live question (master cost ceiling). Raises if over.
    record_provider_call(QUERY)

    # 3. Bounded concurrency — queries are heavy.
    with concurrency_slot():
        logger.info("cache miss, running pipeline (q=%r)", question)
        runtime = request.app.state.runtime
        result = runtime.answer(question)

    cache.store(question, result["answer"], result["sources"])
    return AskResponse(answer=result["answer"], sources=result["sources"], cached=False)
