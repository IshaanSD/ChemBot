"""Request-level throttles: per-IP rate limiting and a global concurrency cap.

These sit in front of the cost caps in `chembot.usage`. Per-IP limiting blunts
abuse; the concurrency cap matters because each ChemBot query is heavy and slow
(MCP calls + embedding + cross-encoder + LLM), so unbounded parallelism would
exhaust a single warm instance.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager

from slowapi import Limiter
from slowapi.util import get_remote_address

# Per-IP limits. Multiple rates are semicolon-separated (handled by `limits`).
RATE_LIMITS = os.getenv("RATE_LIMITS", "5/minute;20/hour")

limiter = Limiter(key_func=get_remote_address, default_limits=[])

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))
ACQUIRE_TIMEOUT = float(os.getenv("CONCURRENCY_ACQUIRE_TIMEOUT", "0.5"))

_semaphore = threading.BoundedSemaphore(MAX_CONCURRENCY)


class ServerBusy(Exception):
    """Raised when the concurrency cap is saturated."""


@contextmanager
def concurrency_slot():
    """Acquire one of the limited in-flight slots, or raise ServerBusy quickly."""
    acquired = _semaphore.acquire(timeout=ACQUIRE_TIMEOUT)
    if not acquired:
        raise ServerBusy()
    try:
        yield
    finally:
        _semaphore.release()
