"""Reusable, single-turn ChemBot runtime.

`graph_cli.py` builds an interactive REPL graph (start -> human -> router -> ...
-> human loop). The API server needs the same pipeline as a stateless,
single-turn callable: given a question, return an answer plus the verified
sources. This module extracts that, building the agent once (models, MCP
clients, compiled graph) so it can be reused across many requests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from .graph_cli import (
    AgentState,
    router_node,
    label_retrieval_node_factory,
    event_retrieval_node_factory,
    pubmed_retrieval_node_factory,
    evidence_fusion_node_factory,
    answer_node_factory,
)
from .openfda_client import OpenFDAClient
from .openfda_mcp_client import OpenFDAMCPClient
from .pubmed_mcp_client import PubMedMCPClient
from .rag import RAGStore


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class RuntimeConfig:
    """ChemBot runtime configuration. Mirrors the CLI flags, but sourced from the
    environment so the same pipeline can be configured in a container."""

    # Storage / embeddings
    chroma_dir: str = "./chroma_db"
    collection: str = "chembot_cache"
    st_model: str = "all-MiniLM-L6-v2"
    st_device: Optional[str] = None
    k: int = 6

    # openFDA
    openfda_api_key: Optional[str] = None
    label_limit: int = 3
    event_limit: int = 3
    use_direct_openfda: bool = False

    # PubMed
    pubmed_limit: int = 5
    disable_pubmed: bool = False

    # Chunking
    chunk_chars: int = 2200
    overlap: int = 250

    # Reranking
    disable_rerank: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 30
    disable_drug_filter: bool = False

    # OpenAI
    openai_model: str = "gpt-4.1-mini"
    temperature: float = 0.2
    max_tokens: int = 900

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        d = cls()
        d.chroma_dir = os.getenv("CHEMBOT_CHROMA_DIR", d.chroma_dir)
        d.collection = os.getenv("CHEMBOT_COLLECTION", d.collection)
        d.st_model = os.getenv("CHEMBOT_ST_MODEL", d.st_model)
        d.st_device = os.getenv("CHEMBOT_ST_DEVICE", d.st_device)
        d.k = int(os.getenv("CHEMBOT_K", d.k))
        d.openfda_api_key = os.getenv("OPENFDA_API_KEY", d.openfda_api_key)
        d.label_limit = int(os.getenv("CHEMBOT_LABEL_LIMIT", d.label_limit))
        d.event_limit = int(os.getenv("CHEMBOT_EVENT_LIMIT", d.event_limit))
        d.use_direct_openfda = _env_flag("CHEMBOT_USE_DIRECT_OPENFDA", d.use_direct_openfda)
        d.pubmed_limit = int(os.getenv("CHEMBOT_PUBMED_LIMIT", d.pubmed_limit))
        d.disable_pubmed = _env_flag("CHEMBOT_DISABLE_PUBMED", d.disable_pubmed)
        d.chunk_chars = int(os.getenv("CHEMBOT_CHUNK_CHARS", d.chunk_chars))
        d.overlap = int(os.getenv("CHEMBOT_OVERLAP", d.overlap))
        d.disable_rerank = _env_flag("CHEMBOT_DISABLE_RERANK", d.disable_rerank)
        d.reranker_model = os.getenv("CHEMBOT_RERANKER_MODEL", d.reranker_model)
        d.rerank_candidates = int(os.getenv("CHEMBOT_RERANK_CANDIDATES", d.rerank_candidates))
        d.disable_drug_filter = _env_flag("CHEMBOT_DISABLE_DRUG_FILTER", d.disable_drug_filter)
        d.openai_model = os.getenv("CHEMBOT_OPENAI_MODEL", d.openai_model)
        d.temperature = float(os.getenv("CHEMBOT_TEMPERATURE", d.temperature))
        d.max_tokens = int(os.getenv("CHEMBOT_MAX_TOKENS", d.max_tokens))
        return d


class ChembotRuntime:
    """A built, reusable ChemBot pipeline. Construct once (loads models, spawns MCP
    subprocesses, compiles the graph); call `answer()` per question."""

    def __init__(self, config: RuntimeConfig):
        self.config = config

        reranker = None
        if not config.disable_rerank:
            from .reranker import Reranker

            reranker = Reranker(config.reranker_model, device=config.st_device)

        self.rag = RAGStore(
            config.chroma_dir,
            config.collection,
            config.st_model,
            device=config.st_device,
            reranker=reranker,
            rerank_candidates=config.rerank_candidates,
        )

        if config.use_direct_openfda:
            self.fda = OpenFDAClient(api_key=config.openfda_api_key)
        else:
            if config.openfda_api_key:
                os.environ["OPENFDA_API_KEY"] = config.openfda_api_key
            self.fda = OpenFDAMCPClient()

        self.pubmed = None if config.disable_pubmed else PubMedMCPClient()
        self.llm = ChatOpenAI(
            model=config.openai_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self.app = self._build_graph()

    def _build_graph(self):
        cfg = self.config
        graph = StateGraph(AgentState)

        graph.add_node("router", router_node)
        graph.add_node(
            "label_retrieval",
            label_retrieval_node_factory(
                rag=self.rag, fda=self.fda, label_limit=cfg.label_limit,
                chunk_chars=cfg.chunk_chars, overlap=cfg.overlap,
            ),
        )
        graph.add_node(
            "event_retrieval",
            event_retrieval_node_factory(
                rag=self.rag, fda=self.fda, event_limit=cfg.event_limit,
                chunk_chars=cfg.chunk_chars, overlap=cfg.overlap,
            ),
        )
        if self.pubmed is not None:
            graph.add_node(
                "pubmed_retrieval",
                pubmed_retrieval_node_factory(
                    rag=self.rag, pubmed=self.pubmed, pubmed_limit=cfg.pubmed_limit,
                    chunk_chars=cfg.chunk_chars, overlap=cfg.overlap,
                ),
            )
        graph.add_node(
            "evidence_fusion",
            evidence_fusion_node_factory(
                rag=self.rag, k=cfg.k, drug_filter=not cfg.disable_drug_filter
            ),
        )
        graph.add_node("answer", answer_node_factory(llm=self.llm))

        # Single-turn flow: router -> fan-out -> retrieval -> fusion -> answer -> END.
        # No start/human nodes; the question is seeded into the initial state.
        pubmed_on = self.pubmed is not None

        def fan_out(state: AgentState) -> list:
            routes = state.get("routes", ["label"])
            if not pubmed_on:
                routes = [r for r in routes if r != "pubmed"]
            sends = []
            for r in routes:
                if r == "label":
                    sends.append(Send("label_retrieval", state))
                elif r == "event":
                    sends.append(Send("event_retrieval", state))
                elif r == "pubmed" and pubmed_on:
                    sends.append(Send("pubmed_retrieval", state))
            return sends or [Send("label_retrieval", state)]

        graph.add_edge(START, "router")
        graph.add_conditional_edges("router", fan_out)
        graph.add_edge("label_retrieval", "evidence_fusion")
        graph.add_edge("event_retrieval", "evidence_fusion")
        if pubmed_on:
            graph.add_edge("pubmed_retrieval", "evidence_fusion")
        graph.add_edge("evidence_fusion", "answer")
        graph.add_edge("answer", END)

        return graph.compile()

    def answer(self, question: str) -> Dict[str, Any]:
        """Run the pipeline on one question. Returns {"answer": str, "sources": [...]}."""
        # One answered question = one OpenAI completion; record it up front so the
        # cap is enforced before we spend (raises DailyCapExceeded if over).
        from .usage import OPENAI, record_provider_call

        record_provider_call(OPENAI)

        state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "retrieved": [],
            "sources": [],
            "routes": [],
            "query_plan": None,
            "finished": False,
        }
        final = self.app.invoke(state)

        answer_text = ""
        for m in reversed(final.get("messages", [])):
            if isinstance(m, AIMessage):
                answer_text = m.content
                break

        return {"answer": answer_text, "sources": final.get("sources", [])}

    def close(self) -> None:
        """Tear down MCP subprocesses. Safe to call multiple times."""
        for client in (self.fda, self.pubmed):
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


def build_runtime(config: Optional[RuntimeConfig] = None) -> ChembotRuntime:
    """Build a ChemBot runtime from the given config (or the environment)."""
    return ChembotRuntime(config or RuntimeConfig.from_env())
