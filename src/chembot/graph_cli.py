from __future__ import annotations

import argparse
from typing import Annotated, TypedDict, List, Literal, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send

from langchain_openai import ChatOpenAI

from .rag import RAGStore
from .prompts import CHEMBOT_SYS, WELCOME
import os
from .openfda_mcp_client import OpenFDAMCPClient
from .openfda_client import OpenFDAClient
from .pubmed_mcp_client import PubMedMCPClient
from .cache_index import label_record_to_docs, event_record_to_docs, pubmed_record_to_docs
from .planner import QueryPlan, QueryPlanner


Route = Literal["label", "event", "pubmed"]


class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    retrieved: List[str]
    sources: List[dict]
    routes: List[str]
    query_plan: Optional[QueryPlan]
    finished: bool


# ----------------------------
# Helpers
# ----------------------------
def last_user_text(state: AgentState) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content.strip()
    return ""

planner = QueryPlanner()


# ----------------------------
# Nodes
# ----------------------------
def start_node(state: AgentState) -> AgentState:
    # One-time welcome when the graph starts
    return state | {"messages": [AIMessage(content=WELCOME)], "retrieved": [], "sources": [], "routes": [], "query_plan": None, "finished": False}


def human_node(state: AgentState) -> AgentState:
    # Print last assistant message if present
    if state.get("messages"):
        last = state["messages"][-1]
        if isinstance(last, AIMessage):
            print("\nChemBot:", last.content)

    user_in = input("\nUser: ").strip()
    if user_in.lower() in {"q", "quit", "exit"}:
        return state | {"finished": True}
    return state | {"messages": [HumanMessage(content=user_in)]}


def router_node(state: AgentState) -> AgentState:
    """
    Multi-route router: uses the QueryPlanner to normalize concepts,
    select routes, and generate optimized search queries.
    """
    q = last_user_text(state)

    if not q:
        return state | {"routes": ["label"], "query_plan": None}

    plan = planner.plan(q)

    if os.environ.get("CHEMBOT_DEBUG") == "1":
        import sys
        if plan.explanation:
            print(f"\n[PLANNER DEBUG] Explanation: {plan.explanation}", file=sys.stderr)
        if plan.drug_mappings:
            mappings_str = ", ".join(f"{brand} -> {', '.join(generics)}" for brand, generics in plan.drug_mappings.items())
            print(f"[PLANNER DEBUG] Synonym mappings: {mappings_str}", file=sys.stderr)
        print(f"[ROUTER DEBUG] Selected routes: {plan.routes} for query: '{q}'", file=sys.stderr)

    return state | {"routes": plan.routes, "query_plan": plan}


def label_retrieval_node_factory(
    rag: RAGStore,
    fda,
    label_limit: int,
    chunk_chars: int,
    overlap: int,
):
    def node(state: AgentState) -> dict:
        plan = state.get("query_plan")
        queries = plan.label_queries if plan else []

        for search in queries:
            # Fetch from openFDA label endpoint and cache into Chroma
            recs = fda.search_labels(search=search, limit=label_limit)

            for rec in recs:
                docs = label_record_to_docs(rec, chunk_chars=chunk_chars, overlap=overlap)
                if not docs:
                    continue
                texts = [t for (t, _) in docs]
                metas = [m for (_, m) in docs]
                ids = [m["chunk_id"] for m in metas]
                embs = rag.st.encode(texts, normalize_embeddings=True).tolist()
                rag.col.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embs)

        return {}
    return node


def event_retrieval_node_factory(
    rag: RAGStore,
    fda,
    event_limit: int,
    chunk_chars: int,
    overlap: int,
):
    def node(state: AgentState) -> dict:
        plan = state.get("query_plan")
        queries = plan.event_queries if plan else []

        for search in queries:
            # Fetch from openFDA event endpoint and cache into Chroma
            recs = fda.search_events(search=search, limit=event_limit)

            for rec in recs:
                docs = event_record_to_docs(rec, chunk_chars=chunk_chars, overlap=overlap)
                if not docs:
                    continue
                texts = [t for (t, _) in docs]
                metas = [m for (_, m) in docs]
                ids = [m["chunk_id"] for m in metas]
                embs = rag.st.encode(texts, normalize_embeddings=True).tolist()
                rag.col.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embs)

        return {}
    return node


def pubmed_retrieval_node_factory(
    rag: RAGStore,
    pubmed: PubMedMCPClient,
    pubmed_limit: int,
    chunk_chars: int,
    overlap: int,
):
    def node(state: AgentState) -> dict:
        plan = state.get("query_plan")
        queries = plan.pubmed_queries if plan else []

        for search in queries:
            # Fetch from PubMed via MCP and cache into Chroma
            recs = pubmed.search(query=search, max_results=pubmed_limit)

            for rec in recs:
                docs = pubmed_record_to_docs(rec, chunk_chars=chunk_chars, overlap=overlap)
                if not docs:
                    continue
                texts = [t for (t, _) in docs]
                metas = [m for (_, m) in docs]
                ids = [m["chunk_id"] for m in metas]
                embs = rag.st.encode(texts, normalize_embeddings=True).tolist()
                rag.col.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embs)

        return {}
    return node


def evidence_fusion_node_factory(rag: RAGStore, k: int):
    """Retrieves top-k chunks from Chroma and groups them by evidence source."""
    def node(state: AgentState) -> AgentState:
        q = last_user_text(state)
        if not q:
            return state

        plan = state.get("query_plan")
        search_query = q
        if plan:
            parts = []
            if plan.normalized_drugs:
                parts.extend(plan.normalized_drugs)
            if plan.normalized_terms:
                parts.extend(plan.normalized_terms)
            if parts:
                search_query = " ".join(parts)
                if os.environ.get("CHEMBOT_DEBUG") == "1":
                    import sys
                    print(f"[RETRIEVAL DEBUG] Reformulated query for Chroma: '{search_query}' (original: '{q}')", file=sys.stderr)

        hits = rag.retrieve(search_query, k=k)

        # Group by source
        groups = {
            "openfda_label": [],
            "openfda_event": [],
            "pubmed": [],
        }
        for h in hits:
            source = h["meta"].get("source", "unknown")
            groups.setdefault(source, []).append(h)

        # Build grouped context blocks + a de-duplicated, structured source list.
        # Each underlying document (a label section / FAERS report / PubMed article)
        # becomes one source entry, tracking which excerpt numbers reference it so
        # the answer's inline (Excerpt N) citations map back to a real citation.
        context_blocks = []
        sources_map = {}  # doc key -> {"excerpts": [int], "line": str}
        excerpt_num = 1

        def add_source(key: str, line: str, n: int):
            entry = sources_map.setdefault(key, {"excerpts": [], "line": line})
            entry["excerpts"].append(n)

        if groups.get("openfda_label"):
            context_blocks.append("=== FDA LABEL EVIDENCE ===")
            for h in groups["openfda_label"]:
                meta = h["meta"]
                context_blocks.append(
                    f"[EXCERPT {excerpt_num}] "
                    f"source={meta.get('source')} "
                    f"drug={meta.get('drug_name')} "
                    f"section={meta.get('section')} "
                    f"effective_time={meta.get('effective_time')}\n"
                    f"{h['text']}"
                )
                add_source(
                    f"label::{meta.get('set_id')}::{meta.get('section')}",
                    f"FDA Label — {meta.get('drug_name')} "
                    f"(section: {meta.get('section')}; effective {meta.get('effective_time')})",
                    excerpt_num,
                )
                excerpt_num += 1

        if groups.get("openfda_event"):
            context_blocks.append("\n=== FAERS ADVERSE EVENT REPORTS ===")
            for h in groups["openfda_event"]:
                meta = h["meta"]
                context_blocks.append(
                    f"[EXCERPT {excerpt_num}] "
                    f"source={meta.get('source')} "
                    f"safetyreportid={meta.get('safetyreportid')} "
                    f"receivedate={meta.get('receivedate')}\n"
                    f"{h['text']}"
                )
                add_source(
                    f"event::{meta.get('safetyreportid')}",
                    f"FAERS Adverse Event Report — ID {meta.get('safetyreportid')} "
                    f"(received {meta.get('receivedate')})",
                    excerpt_num,
                )
                excerpt_num += 1

        if groups.get("pubmed"):
            context_blocks.append("\n=== PUBMED LITERATURE ===")
            for h in groups["pubmed"]:
                meta = h["meta"]
                context_blocks.append(
                    f"[EXCERPT {excerpt_num}] "
                    f"source={meta.get('source')} "
                    f"pmid={meta.get('pmid')} "
                    f"title={meta.get('title')} "
                    f"journal={meta.get('journal')} "
                    f"year={meta.get('year')}\n"
                    f"{h['text']}"
                )
                pmid = meta.get("pmid") or ""
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
                title = (meta.get("title") or "").rstrip(". ")
                add_source(
                    f"pubmed::{pmid}",
                    f"PubMed — PMID {pmid}: {title}. "
                    f"{meta.get('journal')} ({meta.get('year')}). {url}".strip(),
                    excerpt_num,
                )
                excerpt_num += 1

        sources = [
            {"excerpts": v["excerpts"], "line": v["line"]}
            for v in sources_map.values()
        ]

        return state | {"retrieved": context_blocks, "sources": sources}
    return node


def answer_node_factory(llm: ChatOpenAI):
    def node(state: AgentState) -> AgentState:
        q = last_user_text(state)
        if not q:
            return state

        context = "\n\n".join(state.get("retrieved", []))

        plan = state.get("query_plan")
        mapping_str = ""
        if plan and plan.drug_mappings:
            mappings = []
            for k, v in plan.drug_mappings.items():
                if v:
                    mappings.append(f"{k} -> {', '.join(v)}")
            if mappings:
                mapping_str = "Resolved drug mappings:\n" + "\n".join(mappings) + "\n\n"

        messages = [
            SystemMessage(content=CHEMBOT_SYS),
            HumanMessage(
                content=(
                    "EVIDENCE EXCERPTS:\n"
                    f"{context}\n\n"
                    f"{mapping_str}"
                    "USER QUESTION:\n"
                    f"{q}"
                )
            ),
        ]
        out = llm.invoke(messages)
        answer = out.content

        # Append an authoritative, verified Sources section built from the actual
        # retrieved metadata (not model-generated), so every response shows its sources.
        sources = state.get("sources") or []
        if sources:
            lines = ["", "---", "**Sources**"]
            for s in sources:
                excerpts = ", ".join(f"Excerpt {n}" for n in s["excerpts"])
                lines.append(f"- [{excerpts}] {s['line']}")
            answer = f"{answer}\n\n" + "\n".join(lines)

        return state | {"messages": [AIMessage(content=answer)]}
    return node


# ----------------------------
# Conditional edge fns
# ----------------------------
def route_fan_out(state: AgentState) -> list:
    """Fan-out: returns a list of Send objects for each selected route."""
    routes = state.get("routes", ["label"])
    if os.environ.get("CHEMBOT_DEBUG") == "1":
        import sys
        print(f"[ROUTER DEBUG] Fanning out to nodes: {[r + '_retrieval' for r in routes]}", file=sys.stderr)
    sends = []
    for r in routes:
        if r == "label":
            sends.append(Send("label_retrieval", state))
        elif r == "event":
            sends.append(Send("event_retrieval", state))
        elif r == "pubmed":
            sends.append(Send("pubmed_retrieval", state))
    return sends or [Send("label_retrieval", state)]


def continue_or_end(state: AgentState) -> Literal["router", "__end__"]:
    return "__end__" if state.get("finished") else "router"


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    # Storage / embeddings
    ap.add_argument("--chroma_dir", required=True)
    ap.add_argument("--collection", default="chembot_cache")
    ap.add_argument("--st_model", default="all-MiniLM-L6-v2")
    ap.add_argument("--st_device", default=None, help="cpu|cuda|mps")
    ap.add_argument("--k", type=int, default=6, help="top-k retrieved chunks for answer")

    # openFDA
    ap.add_argument("--openfda_api_key", default=None)
    ap.add_argument("--label_limit", type=int, default=3)
    ap.add_argument("--event_limit", type=int, default=3)
    ap.add_argument("--use_direct_openfda", action="store_true", help="Use direct OpenFDA API client instead of MCP server")

    # PubMed
    ap.add_argument("--pubmed_limit", type=int, default=5, help="Max PubMed abstracts to fetch")
    ap.add_argument("--disable_pubmed", action="store_true", help="Disable PubMed literature retrieval")

    # Chunking
    ap.add_argument("--chunk_chars", type=int, default=2200)
    ap.add_argument("--overlap", type=int, default=250)

    # Reranking (cross-encoder; retrieve wider then rerank to --k)
    ap.add_argument("--disable_rerank", action="store_true", help="Disable cross-encoder reranking")
    ap.add_argument("--reranker_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--rerank_candidates", type=int, default=30, help="Dense candidates fetched before reranking to --k")

    # OpenAI
    ap.add_argument("--openai_model", default="gpt-4.1-mini")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=900)

    args = ap.parse_args()

    reranker = None
    if not args.disable_rerank:
        from .reranker import Reranker
        reranker = Reranker(args.reranker_model, device=args.st_device)
    rag = RAGStore(
        args.chroma_dir, args.collection, args.st_model, device=args.st_device,
        reranker=reranker, rerank_candidates=args.rerank_candidates,
    )
    if args.use_direct_openfda:
        fda = OpenFDAClient(api_key=args.openfda_api_key)
    else:
        if args.openfda_api_key:
            os.environ["OPENFDA_API_KEY"] = args.openfda_api_key
        fda = OpenFDAMCPClient()
    llm = ChatOpenAI(model=args.openai_model, temperature=args.temperature, max_tokens=args.max_tokens)

    # PubMed client (unless disabled)
    pubmed = None if args.disable_pubmed else PubMedMCPClient()

    graph = StateGraph(AgentState)

    graph.add_node("start", start_node)
    graph.add_node("human", human_node)
    graph.add_node("router", router_node)
    graph.add_node(
        "label_retrieval",
        label_retrieval_node_factory(
            rag=rag,
            fda=fda,
            label_limit=args.label_limit,
            chunk_chars=args.chunk_chars,
            overlap=args.overlap,
        ),
    )
    graph.add_node(
        "event_retrieval",
        event_retrieval_node_factory(
            rag=rag,
            fda=fda,
            event_limit=args.event_limit,
            chunk_chars=args.chunk_chars,
            overlap=args.overlap,
        ),
    )

    if pubmed is not None:
        graph.add_node(
            "pubmed_retrieval",
            pubmed_retrieval_node_factory(
                rag=rag,
                pubmed=pubmed,
                pubmed_limit=args.pubmed_limit,
                chunk_chars=args.chunk_chars,
                overlap=args.overlap,
            ),
        )

    graph.add_node("evidence_fusion", evidence_fusion_node_factory(rag=rag, k=args.k))
    graph.add_node("answer", answer_node_factory(llm=llm))

    # Flow
    graph.add_edge(START, "start")
    graph.add_edge("start", "human")

    # Router fan-out to 1-3 retrieval branches
    if pubmed is not None:
        graph.add_conditional_edges("router", route_fan_out)
    else:
        # Without PubMed, use simplified routing that filters out pubmed routes
        def route_fan_out_no_pubmed(state: AgentState) -> list:
            routes = [r for r in state.get("routes", ["label"]) if r != "pubmed"]
            if os.environ.get("CHEMBOT_DEBUG") == "1":
                import sys
                print(f"[ROUTER DEBUG] Fanning out (PubMed disabled) to nodes: {[r + '_retrieval' for r in routes]}", file=sys.stderr)
            sends = []
            for r in routes:
                if r == "label":
                    sends.append(Send("label_retrieval", state))
                elif r == "event":
                    sends.append(Send("event_retrieval", state))
            return sends or [Send("label_retrieval", state)]
        graph.add_conditional_edges("router", route_fan_out_no_pubmed)

    # All retrieval branches converge to evidence_fusion
    graph.add_edge("label_retrieval", "evidence_fusion")
    graph.add_edge("event_retrieval", "evidence_fusion")
    if pubmed is not None:
        graph.add_edge("pubmed_retrieval", "evidence_fusion")

    # Evidence fusion → answer → human loop
    graph.add_edge("evidence_fusion", "answer")
    graph.add_edge("answer", "human")

    # Human node: either continue to router or end
    graph.add_conditional_edges("human", continue_or_end, {"router": "router", "__end__": END})

    app = graph.compile()

    state: AgentState = {"messages": [], "retrieved": [], "sources": [], "routes": [], "query_plan": None, "finished": False}
    while True:
        state = app.invoke(state)
        if state.get("finished"):
            print("\nBye!")
            break


if __name__ == "__main__":
    main()