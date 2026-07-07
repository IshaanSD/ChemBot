from __future__ import annotations

import argparse
from typing import Annotated, TypedDict, List, Literal, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_openai import ChatOpenAI

from .rag import RAGStore
from .prompts import CHEMBOT_SYS, WELCOME
import os
from .openfda_mcp_client import OpenFDAMCPClient
from .openfda_client import OpenFDAClient
from .cache_index import label_record_to_docs, event_record_to_docs


Route = Literal["label", "event", "__end__"]


class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    retrieved: List[str]
    route: Optional[str]
    finished: bool


# ----------------------------
# Helpers
# ----------------------------
def last_user_text(state: AgentState) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content.strip()
    return ""


def build_label_search(q: str) -> str:
    # Broad + practical starter query: search by drug identity fields.
    # You can later add a dedicated "query_builder" node if you want.
    q = q.replace('"', "")
    return (
        f'openfda.brand_name:"{q}" OR openfda.generic_name:"{q}" OR openfda.substance_name:"{q}"'
    )


def build_event_search(q: str) -> str:
    # Matches the pattern in your example: reactionmeddrapt:"headache"
    # In "event" mode we interpret q as a reaction term; later you can improve routing/query building.
    q = q.replace('"', "")
    return f'reactionmeddrapt:"{q}"'


# ----------------------------
# Nodes
# ----------------------------
def start_node(state: AgentState) -> AgentState:
    # One-time welcome when the graph starts
    return state | {"messages": [AIMessage(content=WELCOME)], "retrieved": [], "route": None, "finished": False}


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
    Very simple router:
    - If the question looks like adverse event / side effect reporting, route to event.
    - Otherwise route to label.
    """
    q = last_user_text(state).lower()

    # If user didn't ask anything (e.g., right after welcome), go to human.
    # But in this graph design, we only call router after human, so this is just defensive.
    if not q:
        return state | {"route": "label"}

    event_markers = ["adverse", "faers", "reported", "report", "reaction", "side effect", "frequency", "how common"]
    route: Route = "event" if any(m in q for m in event_markers) else "label"
    return state | {"route": route}


def label_retrieval_node_factory(
    rag: RAGStore,
    fda: OpenFDAClient,
    label_limit: int,
    chunk_chars: int,
    overlap: int,
):
    def node(state: AgentState) -> AgentState:
        q = last_user_text(state)
        if not q:
            return state

        # Fetch from openFDA label endpoint and cache into Chroma
        search = build_label_search(q)
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

        return state
    return node


def event_retrieval_node_factory(
    rag: RAGStore,
    fda: OpenFDAClient,
    event_limit: int,
    chunk_chars: int,
    overlap: int,
):
    def node(state: AgentState) -> AgentState:
        q = last_user_text(state)
        if not q:
            return state

        # Fetch from openFDA event endpoint and cache into Chroma
        search = build_event_search(q)
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

        return state
    return node


def answer_node_factory(rag: RAGStore, llm: ChatOpenAI, k: int):
    def node(state: AgentState) -> AgentState:
        q = last_user_text(state)
        if not q:
            return state

        hits = rag.retrieve(q, k=k)
        context_blocks = []
        for i, h in enumerate(hits, 1):
            meta = h["meta"]
            context_blocks.append(
                f"[EXCERPT {i}] "
                f"source={meta.get('source')} "
                f"drug={meta.get('drug_name')} "
                f"section={meta.get('section')} "
                f"effective_time={meta.get('effective_time')}\n"
                f"{h['text']}"
            )
        context = "\n\n".join(context_blocks)

        messages = [
            SystemMessage(content=CHEMBOT_SYS),
            HumanMessage(
                content=(
                    "FDA DATA EXCERPTS:\n"
                    f"{context}\n\n"
                    "USER QUESTION:\n"
                    f"{q}"
                )
            ),
        ]
        out = llm.invoke(messages)
        return state | {"messages": [AIMessage(content=out.content)], "retrieved": context_blocks}
    return node


# ----------------------------
# Conditional edge fns
# ----------------------------
def route_branch(state: AgentState) -> Route:
    if state.get("finished"):
        return "__end__"
    r = state.get("route")
    if r == "event":
        return "event"
    return "label"


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
    ap.add_argument("--use_mcp_openfda", action="store_true", help="Use OpenFDA MCP server instead of direct client")

    # Chunking
    ap.add_argument("--chunk_chars", type=int, default=2200)
    ap.add_argument("--overlap", type=int, default=250)

    # OpenAI
    ap.add_argument("--openai_model", default="gpt-4.1-mini")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=900)

    args = ap.parse_args()

    rag = RAGStore(args.chroma_dir, args.collection, args.st_model, device=args.st_device)
    if args.use_mcp_openfda:
        if args.openfda_api_key:
            os.environ["OPENFDA_API_KEY"] = args.openfda_api_key
        fda = OpenFDAMCPClient()
    else:
        fda = OpenFDAClient(api_key=args.openfda_api_key)
    llm = ChatOpenAI(model=args.openai_model, temperature=args.temperature, max_tokens=args.max_tokens)

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
    graph.add_node("answer", answer_node_factory(rag=rag, llm=llm, k=args.k))

    # Flow
    graph.add_edge(START, "start")
    graph.add_edge("start", "human")
    graph.add_edge("human", "router")

    # Router branches
    graph.add_conditional_edges(
        "router",
        route_branch,
        {
            "label": "label_retrieval",
            "event": "event_retrieval",
            "__end__": END,
        },
    )

    # Both branches converge to answer
    graph.add_edge("label_retrieval", "answer")
    graph.add_edge("event_retrieval", "answer")

    # After answer, go to human; after human, either router again or end
    graph.add_edge("answer", "human")
    graph.add_conditional_edges("human", continue_or_end, {"router": "router", "__end__": END})

    app = graph.compile()

    state: AgentState = {"messages": [], "retrieved": [], "route": None, "finished": False}
    while True:
        state = app.invoke(state)
        if state.get("finished"):
            print("\nBye!")
            break


if __name__ == "__main__":
    main()