from __future__ import annotations

from typing import Any, Dict, List, Optional


class Reranker:
    """Cross-encoder reranker.

    Dense retrieval (bi-encoder) is fast but coarse: it scores query and chunk
    independently. A cross-encoder reads the (query, chunk) pair together and scores
    relevance directly, which is far more accurate — but too slow to run over the whole
    store. The standard recipe is therefore: dense-retrieve a wide candidate set
    (top-N), then rerank it down to the final top-k with the cross-encoder.

    Default is a small, fast, CPU-friendly reranker. For biomedical text, swap in
    'BAAI/bge-reranker-v2-m3' (stronger, larger) or 'ncbi/MedCPT-Cross-Encoder'
    (trained on PubMed) via `model_name`.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self, query: str, hits: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """Score each hit's text against the query and return the top_k, best first.

        Attaches a `rerank_score` to each returned hit. Input hits are the dicts from
        `RAGStore.retrieve` ({text, meta, distance})."""
        if not hits:
            return hits

        pairs = [[query, h["text"]] for h in hits]
        scores = self.model.predict(pairs)
        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)

        ranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
        return ranked[:top_k]
