from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer

class RAGStore:
    def __init__(
        self,
        chroma_dir: str,
        collection: str,
        st_model: str,
        device: str | None = None,
        reranker=None,
        rerank_candidates: int = 30,
    ):
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.col = self.client.get_or_create_collection(name=collection)
        self.st = SentenceTransformer(st_model, device=device)
        # Optional cross-encoder reranker. When set, retrieve() fetches a wider dense
        # candidate set (rerank_candidates) and reranks it down to k.
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        n_fetch = max(k, self.rerank_candidates) if self.reranker else k
        q_emb = self.st.encode([query], normalize_embeddings=True).tolist()[0]
        out = self.col.query(query_embeddings=[q_emb], n_results=n_fetch, include=["documents", "metadatas", "distances"])
        docs = out["documents"][0]
        metas = out["metadatas"][0]
        dists = out["distances"][0]
        hits = [{"text": d, "meta": m, "distance": dist} for d, m, dist in zip(docs, metas, dists)]
        if self.reranker and hits:
            hits = self.reranker.rerank(query, hits, top_k=k)
        return hits
