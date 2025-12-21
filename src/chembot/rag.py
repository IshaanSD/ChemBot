from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer

class RAGStore:
    def __init__(self, chroma_dir: str, collection: str, st_model: str, device: str | None = None):
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.col = self.client.get_or_create_collection(name=collection)
        self.st = SentenceTransformer(st_model, device=device)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self.st.encode([query], normalize_embeddings=True).tolist()[0]
        out = self.col.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas", "distances"])
        docs = out["documents"][0]
        metas = out["metadatas"][0]
        dists = out["distances"][0]
        return [{"text": d, "meta": m, "distance": dist} for d, m, dist in zip(docs, metas, dists)]
