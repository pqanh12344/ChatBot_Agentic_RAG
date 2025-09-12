from app.services.retriever.base_retriever import BaseRetriever
from app.db.neo4j_client import Neo4jClient
from app.core.config import settings
from langchain.schema import Document
import numpy as np
from sentence_transformers import SentenceTransformer

class Neo4jRetriever(BaseRetriever):
    def __init__(self, neo4j_client: Neo4jClient, embed_model: str = None):
        self.neo4j = neo4j_client
        self.embed_model = embed_model or settings.EMBED_MODEL
        self.embedder = SentenceTransformer(self.embed_model)

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray):
        # returns similarity
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get_relevant_documents(self, query: str, k: int = None):
        k = k or settings.TOP_K
        qvec = self.embedder.encode([query])[0].astype("float32")
        rows = self.neo4j.fetch_all_embeddings()
        scored = []
        for r in rows:
            sim = self._cosine_sim(qvec, r["embedding"])
            scored.append((sim, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        docs = []
        for sim, r in scored[:k]:
            docs.append(Document(page_content=r["text"], metadata={**r["meta"], "id": r["id"], "score": sim}))
        return docs
