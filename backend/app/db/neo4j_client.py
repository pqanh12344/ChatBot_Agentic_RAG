from neo4j import GraphDatabase
from app.core.config import settings
import numpy as np
import json

class Neo4jClient:
    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or settings.NEO4J_URI
        self.user = user or settings.NEO4J_USER
        self.password = password or settings.NEO4J_PASSWORD
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def create_document(self, doc_id: str, text: str, meta: dict, embedding: list):
        with self.driver.session() as session:
            session.write_transaction(self._create_doc_tx, doc_id, text, meta, embedding)

    @staticmethod
    def _create_doc_tx(tx, doc_id, text, meta, embedding):
        tx.run(
            """
            MERGE (d:Document {id: $doc_id})
            SET d.text = $text,
                d.meta = $meta,
                d.embedding = $embedding
            """,
            doc_id=doc_id,
            text=text,
            meta=json.dumps(meta, ensure_ascii=False),
            embedding=embedding
        )

    def fetch_all_embeddings(self):
        with self.driver.session() as session:
            res = session.run("MATCH (d:Document) RETURN d.id AS id, d.embedding AS emb, d.text AS text, d.meta AS meta")
            rows = []
            for r in res:
                emb = r["emb"]
                # Neo4j may return lists; convert to numpy
                rows.append({
                    "id": r["id"],
                    "embedding": np.array(emb, dtype="float32"),
                    "text": r["text"],
                    "meta": json.loads(r["meta"]) if r["meta"] else {}
                })
            return rows
