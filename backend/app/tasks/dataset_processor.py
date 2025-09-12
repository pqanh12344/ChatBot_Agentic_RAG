import os
import uuid
from typing import List
from kaggle.api.kaggle_api_extended import KaggleApi
from sentence_transformers import SentenceTransformer
from app.db.neo4j_client import Neo4jClient
from app.core.config import settings
from app.utils.text import chunk_text
import pandas as pd
import json

class DatasetProcessor:
    def __init__(self, dataset_id: str, target_dir: str = None):
        self.dataset_id = dataset_id
        self.target_dir = target_dir or settings.KAGGLE_DATA_DIR
        self.kaggle = KaggleApi()
        self.kaggle.authenticate()
        self.embedder = SentenceTransformer(settings.EMBED_MODEL)
        self.neo4j = Neo4jClient()

    def download(self):
        os.makedirs(self.target_dir, exist_ok=True)
        self.kaggle.dataset_download_files(self.dataset_id, path=self.target_dir, unzip=True)
        print("Downloaded dataset to", self.target_dir)

    def _read_texts_from_dir(self) -> List[dict]:
        items = []
        for root,_,files in os.walk(self.target_dir):
            for f in files:
                p = os.path.join(root, f)
                if f.endswith(".csv"):
                    df = pd.read_csv(p, dtype=str, encoding="utf-8", errors="ignore")
                    # heuristics: find a text-like column
                    for col in ["text","content","article","body","title","Title","content_text"]:
                        if col in df.columns:
                            for i,row in df.iterrows():
                                txt = str(row[col]) if not pd.isna(row[col]) else ""
                                if len(txt.strip())>50:
                                    items.append({"text": txt.strip(), "source_file": f, "row": int(i)})
                            break
                elif f.endswith(".txt"):
                    with open(p,"r",encoding="utf-8",errors="ignore") as fh:
                        text = fh.read().strip()
                        if text:
                            items.append({"text": text, "source_file": f, "row": 0})
        return items

    def process_and_store(self):
        raw_items = self._read_texts_from_dir()
        print(f"Found {len(raw_items)} text items")
        for it in raw_items:
            chunks = chunk_text(it["text"], max_chars=800)  # chunk to manageable pieces
            embeddings = self.embedder.encode(chunks)
            for idx, chunk in enumerate(chunks):
                doc_id = f"{uuid.uuid4()}"
                meta = {"source_file": it["source_file"], "row": it["row"], "chunk_idx": idx}
                emb = embeddings[idx].tolist()
                self.neo4j.create_document(doc_id, chunk, meta, emb)
        print("Done storing to Neo4j")
