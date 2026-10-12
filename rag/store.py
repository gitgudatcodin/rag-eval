"""Minimal numpy vector store (cosine similarity). Chroma/pgvector are drop-in replacements."""
import json
import os

import numpy as np

from . import config


class VectorStore:
    def __init__(self):
        self.ids, self.vectors, self.metas = [], None, []

    def add(self, ids, vectors, metas):
        self.ids.extend(ids)
        self.metas.extend(metas)
        arr = np.asarray(vectors, dtype=np.float32)
        self.vectors = arr if self.vectors is None else np.vstack([self.vectors, arr])

    def search(self, query_vec, k=None):
        k = k or config.TOP_K
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        scores = self.vectors @ q  # cosine: both sides L2-normalized
        idx = np.argsort(scores)[::-1][:k]
        return [(self.ids[i], float(scores[i]), self.metas[i]) for i in idx]

    def save(self, store_dir=None):
        store_dir = store_dir or config.STORE_DIR
        os.makedirs(store_dir, exist_ok=True)
        np.save(os.path.join(store_dir, "vectors.npy"), self.vectors)
        with open(os.path.join(store_dir, "meta.json"), "w") as f:
            json.dump({"ids": self.ids, "metas": self.metas}, f)

    @classmethod
    def load(cls, store_dir=None):
        store_dir = store_dir or config.STORE_DIR
        s = cls()
        s.vectors = np.load(os.path.join(store_dir, "vectors.npy"))
        with open(os.path.join(store_dir, "meta.json")) as f:
            blob = json.load(f)
        s.ids, s.metas = blob["ids"], blob["metas"]
        return s

    def __len__(self):
        return len(self.ids)
