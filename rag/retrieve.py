"""Retrieval: embed the query, cosine-search the store."""
from . import config
from .embeddings import Embedder
from .store import VectorStore


class Retriever:
    def __init__(self, embedder=None, store=None):
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore.load()

    def retrieve(self, query, k=None):
        k = k or config.TOP_K
        qv = self.embedder.embed([query])[0]
        return self.store.search(qv, k=k)
