"""Pluggable embedding backends: sentence-transformers (neural, local),
hosted embeddings API (neural, no heavy install), or TF-IDF (fallback)."""
import numpy as np

from . import config


class Embedder:
    def __init__(self, backend=None):
        backend = backend or config.EMBEDDER
        if backend == "auto":
            try:
                import sentence_transformers  # noqa: F401
                backend = "st"
            except ImportError:
                backend = "tfidf"
        self.backend = backend
        if backend == "st":
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(config.ST_MODEL)
            except ImportError:
                # Requested but not installed (e.g. the ~2 GB PyTorch install
                # failed on a free-tier deploy): degrade to TF-IDF with a loud
                # log warning instead of crashing the whole app.
                print("[embedder] WARNING: sentence-transformers requested "
                      "but not installed — falling back to TF-IDF. Check the "
                      "deploy logs for a failed requirements install.")
                backend = "tfidf"
                self.backend = "tfidf"
            else:
                print(f"[embedder] using sentence-transformers ({config.ST_MODEL})")
        if backend == "api":
            import requests
            self._requests = requests
            self.emb_base_url = config.EMBEDDING_BASE_URL.rstrip("/")
            self.emb_api_key = config.EMBEDDING_API_KEY
            self.emb_model = config.EMBEDDING_MODEL
            if not (self.emb_base_url and self.emb_model):
                raise ValueError("RAG_EMBEDDER=api requires "
                                 "RAG_EMBEDDING_BASE_URL and RAG_EMBEDDING_MODEL")
            print(f"[embedder] using hosted embeddings API ({self.emb_model})")
        elif backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=20000,
                                             ngram_range=(1, 2),
                                             stop_words="english")
            self._fitted = False
            print("[embedder] using TF-IDF fallback (no neural model installed)")
        elif backend != "st":
            raise ValueError(f"unknown embedder backend: {backend}")

    def fit(self, texts):
        if self.backend == "tfidf" and not self._fitted:
            self.vectorizer.fit(texts)
            self._fitted = True

    def embed(self, texts):
        if self.backend == "api":
            headers = {"Content-Type": "application/json"}
            if self.emb_api_key:
                headers["Authorization"] = f"Bearer {self.emb_api_key}"
            resp = self._requests.post(
                f"{self.emb_base_url}/embeddings",
                headers=headers,
                json={"model": self.emb_model, "input": list(texts)},
                timeout=120,
            )
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            vecs = np.array([d["embedding"] for d in data], dtype=float)
        elif self.backend == "st":
            vecs = self.model.encode(texts, show_progress_bar=False,
                                     convert_to_numpy=True)
        else:
            if not self._fitted:
                raise RuntimeError("TF-IDF embedder not fitted; call fit() first")
            vecs = self.vectorizer.transform(texts).toarray()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-12)
