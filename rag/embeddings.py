"""Pluggable embedding backends: sentence-transformers (neural) or TF-IDF (fallback)."""
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
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(config.ST_MODEL)
            print(f"[embedder] using sentence-transformers ({config.ST_MODEL})")
        elif backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=20000,
                                             ngram_range=(1, 2),
                                             stop_words="english")
            self._fitted = False
            print("[embedder] using TF-IDF fallback (no neural model installed)")
        else:
            raise ValueError(f"unknown embedder backend: {backend}")

    def fit(self, texts):
        if self.backend == "tfidf" and not self._fitted:
            self.vectorizer.fit(texts)
            self._fitted = True

    def embed(self, texts):
        if self.backend == "st":
            vecs = self.model.encode(texts, show_progress_bar=False,
                                     convert_to_numpy=True)
        else:
            if not self._fitted:
                raise RuntimeError("TF-IDF embedder not fitted; call fit() first")
            vecs = self.vectorizer.transform(texts).toarray()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-12)
