"""CLI: build the index and/or run the eval harness.

Usage:
    python run_eval.py            # build index if missing, run eval
    python run_eval.py --rebuild  # force re-ingest + re-embed
    python run_eval.py --judge    # also run LLM faithfulness judge (needs LLM configured)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag import config
from rag.embeddings import Embedder
from rag.evaluate import print_report, run_eval
from rag.ingest import build_chunks
from rag.retrieve import Retriever
from rag.store import VectorStore


def build_index():
    chunks = build_chunks()
    print(f"[index] {len(chunks)} chunks from {config.DOCS_DIR}")
    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    embedder.fit(texts)
    vecs = embedder.embed(texts)
    store = VectorStore()
    store.add([c["id"] for c in chunks], vecs,
              [{"doc": c["doc"], "chunk_index": c["chunk_index"], "text": c["text"]}
               for c in chunks])
    store.save()
    # persist embedder choice for serving
    with open(os.path.join(config.STORE_DIR, "embedder.txt"), "w") as f:
        f.write(embedder.backend)
    print(f"[index] saved {len(store)} vectors ({embedder.backend}) -> {config.STORE_DIR}")


def serving_retriever():
    store_dir = config.STORE_DIR
    backend_path = os.path.join(store_dir, "embedder.txt")
    backend = open(backend_path).read().strip() if os.path.exists(backend_path) else None
    embedder = Embedder(backend=backend)
    if backend == "tfidf":
        # refit the vectorizer on the corpus so query embeddings match
        embedder.fit([c["text"] for c in build_chunks()])
    return Retriever(embedder=embedder, store=VectorStore.load(store_dir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()
    if args.rebuild or not os.path.exists(os.path.join(config.STORE_DIR, "vectors.npy")):
        build_index()
    retriever = serving_retriever()
    summary, rows = run_eval(retriever, judge=args.judge)
    print_report(summary, rows)


if __name__ == "__main__":
    main()
