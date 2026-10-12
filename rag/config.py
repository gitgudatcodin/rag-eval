"""Shared configuration for the RAG app."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "data", "docs")
STORE_DIR = os.path.join(BASE_DIR, "data", "store")
EVAL_PATH = os.path.join(BASE_DIR, "data", "eval_questions.json")

CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 80      # characters of overlap between chunks
TOP_K = 4               # chunks retrieved per query

# Embedding backend: "auto" (sentence-transformers if installed, else TF-IDF),
# "st", or "tfidf". Override with RAG_EMBEDDER env var.
EMBEDDER = os.environ.get("RAG_EMBEDDER", "auto")
ST_MODEL = os.environ.get("RAG_ST_MODEL", "all-MiniLM-L6-v2")

# LLM settings (OpenAI-compatible chat completions endpoint).
# Works with Ollama (http://localhost:11434/v1, no key), OpenAI, etc.
# If unset, the app runs in extractive mode (returns top retrieved chunks).
LLM_BASE_URL = os.environ.get("RAG_LLM_BASE_URL", "")
LLM_API_KEY = os.environ.get("RAG_LLM_API_KEY", "")
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "")
