# RAG Document Q&A with Evaluation Harness

A retrieval-augmented generation system over a document corpus, built the way
production AI systems are built: **with an eval harness first**. Ask questions in
a Streamlit UI; every answer shows the retrieved passages and similarity scores.
An 18-question gold set measures retrieval quality (hit rate, MRR) and answer
quality (keyword coverage, optional LLM faithfulness judge).

## Quickstart

```bash
pip install -r requirements.txt
python run_eval.py --rebuild   # ingest docs, build the vector index, run evals
streamlit run app.py           # ask questions in the browser
```

Drop your own `.txt`, `.md`, or `.pdf` files into `data/docs/` and re-run
`--rebuild` to query them. Add questions to `data/eval_questions.json`.

## Deploy to Streamlit Cloud

The app builds its index automatically on first run, so no local build is
needed before deploying:

1. Push this folder to a GitHub repo (the included `.gitignore` excludes
   `.venv/` and the generated index).
2. On [share.streamlit.io](https://share.streamlit.io): **New app** → pick the
   repo → main file `app.py` → **Deploy**. Streamlit installs
   `requirements.txt` and builds the index from `data/docs/` on first boot.
3. Optional LLM mode: add `RAG_LLM_BASE_URL` / `RAG_LLM_API_KEY` /
   `RAG_LLM_MODEL` in the app's **Settings → Secrets**. A local Ollama
   (`localhost`) is not reachable from Streamlit Cloud — use a hosted
   OpenAI-compatible endpoint. Without an LLM the app runs in extractive mode.
4. Optional neural embeddings: by default the cloud deploy uses the TF-IDF
   fallback (sentence-transformers pulls in ~2 GB of PyTorch, heavy for the
   free tier). For neural embeddings without that install, add these Secrets
   instead — the app rebuilds its index with the new embedder automatically:
   `RAG_EMBEDDER = "api"`,
   `RAG_EMBEDDING_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"`,
   `RAG_EMBEDDING_API_KEY = "<Google AI Studio key>"`,
   `RAG_EMBEDDING_MODEL = "gemini-embedding-001"` (free tier).

## Architecture

```
data/docs/  ->  ingest (chunk 500 chars / 80 overlap)
            ->  embed (sentence-transformers, TF-IDF fallback)
            ->  store (numpy cosine-similarity vector store)
            ->  retrieve top-k
            ->  generate (OpenAI-compatible LLM, or extractive fallback)
            ->  evaluate (hit_rate@k, MRR, keyword coverage, LLM judge)
```

## Configuration

| Env var | Purpose |
|---|---|
| `RAG_EMBEDDER` | `auto` (default), `st`, `tfidf`, or `api` |
| `RAG_ST_MODEL` | sentence-transformers model when backend is `st` |
| `RAG_EMBEDDING_BASE_URL` | embeddings API base (backend `api`), e.g. Gemini below |
| `RAG_EMBEDDING_API_KEY` | embeddings API key (backend `api`) |
| `RAG_EMBEDDING_MODEL` | embeddings model (backend `api`), e.g. `gemini-embedding-001` |
| `RAG_LLM_BASE_URL` | e.g. `http://localhost:11434/v1` for Ollama |
| `RAG_LLM_API_KEY` | API key if the endpoint needs one |
| `RAG_LLM_MODEL` | model name, e.g. `llama3.1` |

With no LLM configured, the app runs end-to-end in extractive mode (returns the
top retrieved passages), so the retrieval pipeline and evals work with zero keys.

## Deliberate trade-offs

- **numpy vector store instead of Chroma/pgvector**: at this corpus size a
  brute-force cosine search is instant and has zero moving parts; the `VectorStore`
  interface is a drop-in point for Chroma or pgvector when scale demands it.
- **Pluggable embedder and LLM**: no vendor lock-in; the whole pipeline runs
  offline with the TF-IDF fallback and extractive answers.
- **Eval before polish**: 18 gold questions with deterministic metrics run in CI
  fashion (`python run_eval.py`); the optional LLM judge only adds the
  faithfulness check on top.
- **Chunking 500/80 chars**: word-aware with overlap so concepts aren't split
  mid-sentence; tuned by watching hit_rate@k on the eval set.
