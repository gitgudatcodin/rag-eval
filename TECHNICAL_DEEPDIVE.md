# RAG Document Q&A — Complete Technical Deep-Dive

A file-by-file explanation of the project: what each file does, the key
implementation details, why it is written that way, and exactly how the system
executes. Read this and you can whiteboard the whole system.

---

## 1. Big picture

The system answers questions over a private document corpus using
retrieval-augmented generation (RAG). Instead of asking an LLM to answer from
its training data (which invites hallucination), we first **retrieve** the most
relevant passages from our own documents, then ask the LLM to answer **using
only those passages**. A separate **evaluation harness** measures whether the
retrieval and answering actually work, using 18 gold-standard questions.

Three execution paths share the same building blocks:

- **Index build** (`python run_eval.py --rebuild`): documents → chunks →
  vectors → saved index. Run once per corpus change.
- **Serving** (`streamlit run app.py`): question → embed → top-k search →
  grounded answer → display with sources.
- **Evaluation** (`python run_eval.py`): 18 questions → retrieval metrics +
  answer metrics → printed report.

## 2. Key concepts primer (for readers new to RAG)

Assumes you're comfortable with code and math. Each concept: what it is, the
math, and why it matters in this system.

### Embeddings: text as geometry
An embedding model maps a piece of text to a vector in R^d (here d = 384) such
that *semantic similarity becomes geometric proximity*: paraphrases land near
each other, unrelated texts far apart. This works because of the distributional
hypothesis — text appearing in similar contexts gets similar vectors. Neural
embedding models are trained contrastively: the loss pulls embeddings of
paraphrase pairs together and pushes random pairs apart. Once text is geometry,
"find relevant passages" becomes "find nearest vectors" — a math problem
instead of a keyword problem.

### Cosine similarity: the relevance score
For vectors a and b: cos(theta) = (a . b) / (||a|| ||b||). We L2-normalize every
vector at embed time (divide by its norm), so ||a|| = ||b|| = 1 and cosine
similarity reduces to a plain dot product — which is why `store.search()` is a
single matrix multiply. We use cosine rather than Euclidean distance because we
care about *direction* (topic), not *magnitude*: a long chunk and a short chunk
on the same topic should score similarly, while raw dot products would favor
long vectors.

### TF-IDF: the non-neural baseline
TF-IDF scores a term t in document d as tf(t,d) x log(N / df(t)): how often the
term appears here, discounted by how common it is across all N documents.
"Expected loss" in a credit-risk doc scores high; "the" scores near zero. Our
fallback uses 1-2 grams (so "expected loss" is a feature, not just "expected" +
"loss") and drops English stop words, capped at 20,000 features. The result is a
*sparse* 20k-dim vector versus the *dense* 384-dim neural one. TF-IDF only
matches literal words — it cannot see that "PD times LGD" and "expected loss
formula" mean the same thing. That gap is exactly what neural embeddings close.

### Chunking: why not embed whole documents
Two reasons. First, embedding models have input limits (MiniLM: 512 tokens) —
long docs get truncated. Second, and more important, *retrieval precision*: a
whole document's vector averages over everything it discusses, so a specific
question's signal drowns. Chunks (~500 chars) are small enough that one chunk is
roughly one idea. Too small and you lose context (a pronoun's antecedent lives
in the previous chunk); too big and the signal dilutes again. The 80-char
overlap is insurance: if a definition straddles a boundary, at least one chunk
contains it whole. Chunking is word-aware (never splits mid-word) for the same
reason.

### Top-k and the precision/recall trade-off
k is a knob, not a constant. Small k (1-2): focused context, cheap, but one bad
retrieval sinks the answer. Large k (8+): the right passage is almost surely
included, but you pay in tokens, latency, and noise — irrelevant passages
distract the model ("lost in the middle" is a measured phenomenon: models
overweight the start and end of long contexts). The Streamlit slider lets you
feel this directly; the eval harness measures it (hit_rate@k rises with k,
answer quality eventually falls).

### Grounded generation: the prompt as contract
The LLM never sees the question alone — it sees retrieved passages plus an
instruction: answer *only* from the context, and say "I don't know based on the
provided documents" otherwise. Temperature 0.1 keeps sampling near-greedy (less
creativity, fewer hallucinations). This is the whole anti-hallucination strategy
in one place: constrain the model's knowledge to what you retrieved, and give
it an explicit abstention path instead of forcing an answer.

### Hallucination (and what RAG doesn't fix)
Hallucination = fluent, confident, wrong. RAG attacks the main cause (the model
answering from stale parametric memory), but two failure modes remain:
retrieval fetches the *wrong* passages (then the model faithfully answers from
wrong context), and the model misreads correct passages. That's why the system
shows sources with scores (transparency) and why faithfulness is *measured* by
the judge rather than assumed.

### RAG vs fine-tuning
The industry split: fine-tuning teaches *behavior* (tone, format, task skill);
RAG supplies *knowledge* (facts that change). Baking fast-changing facts into
weights is expensive and goes stale; updating a document corpus is a file copy.
For "answer questions over my docs," RAG wins on cost, freshness, and
citability.

### Eval metrics, precisely
- **hit_rate@k**: fraction of questions whose gold document appears in the
  top-k. Binary per question, then averaged.
- **MRR** (mean reciprocal rank): (1/n) * sum(1/rank_i), where rank_i is the
  1-indexed position of the gold doc (a miss contributes 0). Example: ranks 1,
  2, miss -> (1 + 1/2 + 0)/3 = 0.5. MRR rewards putting the right doc *first*,
  not merely somewhere in the top-k.
- **Keyword coverage**: |expected keywords found in answer| / |expected
  keywords|. Crude — substring match, no semantics — but deterministic and
  key-free.
- **LLM-as-judge faithfulness**: the judge LLM answers YES/NO to "is every
  claim supported by the context?" Powerful, but the judge can itself be wrong
  or biased toward fluent answers, so it's the *last* metric layer, applied on
  top of deterministic ones — never the only one.

### Extractive vs generative answers
Extractive (our fallback): return retrieved passages verbatim. Faithful by
construction — it cannot invent words — but clunky, and it can't synthesize
across passages. Generative (LLM): fluent and synthesizing, but must be
constrained and checked. The pipeline supports both behind the same interface,
which is why the eval harness can compare them apples-to-apples.

---
## 3. Directory structure

```
rag-eval-app/
├── README.md                 # quickstart, architecture, trade-offs
├── TECHNICAL_DEEPDIVE.md     # this document
├── requirements.txt          # numpy, scikit-learn, pypdf, requests, streamlit
├── app.py                    # Streamlit UI (serving path)
├── run_eval.py               # CLI: build index, run eval harness
├── rag/                      # the pipeline package
│   ├── __init__.py           # empty; marks rag/ as a package
│   ├── config.py             # paths, chunk sizes, env-var settings
│   ├── ingest.py             # load .txt/.md/.pdf → word-aware chunks
│   ├── embeddings.py         # sentence-transformers OR TF-IDF backend
│   ├── store.py              # numpy cosine-similarity vector store
│   ├── retrieve.py           # query embedding → top-k search
│   ├── generate.py           # grounded LLM call / extractive fallback
│   └── evaluate.py           # hit_rate@k, MRR, keyword coverage, LLM judge
├── data/
│   ├── docs/                 # corpus: 6 finance .txt files (swap your own in)
│   ├── eval_questions.json   # 18 gold questions (3 per doc)
│   └── store/                # GENERATED: vectors.npy, meta.json, embedder.txt
└── .venv/                    # local dev virtualenv (not shipped in the zip)
```

## 4. Execution flow, end to end

### 4.1 Index build — `python run_eval.py --rebuild`

1. `build_chunks()` (ingest.py) scans `data/docs/`, loads every `.txt`,
   `.md`, `.pdf` (PDFs via pypdf's `PdfReader`, page text concatenated with
   newlines), and splits each document into chunks. Our 6 docs produce
   **20 chunks**.
2. `Embedder()` (embeddings.py) picks a backend: it tries to import
   `sentence_transformers`; if that fails it falls back to TF-IDF. In our
   verified run it used TF-IDF.
3. For TF-IDF, `embedder.fit(texts)` fits the vectorizer on the corpus
   (max 20,000 features, 1–2 grams, English stop words removed). Then
   `embedder.embed(texts)` converts every chunk to a vector and **L2-normalizes**
   each one (divides by its length), so cosine similarity later is just a dot
   product.
4. `VectorStore.add()` stacks the 20 vectors into one numpy matrix and keeps
   parallel lists of chunk ids (`factor_investing.txt#0`, …) and metadata
   (doc name, chunk index, raw text).
5. `store.save()` writes `data/store/vectors.npy` (the matrix) and
   `data/store/meta.json` (ids + metadata). `run_eval.py` additionally writes
   `embedder.txt` recording which backend was used, so serving uses the same one.

### 4.2 Serving a question — Streamlit app

1. On first load, `app.py` calls `serving_retriever()` (run_eval.py), which
   reads `embedder.txt`, rebuilds the matching `Embedder` (for TF-IDF it
   **re-fits the vectorizer on the corpus** — the chunking is deterministic,
   so the vocabulary matches the indexed vectors exactly), and loads the
   saved `VectorStore`. The retriever is cached in `st.session_state` so this
   happens once.
2. You type a question and press Ask. `retriever.retrieve(query, k)` embeds
   the single query string with the same embedder, then `store.search()`
   computes `scores = matrix @ query_vector` — one matrix multiply gives the
   cosine similarity of the query against all 20 chunks — sorts descending,
   and returns the top-k (default 4) as `(chunk_id, score, metadata)` tuples.
3. `answer_question()` (generate.py) builds the prompt: a system instruction
   ("answer using ONLY the context; if the answer isn't there, say exactly
   *I don't know based on the provided documents*") plus the retrieved
   passages formatted as `[source: docname, score: 0.842]` blocks, plus your
   question.
4. If `RAG_LLM_BASE_URL` and `RAG_LLM_MODEL` are set, `LLMClient.complete()`
   POSTs to `{base}/chat/completions` in OpenAI's format (this is why Ollama
   at `http://localhost:11434/v1`, OpenAI, vLLM all work unchanged) with
   temperature 0.1 and returns the reply. If no LLM is configured, the
   **extractive fallback** returns the top passage verbatim, clearly labeled.
5. The UI shows the answer, then an expander per passage with its source
   document and similarity score — so every answer is auditable.

### 4.3 Evaluation — `python run_eval.py`

For each of the 18 gold questions:

1. **Retrieve** top-k (k=4). Find the rank of the gold document (1 = top hit).
   A question "hits" if its gold doc appears anywhere in the top-k.
2. **Generate** an answer (LLM or extractive, whichever is configured).
3. **Keyword coverage**: what fraction of the question's expected keywords
   appear in the answer (case-insensitive substring match).
4. Optionally (`--judge`, needs an LLM): the **faithfulness judge** asks the
   LLM a single YES/NO question — "is every claim in this answer supported by
   the context?" — and records the verdict.

Aggregate metrics printed at the end:

- **hit_rate@k** = fraction of questions whose gold doc was retrieved.
- **MRR** (mean reciprocal rank) = average of 1/rank — rewards putting the
  right doc first, not just in the top-k.
- **avg_coverage** = mean keyword coverage across questions.
- **faithfulness** = fraction of judged answers with no unsupported claims.

Our verified run (TF-IDF embeddings, extractive answers, k=4):
hit_rate@k = **1.00**, MRR = **0.972**, avg coverage = **0.741**.
Honest context for interviews: that 1.00 is the TF-IDF fallback on a small,
well-separated 6-document corpus — not neural embeddings at scale. The
coverage misses (e.g. q07, q16, q18 at 0.00) are extractive-mode artifacts:
the top chunk didn't contain every expected keyword, which is exactly what a
generative LLM pass would improve.

---

## 5. File by file

### `rag/config.py` — all knobs in one place

Centralizes paths (`data/docs`, `data/store`, eval JSON), chunking parameters
(`CHUNK_SIZE = 500` chars, `CHUNK_OVERLAP = 80`), default `TOP_K = 4`, and every
environment variable: `RAG_EMBEDDER` (auto/st/tfidf), `RAG_ST_MODEL`
(default `all-MiniLM-L6-v2`), `RAG_LLM_BASE_URL`, `RAG_LLM_API_KEY`,
`RAG_LLM_MODEL`. Nothing else in the codebase hardcodes these values, so
tuning is a config change, not a code change.

### `rag/ingest.py` — documents → chunks

- `load_documents()`: globs `data/docs/*`, dispatches on extension —
  `.txt`/`.md` read as UTF-8, `.pdf` via pypdf (`PdfReader`, `page.extract_text()`
  per page, joined with newlines; `or ""` guards pages with no extractable text).
  Returns `(doc_name, text)` pairs, sorted for determinism.
- `chunk_text()`: **word-aware** character chunking. It accumulates whole words
  until adding the next word would exceed 500 chars, emits the chunk, then
  carries trailing words totaling ≤ 80 chars into the next chunk as overlap.
  Word-aware matters: naive character slicing can split "price-to-earnings"
  mid-token, which poisons both embeddings and the eval's keyword matching.
  Overlap matters: definitions spanning a boundary stay retrievable.
- `build_chunks()`: returns a list of dicts `{id, doc, chunk_index, text}`
  with ids like `factor_investing.txt#0`. Deterministic: same corpus always
  yields the same chunks in the same order — this is what makes the TF-IDF
  refit-at-serve-time trick valid.

### `rag/embeddings.py` — text → vectors

- `Embedder(backend)`: `"auto"` tries `import sentence_transformers`; success →
  `"st"`, ImportError → `"tfidf"`. The backend choice is explicit and logged.
- `"st"` path: loads `all-MiniLM-L6-v2` (384-dimensional, fast, good
  general-purpose quality) and encodes with `convert_to_numpy=True`.
- `"tfidf"` path: `TfidfVectorizer(max_features=20000, ngram_range=(1,2),
  stop_words="english")`. Bigrams (`ngram_range=(1,2)`) capture phrases like
  "expected loss"; stop-word removal drops noise like "the". Must call
  `.fit(corpus)` before `.embed()` — enforced with a RuntimeError.
- `embed()` **L2-normalizes** every vector. This is the detail that makes the
  store trivial: for unit vectors, cosine similarity = dot product, so search
  is a single matrix multiplication with no division at query time.

### `rag/store.py` — the vector index

A minimal vector database on numpy, deliberately: at 20 chunks (and even
20,000), brute-force search is instant and has zero dependencies or moving
parts. `add()` stacks vectors with `np.vstack`; `search()` does
`scores = self.vectors @ q` and `argsort()[::-1][:k]`. `save()`/`load()`
persist `vectors.npy` + `meta.json` (ids and metadata in parallel arrays).
The class is a clean seam: swapping in Chroma or pgvector later means
reimplementing these four methods, touching nothing else. That's a good
interview talking point — you designed the interface before you needed the
scale.

### `rag/retrieve.py` — query → passages

Thin glue: `Retriever` holds an `Embedder` and a `VectorStore`;
`retrieve(query, k)` embeds the one query string and delegates to
`store.search()`. Kept separate from generation so the eval harness can test
retrieval in isolation — a pipeline you can unit-test stage by stage.

### `rag/generate.py` — passages → answer

- `LLMClient`: a ~20-line wrapper over any **OpenAI-compatible**
  `/chat/completions` endpoint. `configured` is true only when both base URL
  and model are set. `complete()` sends system + user messages at temperature
  0.1 (low temperature = less creativity = fewer hallucinations) with a
  120-second timeout, and extracts `choices[0].message.content`.
- `SYSTEM_PROMPT`: the anti-hallucination contract — "answer using ONLY the
  context; if the answer is not in the context, say exactly: *I don't know
  based on the provided documents.*" Grounded generation lives or dies on
  this instruction.
- `build_context()`: formats passages as `[source: doc, score: 0.xxx]`
  blocks separated by `---`. Showing scores isn't just UI sugar — during
  development it reveals when the retriever is uncertain (flat scores) versus
  confident (one clear winner).
- `answer_question()`: tries the LLM; otherwise the **extractive fallback**
  returns the top passage verbatim, explicitly labeled as such. This is why
  the whole pipeline — and the eval harness — runs with zero API keys.

### `rag/evaluate.py` — the harness

- `load_eval_questions()`: reads the 18 gold questions. Each has `id`,
  `question`, `gold_doc`, and `expected_keywords` (3–5 terms that a correct
  answer should contain).
- `keyword_coverage()`: case-insensitive substring hit rate of expected
  keywords in the generated answer. Crude but deterministic and key-free —
  it measures whether the answer says the right things, not how pretty it is.
- `judge_faithfulness()`: optional LLM-as-judge. It prompts the LLM with the
  context, question, and answer and demands exactly one word: YES (every claim
  supported) or NO. Single-word verdicts avoid parsing headaches.
- `run_eval()`: loops questions → retrieve → rank → answer → coverage →
  optional judge; aggregates hit_rate@k, MRR, avg coverage, faithfulness.
- `print_report()`: human-readable summary plus a per-question line
  (`[HIT ] rank= 1 cov=1.00 q01: ...`) so failures are immediately visible —
  when a question misses, you see which one and can inspect that chunk boundary
  or phrasing.

### `run_eval.py` — the CLI

- `--rebuild`: force re-ingest and re-embed (otherwise it reuses the saved
  index if `vectors.npy` exists).
- `build_index()`: chunks → fit → embed → `VectorStore.add()` → save →
  writes `embedder.txt` so serving uses the identical backend.
- `serving_retrieving()`: reads `embedder.txt`; for TF-IDF, **refits the
  vectorizer on the corpus** (the vectorizer itself isn't persisted — refit is
  exact because chunking is deterministic, and it keeps the saved state to
  two small files); loads the saved store; returns a `Retriever`.
- `--judge`: also runs the LLM faithfulness judge (requires an LLM configured).

### `app.py` — the Streamlit UI

- Builds the retriever once per session (`st.session_state`) — rebuilding
  embeddings on every question would be wasteful.
- Sidebar shows live status: which embedder backend is active, whether an LLM
  is configured (and which model) or that you're in extractive mode, plus a
  top-k slider (1–8) so you can feel the precision/recall trade-off
  interactively: small k = focused but may miss; large k = more context but
  noisier and more tokens.
- Main flow: text input → Ask → spinner while retrieving → spinner while
  generating → answer rendered with its mode (`llm`/`extractive`) → each
  retrieved passage in an expander showing source doc and similarity score.
  If no index exists (e.g. a fresh deploy), it builds one automatically from
  `data/docs/` on first run instead of crashing.

### `data/` — corpus, eval set, generated index

- `data/docs/`: 6 finance `.txt` files (~250–450 words each): factor
  investing, 10-K basics, P/E ratios, backtesting pitfalls, credit risk
  (PD/LGD/EAD), dollar-cost averaging. Chosen because they're in Ben's domain —
  he can discuss every document's content in an interview. Swap in any
  `.txt`/`.md`/`.pdf` files to query your own corpus.
- `data/eval_questions.json`: 18 questions, 3 per doc, each with a gold doc
  and 3–5 expected keywords. This file *is* the quality bar — extend it as
  the corpus grows.
- `data/store/` (generated, not in the zip): `vectors.npy`, `meta.json`,
  `embedder.txt`. Regenerable at any time; safe to delete.

### `requirements.txt`

`numpy`, `scikit-learn` (TF-IDF), `pypdf` (PDF ingestion), `requests` (LLM
HTTP calls), `streamlit` (UI). `sentence-transformers` is commented out as
optional — install it to switch the default backend from TF-IDF to neural
embeddings with no code changes (`RAG_EMBEDDER=st` or just install it and
`auto` picks it up).

---

## 6. Configuration reference

| Variable | Default | Effect |
|---|---|---|
| `RAG_EMBEDDER` | `auto` | `st` = sentence-transformers, `tfidf` = TF-IDF fallback |
| `RAG_ST_MODEL` | `all-MiniLM-L6-v2` | Hugging Face model for the `st` backend |
| `RAG_LLM_BASE_URL` | (unset) | e.g. `http://localhost:11434/v1` for Ollama |
| `RAG_LLM_API_KEY` | (unset) | sent as Bearer token if the endpoint needs one |
| `RAG_LLM_MODEL` | (unset) | e.g. `llama3.1`, `gpt-4o-mini` |
| chunk size / overlap / top-k | 500 / 80 / 4 | edit in `rag/config.py` |

With nothing set, the app runs fully offline: TF-IDF retrieval + extractive
answers + deterministic evals. Set the three LLM vars (Ollama needs only base
URL + model, no key) for full generative RAG.

## 7. How to run it

```bash
pip install -r requirements.txt
python run_eval.py --rebuild   # build index, run the 18-question eval
streamlit run app.py           # ask questions in the browser
```

To make it yours: drop documents into `data/docs/`, add questions with gold
docs and keywords to `data/eval_questions.json`, re-run `--rebuild`, and watch
hit_rate@k move. To upgrade retrieval: `pip install sentence-transformers`.

## 8. If an interviewer asks "what would you do next?"

- Swap the numpy store for pgvector/FAISS and an ANN index (HNSW) past ~1M chunks.
- Hybrid retrieval: BM25 + vector scores fused (reciprocal rank fusion).
- Reranker (cross-encoder) on the top-20 before the top-4 cut.
- Bigger eval set, adversarial questions (unanswerable ones — the "I don't
  know" path is currently untested), and human spot-checks.
- Ingestion as a batch pipeline with incremental updates instead of full rebuilds.
