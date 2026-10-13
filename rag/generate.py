"""Answer generation: grounded LLM call (OpenAI-compatible) or extractive fallback."""
import requests

from . import config

SYSTEM_PROMPT = (
    "Answer the question using ONLY the context below. Be concise. "
    "If the answer is not in the context, say exactly: "
    "\"I don't know based on the provided documents.\""
)


class LLMClient:
    """Thin wrapper over any OpenAI-compatible /chat/completions endpoint
    (OpenAI, Ollama at http://localhost:11434/v1, vLLM, ...)."""

    def __init__(self, base_url=None, api_key=None, model=None):
        self.base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
        self.api_key = api_key or config.LLM_API_KEY
        self.model = model or config.LLM_MODEL

    @property
    def configured(self):
        return bool(self.base_url and self.model)

    def complete(self, system, user):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={"model": self.model,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                  "temperature": 0.1},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def build_context(chunks):
    """chunks: list of (id, score, meta) from the retriever."""
    parts = []
    for cid, score, meta in chunks:
        parts.append(f"[source: {meta['doc']}, score: {score:.3f}]\n{meta['text']}")
    return "\n\n---\n\n".join(parts)


def answer_question(query, chunks, llm=None):
    """Return (answer, mode). mode is 'llm' or 'extractive'."""
    llm = llm or LLMClient()
    if llm.configured:
        prompt = f"Context:\n{build_context(chunks)}\n\nQuestion: {query}"
        try:
            return llm.complete(SYSTEM_PROMPT, prompt), "llm"
        except Exception as exc:
            # Never crash the app on an LLM error (bad/retired model name,
            # rate limit, network): degrade to the extractive answer and
            # say why, so the failure is visible instead of a traceback.
            top = chunks[0][2]["text"]
            answer = (f"(LLM call failed: {exc} — showing the top retrieved "
                      f"passage instead.)\n\n" + top[:800])
            return answer, "extractive"
    # Extractive fallback: no LLM configured — return the top chunks verbatim.
    top = chunks[0][2]["text"]
    answer = ("Based on the retrieved documents (no LLM configured, "
              "showing top passage):\n\n" + top[:800])
    return answer, "extractive"
