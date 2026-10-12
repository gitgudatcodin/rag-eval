"""Eval harness: deterministic retrieval + answer metrics over gold questions.

Metrics:
  hit_rate@k  - fraction of questions where the gold doc is in the top-k
  mrr         - mean reciprocal rank of the gold doc
  coverage    - fraction of expected answer keywords present in the answer
  faithful    - (optional, needs LLM) LLM-judge: no unsupported claims
"""
import json

from . import config
from .generate import LLMClient, answer_question


def load_eval_questions(path=None):
    with open(path or config.EVAL_PATH) as f:
        return json.load(f)


def keyword_coverage(answer, keywords):
    a = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in a)
    return hits / max(len(keywords), 1)


def judge_faithfulness(question, answer, context, llm):
    prompt = (
        "You are checking whether an ANSWER is faithful to the CONTEXT.\n"
        "Reply with exactly one word: YES if every claim in the answer is "
        "supported by the context, NO if any claim is unsupported or contradicted.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER: {answer}"
    )
    verdict = llm.complete("You are a strict evaluation judge.", prompt)
    return "YES" in verdict.upper()


def run_eval(retriever, questions=None, k=None, llm=None, judge=False):
    from .generate import build_context
    questions = questions or load_eval_questions()
    k = k or config.TOP_K
    llm = llm or LLMClient()
    rows = []
    for q in questions:
        hits = retriever.retrieve(q["question"], k=k)
        docs = [meta["doc"] for _, _, meta in hits]
        rank = next((i + 1 for i, d in enumerate(docs) if d == q["gold_doc"]), None)
        answer, mode = answer_question(q["question"], hits, llm=llm)
        cov = keyword_coverage(answer, q["expected_keywords"])
        faithful = None
        if judge and llm.configured:
            faithful = judge_faithfulness(q["question"], answer,
                                          build_context(hits), llm)
        rows.append({"id": q["id"], "question": q["question"],
                     "gold_doc": q["gold_doc"], "rank": rank,
                     "hit": rank is not None, "coverage": round(cov, 3),
                     "faithful": faithful, "mode": mode})
    n = len(rows)
    summary = {
        "n": n, "k": k, "mode": rows[0]["mode"] if rows else None,
        "hit_rate@k": round(sum(r["hit"] for r in rows) / n, 3),
        "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3),
        "avg_coverage": round(sum(r["coverage"] for r in rows) / n, 3),
    }
    judged = [r for r in rows if r["faithful"] is not None]
    if judged:
        summary["faithfulness"] = round(
            sum(r["faithful"] for r in judged) / len(judged), 3)
    return summary, rows


def print_report(summary, rows):
    print(f"\n== RAG eval: {summary['n']} questions, k={summary['k']}, mode={summary['mode']} ==")
    print(f"hit_rate@k : {summary['hit_rate@k']}")
    print(f"mrr        : {summary['mrr']}")
    print(f"avg_cover  : {summary['avg_coverage']}")
    if "faithfulness" in summary:
        print(f"faithful   : {summary['faithfulness']}")
    print("\nper-question:")
    for r in rows:
        flag = "HIT " if r["hit"] else "MISS"
        print(f"  [{flag}] rank={str(r['rank']):>4} cov={r['coverage']:.2f}  {r['id']}: {r['question'][:60]}")
