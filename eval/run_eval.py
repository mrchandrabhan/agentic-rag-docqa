"""Labeled evaluation: retrieval hit@k / MRR, answer accuracy, citation precision, abstention.

python -m eval.run_eval                      # MiniLM + FAISS + extractive answers
python -m eval.run_eval --embedder hash      # offline (what CI runs)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag import DocQA
from rag.graph import NO_ANSWER
from rag.llm import make_llm

HERE = Path(__file__).parent


def hit_rank(hits, source, keyword) -> int | None:
    for rank, (chunk, _) in enumerate(hits, 1):
        if chunk.source == source and keyword.lower() in chunk.text.lower():
            return rank
    return None


def evaluate(embedder="minilm", store="faiss", k=4, chunk_size=300, chunk_overlap=50) -> dict:
    qa = DocQA(embedder=embedder, store=store, llm=make_llm(), k=k,
               chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    qa.ingest(sorted((HERE / "corpus").iterdir()))
    rows = [json.loads(l) for l in (HERE / "dataset.jsonl").read_text().splitlines() if l.strip()]

    answerable = [r for r in rows if r["source"]]
    unanswerable = [r for r in rows if not r["source"]]
    hits = mrr = correct = cited = cited_ok = abstained = 0
    failures = []

    for r in answerable:
        rank = hit_rank(qa.retrieve(r["question"], k), r["source"], r["keyword"])
        hits += rank is not None
        mrr += 1 / rank if rank else 0
        res = qa.ask(r["question"])
        ok = r["keyword"].lower() in res["answer"].lower()
        correct += ok
        for c in res["citations"]:
            cited += 1
            cited_ok += c["source"] == r["source"]
        if not (rank and ok):
            failures.append({"id": r["id"], "rank": rank, "answer": res["answer"][:160]})

    for r in unanswerable:
        res = qa.ask(r["question"])
        abstained += res["answer"] == NO_ANSWER
        if res["answer"] != NO_ANSWER:
            failures.append({"id": r["id"], "expected": "abstain", "answer": res["answer"][:160]})

    n = len(answerable)
    return {
        "config": {"embedder": qa.embedder.model_name, "store": store, "llm": qa.llm.name, "k": k, "chunk_size": chunk_size},
        "cases": len(rows),
        f"hit@{k}": round(hits / n, 3),
        "mrr": round(mrr / n, 3),
        "answer_accuracy": round(correct / n, 3),
        "citation_precision": round(cited_ok / cited, 3) if cited else None,
        "abstention_accuracy": round(abstained / len(unanswerable), 3) if unanswerable else None,
        "failures": failures,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="minilm", choices=["minilm", "hash"])
    ap.add_argument("--store", default="faiss", choices=["faiss", "chroma"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=300)
    a = ap.parse_args()
    print(json.dumps(evaluate(a.embedder, a.store, a.k, a.chunk_size), indent=2))
