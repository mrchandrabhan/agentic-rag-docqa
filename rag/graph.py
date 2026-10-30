r"""Hybrid retrieval -> grade -> (generate/cite | rewrite | abstain)."""
from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .chunking import Chunk
from .retrieval import hybrid_search

ANSWER_SYSTEM = (
    "You answer questions using ONLY the numbered context passages. "
    "Cite every claim with the passage number in square brackets, like [1] or [2][3]. "
    "If the passages don't contain the answer, reply exactly: I couldn't find that in the uploaded documents. "
    "Be concise: 1-4 sentences."
)
NO_ANSWER = "I couldn't find that in the uploaded documents."


class RAGState(TypedDict, total=False):
    question: str
    query: str
    hits: list[tuple[Chunk, float]]
    relevant: list[tuple[Chunk, float]]
    attempts: int
    answer: str
    citations: list[dict]
    trace: list[str]


def build_graph(
    store,
    embedder,
    llm,
    k: int = 4,
    min_score: float = 0.012,
    semantic_min_score: float = 0.25,
    max_rewrites: int = 1,
):
    def retrieve(state: RAGState) -> RAGState:
        query = state.get("query") or state["question"]
        vector = embedder.embed([query])[0]
        hits = hybrid_search(
            store.chunks,
            store.search,
            vector,
            query,
            k,
            semantic_min_score,
        )
        if hits:
            top_chunk, top_score = hits[0]
            top = f"{top_score:.3f} ({top_chunk.source}:p{top_chunk.page})"
        else:
            top = "n/a"
        return {
            "query": query,
            "hits": hits,
            "trace": state.get("trace", []) + [
                f"retrieve: {len(hits)} hybrid chunks; top {top}"
            ],
        }

    def grade(state: RAGState) -> RAGState:
        relevant = [(c, s) for c, s in state["hits"] if s >= min_score]
        return {
            "relevant": relevant,
            "trace": state["trace"] + [
                f"grade: {len(relevant)}/{len(state['hits'])} passed hybrid_min_score={min_score}"
            ],
        }

    def route(state: RAGState) -> str:
        if state["relevant"]:
            return "generate"
        return "rewrite" if state.get("attempts", 0) < max_rewrites else "no_answer"

    def rewrite(state: RAGState) -> RAGState:
        new_q = llm.rewrite(state["question"]).strip() or state["question"]
        return {
            "query": new_q,
            "attempts": state.get("attempts", 0) + 1,
            "trace": state["trace"] + [f"rewrite: '{new_q}'"],
        }

    def generate(state: RAGState) -> RAGState:
        ctx = "\n".join(f"[{i}] {c.text}" for i, (c, _) in enumerate(state["relevant"], 1))
        answer = llm.complete(ANSWER_SYSTEM, f"CONTEXT:\n{ctx}\n\nQUESTION: {state['question']}")
        return {"answer": answer, "trace": state["trace"] + [f"generate: {llm.name}"]}

    def cite(state: RAGState) -> RAGState:
        used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", state["answer"])})
        rel = state["relevant"]
        valid = [n for n in used if 1 <= n <= len(rel)]
        if state["answer"].startswith(NO_ANSWER[:20]):
            valid = []
        cites = [
            {
                "n": n,
                "source": rel[n - 1][0].source,
                "page": rel[n - 1][0].page,
                "score": round(rel[n - 1][1], 3),
                "text": rel[n - 1][0].text,
            }
            for n in valid
        ]
        dropped = len(used) - len(valid)
        note = f"cite: {len(cites)} citations" + (f", dropped {dropped} invalid" if dropped else "")
        return {"citations": cites, "trace": state["trace"] + [note]}

    def no_answer(state: RAGState) -> RAGState:
        return {
            "answer": NO_ANSWER,
            "citations": [],
            "trace": state["trace"] + ["no_answer: nothing relevant after rewrite"],
        }

    g = StateGraph(RAGState)
    for name, fn in [
        ("retrieve", retrieve),
        ("grade", grade),
        ("rewrite", rewrite),
        ("generate", generate),
        ("cite", cite),
        ("no_answer", no_answer),
    ]:
        g.add_node(name, fn)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        route,
        {"generate": "generate", "rewrite": "rewrite", "no_answer": "no_answer"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "cite")
    g.add_edge("cite", END)
    g.add_edge("no_answer", END)
    return g.compile()
