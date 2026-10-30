"""Hybrid lexical + vector retrieval.

Vector similarity finds paraphrases; BM25-style lexical ranking protects exact
names, technologies, acronyms, and other rare terms in resumes and runbooks.
The combined ranking uses reciprocal-rank fusion so the two score scales do
not need to be comparable.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from .chunking import Chunk


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "how",
    "i", "in", "into", "is", "it", "its", "may", "of", "on", "or", "our",
    "should", "that", "the", "their", "them", "then", "there", "these", "this",
    "those", "to", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "you", "your",
}


def tokenize(text: str) -> list[str]:
    # Keep useful technical punctuation in terms such as C++, C#, .NET and CI/CD.
    terms = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9+#./-]*", text.lower())
    normalized = []
    for term in terms:
        term = term.strip("./-")
        if len(term) > 1 and term not in STOP_WORDS:
            normalized.append(term)
    return normalized


def _ordered_chunks(chunks: list[Chunk] | dict[str, Chunk]) -> list[Chunk]:
    if isinstance(chunks, dict):
        return list(chunks.values())
    return list(chunks)


def hybrid_search(
    chunks: list[Chunk] | dict[str, Chunk],
    vector_search,
    query_vector,
    query: str,
    k: int,
    semantic_min_score: float,
    *,
    rrf_constant: int = 60,
) -> list[tuple[Chunk, float]]:
    """Return top-k chunks ranked by semantic and BM25-style lexical evidence."""
    all_chunks = _ordered_chunks(chunks)
    if not all_chunks or k <= 0:
        return []

    semantic_hits = vector_search(query_vector, min(len(all_chunks), max(k * 8, k)))
    semantic_ranks: dict[int, int] = {}
    for rank, (chunk, score) in enumerate(semantic_hits, start=1):
        if score >= semantic_min_score:
            semantic_ranks[chunk.id] = rank

    query_terms = list(dict.fromkeys(tokenize(query)))
    lexical_ranks: dict[int, int] = {}
    if query_terms:
        tokenized = [tokenize(chunk.text) for chunk in all_chunks]
        doc_freq = Counter(
            term for term in query_terms
            if any(term in doc_terms for doc_terms in tokenized)
        )
        total_docs = len(tokenized)
        avg_length = sum(len(doc_terms) for doc_terms in tokenized) / max(total_docs, 1)
        scored: list[tuple[float, int, int]] = []

        for chunk, doc_terms in zip(all_chunks, tokenized):
            frequencies = Counter(doc_terms)
            matched = [term for term in query_terms if frequencies[term] > 0]
            if not matched:
                continue

            weighted_coverage = 0.0
            total_weight = 0.0
            bm25_score = 0.0
            length_norm = 1 - 0.75 + 0.75 * len(doc_terms) / max(avg_length, 1.0)
            for term in query_terms:
                df = doc_freq.get(term, 0)
                if not df:
                    continue
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                total_weight += idf
                if frequencies[term]:
                    weighted_coverage += idf
                    tf = frequencies[term]
                    bm25_score += idf * (tf * 2.2) / (tf + 1.2 * length_norm)

            coverage = weighted_coverage / total_weight if total_weight else 0.0
            # One distinctive exact term (e.g. Kubernetes) is useful; common
            # single-word matches are not enough to retrieve an unrelated chunk.
            distinctive_single = (
                len(matched) == 1
                and len(matched[0]) >= 5
                and doc_freq.get(matched[0], total_docs) < total_docs
            )
            if coverage >= 0.25 or len(matched) >= 2 or distinctive_single:
                scored.append((bm25_score, chunk.id, len(matched)))

        scored.sort(key=lambda item: (-item[0], item[1]))
        for rank, (_, chunk_id, _) in enumerate(scored[: min(len(scored), max(k * 8, k))], start=1):
            lexical_ranks[chunk_id] = rank

    if not semantic_ranks and not lexical_ranks:
        return []

    by_id = {chunk.id: chunk for chunk in all_chunks}
    fused: list[tuple[Chunk, float]] = []
    for chunk_id in set(semantic_ranks) | set(lexical_ranks):
        score = 0.0
        if chunk_id in semantic_ranks:
            score += 1.0 / (rrf_constant + semantic_ranks[chunk_id])
        if chunk_id in lexical_ranks:
            score += 1.0 / (rrf_constant + lexical_ranks[chunk_id])
        fused.append((by_id[chunk_id], score))

    fused.sort(key=lambda item: (-item[1], item[0].id))
    return fused[:k]
