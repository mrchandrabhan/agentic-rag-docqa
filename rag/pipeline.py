"""DocQA: ingest documents once, then ask questions through the agentic graph."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .chunking import chunk_pages
from .embeddings import make_embedder
from .graph import build_graph
from .llm import make_llm
from .loaders import load_files
from .retrieval import hybrid_search
from .stores import make_store

# Minimum semantic similarity depends on the embedder's score scale. Hybrid
# retrieval also admits BM25 lexical matches, then uses reciprocal-rank fusion.
SEMANTIC_MIN_SCORE = {"minilm": 0.25, "hash": 0.08}
HYBRID_MIN_SCORE = 0.012


class DocQA:
    def __init__(self, embedder="minilm", store="faiss", llm=None, k: int = 4,
                 chunk_size: int = 800, chunk_overlap: int = 120, log_path: str | None = None):
        self.embedder = make_embedder(embedder) if isinstance(embedder, str) else embedder
        kind = "hash" if self.embedder.model_name.startswith("hash") else "minilm"
        self.semantic_min_score = SEMANTIC_MIN_SCORE[kind]
        self.min_score = HYBRID_MIN_SCORE
        self.store_kind = store
        self.store = make_store(store, self.embedder.dim)
        self.llm = llm or make_llm("extractive")
        self.k, self.chunk_size, self.chunk_overlap = k, chunk_size, chunk_overlap
        self.log_path = Path(log_path) if log_path else None
        self.sources: list[str] = []
        self._graph = None

    def ingest(self, paths) -> dict:
        pages = load_files(paths)
        chunks = chunk_pages(pages, self.chunk_size, self.chunk_overlap)
        offset = len(self.store)
        for c in chunks:
            c.id += offset
        if chunks:
            self.store.add(chunks, self.embedder.embed([c.text for c in chunks]))
        self.sources += sorted({p.source for p in pages} - set(self.sources))
        self._graph = None
        return {
            "files": len({p.source for p in pages}),
            "pages": len(pages),
            "chunks": len(chunks),
            "characters": sum(len(page.text) for page in pages),
        }

    def set_llm(self, llm) -> None:
        self.llm, self._graph = llm, None

    @property
    def graph(self):
        if self._graph is None:
            self._graph = build_graph(
                self.store,
                self.embedder,
                self.llm,
                k=self.k,
                min_score=self.min_score,
                semantic_min_score=self.semantic_min_score,
            )
        return self._graph

    def ask(self, question: str) -> dict:
        t0 = time.perf_counter()
        out = self.graph.invoke({"question": question, "attempts": 0, "trace": []})
        result = {
            "question": question,
            "answer": out["answer"],
            "citations": out.get("citations", []),
            "trace": out["trace"],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(
                    {k: v for k, v in result.items() if k != "citations"}
                    | {"sources": [f"{c['source']}:p{c['page']}" for c in result["citations"]]}
                ) + "\n")
        return result

    def retrieve(self, question: str, k: int | None = None):
        vector = self.embedder.embed([question])[0]
        return hybrid_search(
            self.store.chunks,
            self.store.search,
            vector,
            question,
            k or self.k,
            self.semantic_min_score,
        )
