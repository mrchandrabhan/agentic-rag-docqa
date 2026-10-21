"""Vector stores behind one small interface, so FAISS and ChromaDB are swappable."""
from __future__ import annotations

import numpy as np

from .chunking import Chunk


class FaissStore:
    name = "faiss"

    def __init__(self, dim: int):
        import faiss

        self.index = faiss.IndexFlatIP(dim)  # cosine similarity on normalized vectors
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        scores, ids = self.index.search(vector.reshape(1, -1), min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def __len__(self) -> int:
        return len(self.chunks)


class ChromaStore:
    name = "chroma"

    def __init__(self, dim: int):
        import chromadb

        self.client = chromadb.EphemeralClient()
        self.col = self.client.create_collection(f"docs_{id(self)}", metadata={"hnsw:space": "cosine"})
        self.chunks: dict[str, Chunk] = {}

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        ids = [str(c.id) for c in chunks]
        self.col.add(ids=ids, embeddings=vectors.tolist(), documents=[c.text for c in chunks])
        self.chunks.update(zip(ids, chunks))

    def search(self, vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        res = self.col.query(query_embeddings=[vector.tolist()], n_results=min(k, len(self.chunks)))
        return [(self.chunks[i], 1.0 - float(d)) for i, d in zip(res["ids"][0], res["distances"][0])]

    def __len__(self) -> int:
        return len(self.chunks)


def make_store(kind: str, dim: int):
    return {"faiss": FaissStore, "chroma": ChromaStore}[kind](dim)
