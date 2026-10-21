"""Embedding backends. All return L2-normalized float32 vectors so inner product == cosine similarity."""
from __future__ import annotations

import hashlib
import re

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (v / norms).astype("float32")


class SentenceTransformerEmbedder:
    """Default semantic embedder (all-MiniLM-L6-v2, 384-d), runs on CPU."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
        return _normalize(np.asarray(vecs, dtype="float32"))


class HashEmbedder:
    """Dependency-free lexical embedder (hashed unigrams + bigrams).

    Used for offline tests and CI, and as a fallback when the model can't be downloaded.
    """

    def __init__(self, dim: int = 1024):
        self.model_name = f"hash-{dim}"
        self.dim = dim

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in self._tokens(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        return _normalize(out)


def make_embedder(kind: str = "minilm"):
    if kind == "hash":
        return HashEmbedder()
    try:
        return SentenceTransformerEmbedder()
    except Exception as exc:  # model download blocked / package missing
        print(f"[embeddings] falling back to HashEmbedder: {exc}")
        return HashEmbedder()
