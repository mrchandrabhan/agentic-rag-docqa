"""Embedding backends. All return L2-normalized float32 vectors so inner product == cosine similarity."""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (v / norms).astype("float32")


MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE = os.getenv("MODEL_CACHE", str(Path(__file__).resolve().parent.parent / ".models"))


class FastEmbedEmbedder:
    """Default semantic embedder: all-MiniLM-L6-v2 (384-d) on ONNX Runtime via fastembed.

    Same model as sentence-transformers but no PyTorch, so it fits a 512 MB free-tier instance.
    """

    def __init__(self, model_name: str = MINILM):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name, cache_dir=MODEL_CACHE)
        self.dim = 384

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = list(self.model.embed(texts, batch_size=32))
        return _normalize(np.asarray(vecs, dtype="float32"))


class SentenceTransformerEmbedder:
    """Same MiniLM model on PyTorch (used if fastembed isn't installed)."""

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
    for cls in (FastEmbedEmbedder, SentenceTransformerEmbedder):
        try:
            return cls()
        except Exception as exc:  # package missing / model download blocked
            print(f"[embeddings] {cls.__name__} unavailable: {exc}", file=sys.stderr)
    print("[embeddings] falling back to HashEmbedder", file=sys.stderr)
    return HashEmbedder()


def warmup() -> None:
    """Download the model at build time so the first request doesn't wait for it."""
    e = make_embedder()
    print(f"[embeddings] ready: {e.model_name} (dim {e.dim}, cache {MODEL_CACHE})")
