"""Local vector retrieval for Reflection lessons."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence


class TextEncoder(Protocol):
    def encode(self, text: str) -> list[float]:
        """Return one dense vector for text."""


class HashingTextEncoder:
    """A tiny local encoder that works without downloading model weights."""

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive.")
        self.dimensions = dimensions

    def encode(self, text: str) -> list[float]:
        # 把词映射到固定长度向量，适合做本项目的小规模 lesson 检索兜底。
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        return vector


class SentenceTransformerEncoder:
    """Lazy wrapper so recent-memory experiments do not require embedding deps."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ValueError(
                "Embedding retrieval requires sentence-transformers. "
                "Run `.venv/bin/python -m pip install sentence-transformers`."
            ) from error
        try:
            self._model = SentenceTransformer(model_name)
        except OSError as error:
            raise ValueError(
                f"Could not load embedding model {model_name!r}. "
                "Use `--embedding-model local-hash` for offline local retrieval, "
                "or download/cache the Sentence-Transformers model first."
            ) from error

    def encode(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Calculate cosine similarity without adding a separate vector-database service."""
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match.")
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)
