"""Deterministic hashing embedder for tests/CI (lexical, no model download)."""

from __future__ import annotations

import hashlib
import itertools
import math
import re

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbeddings:
    """Feature-hashed bag of words + bigrams, L2-normalised.

    Texts sharing vocabulary land close together, which is enough for retrieval
    tests to be meaningful without any network or model weights.
    """

    def __init__(self, dim: int = 384) -> None:
        self.model_name = f"mock/hashing-{dim}"
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        tokens = _TOKEN.findall(text.lower())
        features = tokens + [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
        vec = [0.0] * self.dim
        for feat in features:
            digest = hashlib.blake2b(feat.encode(), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
