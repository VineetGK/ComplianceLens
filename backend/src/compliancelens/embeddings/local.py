"""Offline ``sentence-transformers`` embeddings (default BAAI/bge-small-en-v1.5).

Requires the optional extra: ``uv sync --extra local-embeddings``.
"""

from __future__ import annotations

import asyncio
from typing import Any

# BGE v1.5 recommends this instruction on queries (not passages) for retrieval.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class LocalEmbeddings:
    """CPU embeddings; model loads lazily on first use and runs in a worker thread."""

    def __init__(self, model: str, batch_size: int = 32) -> None:
        self.model_name = model
        self.dim = 384 if "small" in model else 768
        self._batch_size = batch_size
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise RuntimeError(
                    "EMBEDDING_PROVIDER=local needs `uv sync --extra local-embeddings`"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(texts, batch_size=self._batch_size, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode, texts)

    async def embed_query(self, text: str) -> list[float]:
        prefix = BGE_QUERY_INSTRUCTION if "bge" in self.model_name.lower() else ""
        return (await asyncio.to_thread(self._encode, [prefix + text]))[0]
