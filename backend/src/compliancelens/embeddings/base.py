"""Embedding provider interface."""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Turns text into vectors. ``model_name`` is stored with every vector so a
    provider switch is detected and triggers a re-index instead of mixing spaces."""

    model_name: str
    dim: int

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed documents/chunks for indexing."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        ...
