"""Embedding providers: nvidia | local | mock."""

from __future__ import annotations

from compliancelens.config import Settings
from compliancelens.embeddings.base import EmbeddingProvider
from compliancelens.embeddings.mock import HashingEmbeddings
from compliancelens.llm.backoff import BackoffPolicy
from compliancelens.llm.errors import ProviderConfigError
from compliancelens.llm.rate_limiter import shared_limiter


def build_embedder(settings: Settings) -> EmbeddingProvider:
    """Instantiate the embedder named by ``EMBEDDING_PROVIDER``."""
    if settings.embedding_provider == "mock":
        return HashingEmbeddings()
    if settings.embedding_provider == "local":
        from compliancelens.embeddings.local import LocalEmbeddings

        return LocalEmbeddings(settings.local_embedding_model, settings.embedding_batch_size)
    if settings.nvidia_api_key is None or not settings.nvidia_api_key.get_secret_value():
        raise ProviderConfigError("EMBEDDING_PROVIDER=nvidia but NVIDIA_API_KEY is not set")
    from compliancelens.embeddings.nvidia import NvidiaEmbeddings

    return NvidiaEmbeddings(
        api_key=settings.nvidia_api_key.get_secret_value(),
        base_url=settings.nvidia_base_url,
        model=settings.nvidia_embedding_model,
        limiter=shared_limiter("nvidia", settings.llm_rpm),
        backoff=BackoffPolicy(
            max_retries=settings.llm_max_retries,
            base_seconds=settings.llm_backoff_base_seconds,
            max_seconds=settings.llm_backoff_max_seconds,
        ),
        batch_size=settings.embedding_batch_size,
    )


__all__ = ["EmbeddingProvider", "HashingEmbeddings", "build_embedder"]
