"""Build a ``StructuredLLM`` from settings."""

from __future__ import annotations

from compliancelens.config import Settings
from compliancelens.llm.audit import AuditSink
from compliancelens.llm.backoff import BackoffPolicy
from compliancelens.llm.cache import DiskCache
from compliancelens.llm.client import StructuredLLM
from compliancelens.llm.errors import ProviderConfigError
from compliancelens.llm.prompts import PromptRegistry
from compliancelens.llm.providers.base import ChatProvider
from compliancelens.llm.providers.mock import MockProvider
from compliancelens.llm.rate_limiter import RateLimiter, shared_limiter


def build_provider(settings: Settings) -> ChatProvider:
    """Instantiate the provider named by ``LLM_PROVIDER``."""
    if settings.llm_provider == "mock":
        return MockProvider(fixtures_dir=settings.mock_fixtures_dir)
    if settings.llm_provider == "nvidia":
        if settings.nvidia_api_key is None or not settings.nvidia_api_key.get_secret_value():
            raise ProviderConfigError("LLM_PROVIDER=nvidia but NVIDIA_API_KEY is not set")
        from compliancelens.llm.providers.nvidia import NvidiaProvider

        return NvidiaProvider(
            api_key=settings.nvidia_api_key.get_secret_value(),
            base_url=settings.nvidia_base_url,
            mode=settings.nvidia_structured_mode,
            timeout=settings.llm_timeout_seconds,
        )
    if settings.anthropic_api_key is None or not settings.anthropic_api_key.get_secret_value():
        raise ProviderConfigError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
    from compliancelens.llm.providers.anthropic_provider import AnthropicProvider

    return AnthropicProvider(
        api_key=settings.anthropic_api_key.get_secret_value(),
        effort=settings.anthropic_effort,
        timeout=settings.llm_timeout_seconds,
    )


def build_llm(settings: Settings, *, audit: AuditSink | None = None) -> StructuredLLM:
    """Wire provider, limiter, backoff, cache, prompts and audit sink."""
    limiter: RateLimiter | None = None
    if settings.llm_provider != "mock":
        limiter = shared_limiter(settings.llm_provider, settings.llm_rpm)
    return StructuredLLM(
        provider=build_provider(settings),
        model_for=settings.model_for,
        prompts=PromptRegistry(settings.prompts_dir),
        limiter=limiter,
        backoff=BackoffPolicy(
            max_retries=settings.llm_max_retries,
            base_seconds=settings.llm_backoff_base_seconds,
            max_seconds=settings.llm_backoff_max_seconds,
        ),
        cache=DiskCache(settings.llm_cache_dir) if settings.llm_cache_enabled else None,
        audit=audit,
        max_tokens=settings.llm_max_output_tokens,
    )
