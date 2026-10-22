"""Exceptions shared by all LLM providers."""

from __future__ import annotations

from collections.abc import Mapping


def retry_after_seconds(headers: Mapping[str, str] | None) -> float | None:
    """Parse a numeric ``Retry-After`` header (seconds); ignore HTTP-date forms."""
    value = headers.get("retry-after") if headers is not None else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


class LLMError(Exception):
    """Base class for LLM-layer failures."""


class RetryableLLMError(LLMError):
    """Transient failure (HTTP 429 / 5xx / connection). Safe to retry."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class FatalLLMError(LLMError):
    """Non-retryable failure (auth, bad request, unknown model)."""


class LLMRefusalError(LLMError):
    """The model declined to answer; the item must go to human review."""


class ProviderConfigError(LLMError):
    """Provider selected but not configured (e.g. missing API key)."""
