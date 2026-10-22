"""The interface every chat provider implements."""

from __future__ import annotations

from typing import Any, Protocol

from compliancelens.llm.types import RawCompletion, RenderedPrompt


class ChatProvider(Protocol):
    """A backend that turns a rendered prompt into JSON text for a schema.

    Implementations perform exactly one network attempt per call and translate
    transport errors into :class:`RetryableLLMError` / :class:`FatalLLMError`;
    retries, rate limiting, caching and validation live in ``StructuredLLM``.
    """

    name: str

    async def complete_json(
        self,
        *,
        model: str,
        prompt: RenderedPrompt,
        schema_name: str,
        schema: dict[str, Any],
        max_tokens: int,
    ) -> RawCompletion:
        """Return the model's JSON answer for ``prompt`` constrained to ``schema``."""
        ...

    def generation_params(self) -> dict[str, Any]:
        """Params that change output (part of the cache key)."""
        ...
