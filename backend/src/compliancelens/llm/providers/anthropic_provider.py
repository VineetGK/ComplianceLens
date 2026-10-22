"""Anthropic Claude provider (official SDK) with prompt caching."""

from __future__ import annotations

from typing import Any, Literal

import anthropic
import httpx2
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, OutputConfigParam, TextBlockParam

from compliancelens.llm.errors import (
    FatalLLMError,
    LLMRefusalError,
    RetryableLLMError,
    retry_after_seconds,
)
from compliancelens.llm.schema import strict_schema
from compliancelens.llm.types import RawCompletion, RenderedPrompt, Turn

Effort = Literal["low", "medium", "high"]


def _message(turn: Turn) -> MessageParam:
    if turn.role == "user":
        return {"role": "user", "content": turn.content}
    return {"role": "assistant", "content": turn.content}


class AnthropicProvider:
    """Messages API with ``output_config.format`` JSON-schema structured output.

    The stable system prompt and the (large, repeated) context block - e.g. the
    regulation clauses - are sent as system blocks with a ``cache_control``
    breakpoint on the last one, so repeated calls over the same RFP read them from
    the prompt cache. Per-request document text goes after the breakpoint.
    No sampling params are sent (temperature is rejected by current models).
    """

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        effort: Effort = "low",
        timeout: float = 120.0,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self.effort: Effort = effort
        self._client = AsyncAnthropic(
            api_key=api_key,
            timeout=timeout,
            max_retries=0,  # StructuredLLM owns retries so attempts are audited
            http_client=http_client,
        )

    def generation_params(self) -> dict[str, Any]:
        return {"effort": self.effort}

    @staticmethod
    def system_blocks(prompt: RenderedPrompt) -> list[TextBlockParam]:
        """System content blocks with a cache breakpoint after the stable prefix."""
        blocks: list[TextBlockParam] = [{"type": "text", "text": prompt.system}]
        if prompt.context:
            blocks.append({"type": "text", "text": prompt.context})
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
        return blocks

    async def complete_json(
        self,
        *,
        model: str,
        prompt: RenderedPrompt,
        schema_name: str,
        schema: dict[str, Any],
        max_tokens: int,
    ) -> RawCompletion:
        messages = [_message(t) for t in prompt.turns]
        output_config: OutputConfigParam = {
            "effort": self.effort,
            "format": {"type": "json_schema", "schema": strict_schema(schema)},
        }
        try:
            resp = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=self.system_blocks(prompt),
                messages=messages,
                output_config=output_config,
            )
        except anthropic.RateLimitError as exc:
            raise RetryableLLMError(
                "anthropic 429", retry_after=retry_after_seconds(exc.response.headers)
            ) from exc
        except anthropic.InternalServerError as exc:
            raise RetryableLLMError(f"anthropic {exc.status_code}") from exc
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
            raise RetryableLLMError(f"anthropic connection: {type(exc).__name__}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code in (408, 409, 529):
                raise RetryableLLMError(f"anthropic {exc.status_code}") from exc
            raise FatalLLMError(f"anthropic {exc.status_code}: {exc.message}") from exc

        if resp.stop_reason == "refusal":
            raise LLMRefusalError("anthropic refusal")

        text = next((b.text for b in resp.content if b.type == "text"), "")
        usage = resp.usage
        return RawCompletion(
            text=text,
            model=resp.model or model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
            finish_reason=resp.stop_reason,
        )
