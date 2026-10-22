"""NVIDIA API Catalog provider (OpenAI-compatible endpoint, free tier)."""

from __future__ import annotations

from typing import Any

import httpx2
import openai
from openai import AsyncOpenAI

from compliancelens.config import NvidiaStructuredMode
from compliancelens.llm.errors import (
    FatalLLMError,
    LLMRefusalError,
    RetryableLLMError,
    retry_after_seconds,
)
from compliancelens.llm.schema import schema_json, strict_schema
from compliancelens.llm.types import RawCompletion, RenderedPrompt

_JSON_INSTRUCTION = (
    "\n\nOUTPUT FORMAT: respond with a single JSON object that validates against this "
    "JSON Schema. No prose, no markdown fences.\nSCHEMA: {schema}"
)


class NvidiaProvider:
    """Chat completions against ``integrate.api.nvidia.com`` via the ``openai`` SDK.

    Structured output modes:

    * ``guided_json`` - NIM ``nvext.guided_json`` constrained decoding (default)
    * ``json_schema`` - OpenAI-style ``response_format`` JSON schema
    * ``tool`` - single forced function call whose arguments are the object
    """

    name = "nvidia"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        mode: NvidiaStructuredMode = "guided_json",
        timeout: float = 120.0,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self.mode = mode
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,  # StructuredLLM owns retries so attempts are audited
            http_client=http_client,
        )

    def generation_params(self) -> dict[str, Any]:
        return {"temperature": 0.0, "mode": self.mode}

    async def complete_json(
        self,
        *,
        model: str,
        prompt: RenderedPrompt,
        schema_name: str,
        schema: dict[str, Any],
        max_tokens: int,
    ) -> RawCompletion:
        strict = strict_schema(schema)
        system = prompt.system
        if prompt.context:
            system = f"{system}\n\n{prompt.context}"
        system += _JSON_INSTRUCTION.format(schema=schema_json(strict))

        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        messages += [{"role": t.role, "content": t.content} for t in prompt.turns]

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if self.mode == "guided_json":
            kwargs["extra_body"] = {"nvext": {"guided_json": strict}}
        elif self.mode == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": strict, "strict": True},
            }
        else:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": schema_name,
                        "description": "Return the result.",
                        "parameters": strict,
                    },
                }
            ]
            kwargs["tool_choice"] = {"type": "function", "function": {"name": schema_name}}

        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except openai.RateLimitError as exc:
            raise RetryableLLMError(
                "nvidia 429", retry_after=retry_after_seconds(exc.response.headers)
            ) from exc
        except openai.InternalServerError as exc:
            raise RetryableLLMError(f"nvidia {exc.status_code}") from exc
        except (openai.APIConnectionError, openai.APITimeoutError) as exc:
            raise RetryableLLMError(f"nvidia connection: {type(exc).__name__}") from exc
        except openai.APIStatusError as exc:
            raise FatalLLMError(f"nvidia {exc.status_code}: {exc.message}") from exc

        if not resp.choices:
            raise RetryableLLMError("nvidia returned no choices")
        choice = resp.choices[0]
        if choice.finish_reason == "content_filter":
            raise LLMRefusalError("nvidia content filter")

        if self.mode == "tool" and choice.message.tool_calls:
            call = choice.message.tool_calls[0]
            text = call.function.arguments if call.type == "function" else ""
        else:
            text = choice.message.content or ""

        usage = resp.usage
        return RawCompletion(
            text=text,
            model=resp.model or model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason,
        )
