"""``StructuredLLM``: the one entry point the rest of the app uses for LLM calls.

Pipeline per call::

    disk cache ─hit─────────────────────────────┐
        │miss                                   ▼
    rate limiter → provider (with backoff) → Pydantic validate ─ok→ result
                                                  │fail (once)
                                     repair turn with errors → validate ─fail→ needs_review

Every call — cached or not, successful or not — emits one :class:`LLMCallRecord`
to the audit sink and one OpenTelemetry span.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from pydantic import BaseModel, ValidationError

from compliancelens.config import LLMTask
from compliancelens.llm.audit import AuditSink, NullAuditSink
from compliancelens.llm.backoff import BackoffPolicy, retry_async
from compliancelens.llm.cache import DiskCache, cache_key, prompt_hash
from compliancelens.llm.errors import LLMError, LLMRefusalError
from compliancelens.llm.pricing import estimate_cost
from compliancelens.llm.prompts import PromptRegistry
from compliancelens.llm.providers.base import ChatProvider
from compliancelens.llm.rate_limiter import RateLimiter
from compliancelens.llm.schema import extract_json_text, output_schema
from compliancelens.llm.types import LLMCallRecord, RawCompletion, RenderedPrompt, Turn

tracer = trace.get_tracer("compliancelens.llm")

MAX_VALIDATION_RETRIES = 1


@dataclass
class StructuredResult[T: BaseModel]:
    """Outcome of a structured call. ``value is None`` means route to human review."""

    value: T | None
    record: LLMCallRecord
    validation_error: str | None = None

    @property
    def needs_review(self) -> bool:
        """True when no validated value was produced."""
        return self.value is None


def format_validation_errors(exc: ValidationError, limit: int = 10) -> list[str]:
    """Compact, model-readable list of Pydantic errors."""
    out = []
    for err in exc.errors(include_url=False)[:limit]:
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        out.append(f"{loc}: {err['msg']}")
    return out


class StructuredLLM:
    """Provider-agnostic structured-output client."""

    def __init__(
        self,
        *,
        provider: ChatProvider,
        model_for: Callable[[LLMTask], str],
        prompts: PromptRegistry,
        limiter: RateLimiter | None = None,
        backoff: BackoffPolicy | None = None,
        cache: DiskCache | None = None,
        audit: AuditSink | None = None,
        max_tokens: int = 8000,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.provider = provider
        self.model_for = model_for
        self.prompts = prompts
        self.limiter = limiter
        self.backoff = backoff or BackoffPolicy()
        self.cache = cache
        self.audit = audit or NullAuditSink()
        self.max_tokens = max_tokens
        self._sleep = sleep

    async def generate[T: BaseModel](
        self,
        *,
        task: LLMTask,
        prompt: RenderedPrompt,
        output: type[T],
        project_id: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> StructuredResult[T]:
        """Run ``prompt`` and return a validated ``output`` instance (or needs_review).

        Raises :class:`LLMError` subclasses other than refusals after recording them.
        """
        model = self.model_for(task)
        schema = output_schema(output)
        record = LLMCallRecord(
            provider=self.provider.name,
            model=model,
            task=task.value,
            prompt_name=prompt.name,
            prompt_version=prompt.version,
            prompt_hash=prompt_hash(prompt),
            metadata=metadata or {},
        )
        started = time.perf_counter()
        result: StructuredResult[T] = StructuredResult(value=None, record=record)

        with tracer.start_as_current_span(f"llm.{prompt.name}") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("llm.provider", self.provider.name)
            span.set_attribute("llm.model_name", model)
            span.set_attribute("prompt.name", prompt.name)
            span.set_attribute("prompt.version", prompt.version)
            try:
                result = await self._generate(model, prompt, output, schema, record)
            except LLMRefusalError as exc:
                record.status = "error"
                record.error = f"refusal: {exc}"
            except LLMError as exc:
                record.status = "error"
                record.error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record.latency_ms = round((time.perf_counter() - started) * 1000, 2)
                span.set_attribute("llm.token_count.prompt", record.input_tokens)
                span.set_attribute("llm.token_count.completion", record.output_tokens)
                span.set_attribute("llm.cache_hit", record.cache_hit)
                span.set_attribute("llm.status", record.status)
                await self.audit.record_llm_call(record, project_id=project_id)
        return result

    async def _generate[T: BaseModel](
        self,
        model: str,
        prompt: RenderedPrompt,
        output: type[T],
        schema: dict[str, Any],
        record: LLMCallRecord,
    ) -> StructuredResult[T]:
        current = prompt
        last_error: str | None = None
        first_key = ""
        for repair_round in range(MAX_VALIDATION_RETRIES + 1):
            key = cache_key(
                provider=self.provider.name,
                model=model,
                prompt=current,
                schema=schema,
                params=self.provider.generation_params(),
            )
            if repair_round == 0:
                first_key = key
            raw = self.cache.get(key) if self.cache else None
            if raw is not None:
                record.cache_hit = True
            else:
                raw = await self._call_provider(model, current, output.__name__, schema, record)
                self._account(raw, record)

            try:
                value = output.model_validate_json(extract_json_text(raw.text))
            except ValidationError as exc:
                errors = format_validation_errors(exc)
                last_error = "; ".join(errors)
                if repair_round < MAX_VALIDATION_RETRIES:
                    record.validation_retries += 1
                    current = self._with_repair(current, raw.text, errors)
                    continue
                record.status = "validation_failed"
                record.error = last_error
                return StructuredResult(value=None, record=record, validation_error=last_error)

            if self.cache is not None and not record.cache_hit:
                self.cache.set(key, raw)
                if first_key != key:
                    # A repaired answer is also served for the original prompt next time.
                    self.cache.set(first_key, raw)
            record.status = "ok"
            return StructuredResult(value=value, record=record)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _call_provider(
        self,
        model: str,
        prompt: RenderedPrompt,
        schema_name: str,
        schema: dict[str, Any],
        record: LLMCallRecord,
    ) -> RawCompletion:
        async def attempt() -> RawCompletion:
            if self.limiter is not None:
                await self.limiter.acquire()
            return await self.provider.complete_json(
                model=model,
                prompt=prompt,
                schema_name=schema_name,
                schema=schema,
                max_tokens=self.max_tokens,
            )

        kwargs: dict[str, Any] = {}
        if self._sleep is not None:
            kwargs["sleep"] = self._sleep
        raw, attempts = await retry_async(attempt, self.backoff, **kwargs)
        record.attempts += attempts
        return raw

    @staticmethod
    def _account(raw: RawCompletion, record: LLMCallRecord) -> None:
        record.input_tokens += raw.input_tokens
        record.output_tokens += raw.output_tokens
        record.cache_read_tokens += raw.cache_read_tokens
        record.cache_write_tokens += raw.cache_write_tokens
        record.estimated_cost_usd = round(record.estimated_cost_usd + estimate_cost(raw), 8)
        record.model = raw.model or record.model

    def _with_repair(
        self, prompt: RenderedPrompt, bad_output: str, errors: list[str]
    ) -> RenderedPrompt:
        repair = self.prompts.render_partial("_partials/validation_repair.jinja", errors=errors)
        turns = [
            *prompt.turns,
            Turn(role="assistant", content=bad_output or "{}"),
            Turn(role="user", content=repair),
        ]
        return prompt.model_copy(update={"turns": turns})
