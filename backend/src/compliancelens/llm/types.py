"""Provider-neutral data types for LLM calls."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["user", "assistant"]


class Turn(BaseModel):
    """One conversational message after the system prompt."""

    role: Role
    content: str


class RenderedPrompt(BaseModel):
    """A prompt template rendered with its variables.

    ``system`` holds stable instructions, ``context`` holds large reusable reference
    material (e.g. regulation clauses) that providers may cache, and ``turns`` holds
    the per-request conversation (document text, repair feedback).
    """

    name: str
    version: str
    system: str
    context: str | None = None
    turns: list[Turn]


class RawCompletion(BaseModel):
    """What a provider returns before schema validation."""

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    finish_reason: str | None = None


CallStatus = Literal["ok", "validation_failed", "error"]


class LLMCallRecord(BaseModel):
    """Audit record emitted for every logical structured call (incl. cache hits)."""

    provider: str
    model: str
    task: str
    prompt_name: str
    prompt_version: str
    prompt_hash: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    cache_hit: bool = False
    attempts: int = 0
    validation_retries: int = 0
    status: CallStatus = "ok"
    error: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
