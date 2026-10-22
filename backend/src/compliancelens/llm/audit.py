"""Where LLM call records go. Phase 5 adds a Postgres-backed sink."""

from __future__ import annotations

from typing import Protocol

from compliancelens.llm.types import LLMCallRecord


class AuditSink(Protocol):
    """Receives one record per logical LLM call."""

    async def record_llm_call(self, record: LLMCallRecord, *, project_id: str | None) -> None:
        """Persist ``record``."""
        ...


class NullAuditSink:
    """Drops records."""

    async def record_llm_call(self, record: LLMCallRecord, *, project_id: str | None) -> None:
        return None


class InMemoryAuditSink:
    """Keeps records in a list (tests, scripts)."""

    def __init__(self) -> None:
        self.records: list[tuple[str | None, LLMCallRecord]] = []

    async def record_llm_call(self, record: LLMCallRecord, *, project_id: str | None) -> None:
        self.records.append((project_id, record))
