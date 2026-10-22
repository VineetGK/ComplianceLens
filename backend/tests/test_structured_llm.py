from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

from compliancelens.config import LLMTask, Settings
from compliancelens.llm import build_llm
from compliancelens.llm.audit import InMemoryAuditSink
from compliancelens.llm.backoff import BackoffPolicy
from compliancelens.llm.cache import DiskCache, prompt_hash
from compliancelens.llm.client import StructuredLLM
from compliancelens.llm.errors import FatalLLMError, LLMRefusalError, RetryableLLMError
from compliancelens.llm.prompts import PromptRegistry
from compliancelens.llm.providers.mock import MockProvider, register_mock_handler, schema_stub
from compliancelens.llm.types import RawCompletion, RenderedPrompt


class Guess(BaseModel):
    modality: Literal["shall", "must", "will", "should", "may"]
    reason: str
    confidence: float = Field(ge=0, le=1)


class ScriptedProvider:
    """Returns queued outputs / raises queued errors, recording every prompt."""

    name = "scripted"

    def __init__(self, script: list[str | Exception]) -> None:
        self.script = list(script)
        self.prompts: list[RenderedPrompt] = []

    def generation_params(self) -> dict[str, Any]:
        return {}

    async def complete_json(
        self,
        *,
        model: str,
        prompt: RenderedPrompt,
        schema_name: str,
        schema: dict[str, Any],
        max_tokens: int,
    ) -> RawCompletion:
        self.prompts.append(prompt)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return RawCompletion(
            text=item, model="claude-haiku-5-5", input_tokens=1000, output_tokens=100
        )


async def _no_sleep(_: float) -> None:
    return None


def _llm(
    provider: Any, prompts: PromptRegistry, tmp_path: Path, audit: InMemoryAuditSink
) -> StructuredLLM:
    return StructuredLLM(
        provider=provider,
        model_for=lambda task: "claude-haiku-5-5",
        prompts=prompts,
        backoff=BackoffPolicy(max_retries=3),
        cache=DiskCache(tmp_path / "cache"),
        audit=audit,
        sleep=_no_sleep,
    )


GOOD = '{"modality": "shall", "reason": "uses shall", "confidence": 0.9}'


async def test_valid_output_then_cache_hit(prompts: PromptRegistry, tmp_path: Path) -> None:
    audit = InMemoryAuditSink()
    provider = ScriptedProvider([GOOD])
    llm = _llm(provider, prompts, tmp_path, audit)
    prompt = prompts.render("smoke_test", sentence="The Contractor shall comply.")

    first = await llm.generate(task=LLMTask.FAST, prompt=prompt, output=Guess, project_id="p1")
    second = await llm.generate(task=LLMTask.FAST, prompt=prompt, output=Guess, project_id="p1")

    assert (
        first.value == second.value == Guess(modality="shall", reason="uses shall", confidence=0.9)
    )
    assert len(provider.prompts) == 1  # second call served from disk cache
    assert [r.cache_hit for _, r in audit.records] == [False, True]
    rec = audit.records[0][1]
    assert rec.status == "ok" and rec.attempts == 1
    assert rec.prompt_hash == prompt_hash(prompt)
    assert rec.estimated_cost_usd == pytest.approx((1000 * 0.10 + 100 * 0.50) / 1e6)


async def test_validation_failure_is_repaired_once(prompts: PromptRegistry, tmp_path: Path) -> None:
    audit = InMemoryAuditSink()
    bad = '{"modality": "required", "reason": "x", "confidence": 3}'
    provider = ScriptedProvider([bad, GOOD])
    llm = _llm(provider, prompts, tmp_path, audit)
    prompt = prompts.render("smoke_test", sentence="s")

    result = await llm.generate(task=LLMTask.FAST, prompt=prompt, output=Guess)

    assert result.value is not None and result.value.modality == "shall"
    repair_prompt = provider.prompts[1]
    assert [t.role for t in repair_prompt.turns] == ["user", "assistant", "user"]
    assert "modality" in repair_prompt.turns[2].content
    assert "confidence" in repair_prompt.turns[2].content
    assert audit.records[0][1].validation_retries == 1

    # The repaired answer is cached under the original prompt too: no new calls.
    again = await llm.generate(task=LLMTask.FAST, prompt=prompt, output=Guess)
    assert again.value == result.value and len(provider.prompts) == 2


async def test_second_validation_failure_routes_to_review(
    prompts: PromptRegistry, tmp_path: Path
) -> None:
    audit = InMemoryAuditSink()
    provider = ScriptedProvider(["not json at all", '{"modality": "shall"}'])
    llm = _llm(provider, prompts, tmp_path, audit)

    result = await llm.generate(
        task=LLMTask.FAST, prompt=prompts.render("smoke_test", sentence="s"), output=Guess
    )

    assert result.needs_review and result.value is None
    assert result.validation_error and "reason" in result.validation_error
    assert audit.records[0][1].status == "validation_failed"
    assert not any((tmp_path / "cache").rglob("*.json"))  # invalid output never cached


async def test_transient_errors_are_retried_and_counted(
    prompts: PromptRegistry, tmp_path: Path
) -> None:
    audit = InMemoryAuditSink()
    provider = ScriptedProvider([RetryableLLMError("429"), RetryableLLMError("502"), GOOD])
    llm = _llm(provider, prompts, tmp_path, audit)

    result = await llm.generate(
        task=LLMTask.FAST, prompt=prompts.render("smoke_test", sentence="s"), output=Guess
    )

    assert result.value is not None
    assert audit.records[0][1].attempts == 3


async def test_refusal_yields_needs_review(prompts: PromptRegistry, tmp_path: Path) -> None:
    audit = InMemoryAuditSink()
    llm = _llm(ScriptedProvider([LLMRefusalError("no")]), prompts, tmp_path, audit)

    result = await llm.generate(
        task=LLMTask.FAST, prompt=prompts.render("smoke_test", sentence="s"), output=Guess
    )

    assert result.needs_review
    assert audit.records[0][1].error == "refusal: no"


async def test_fatal_error_is_audited_then_raised(prompts: PromptRegistry, tmp_path: Path) -> None:
    audit = InMemoryAuditSink()
    llm = _llm(ScriptedProvider([FatalLLMError("401 bad key")]), prompts, tmp_path, audit)

    with pytest.raises(FatalLLMError):
        await llm.generate(
            task=LLMTask.FAST, prompt=prompts.render("smoke_test", sentence="s"), output=Guess
        )
    assert audit.records[0][1].status == "error"


async def test_mock_provider_default_stub_is_schema_valid(settings: Settings) -> None:
    llm = build_llm(settings)
    result = await llm.generate(
        task=LLMTask.FAST, prompt=llm.prompts.render("smoke_test", sentence="s"), output=Guess
    )
    assert result.value == Guess(modality="shall", reason="mock", confidence=0.0)
    assert result.record.provider == "mock"


async def test_mock_handler_and_fixture_override(settings: Settings, tmp_path: Path) -> None:
    @register_mock_handler("smoke_test")
    def handler(prompt: RenderedPrompt, schema: dict[str, Any]) -> dict[str, Any]:
        modality = "must" if "must" in prompt.turns[0].content else "may"
        return {"modality": modality, "reason": "rule", "confidence": 0.5}

    try:
        llm = build_llm(settings)
        prompt = llm.prompts.render("smoke_test", sentence="Offerors must register in SAM.")
        result = await llm.generate(task=LLMTask.FAST, prompt=prompt, output=Guess)
        assert result.value is not None and result.value.modality == "must"

        fixtures = tmp_path / "fixtures" / "smoke_test"
        fixtures.mkdir(parents=True)
        (fixtures / f"{prompt_hash(prompt)}.json").write_text(GOOD)
        fixture_llm = build_llm(
            settings.model_copy(
                update={"mock_fixtures_dir": tmp_path / "fixtures", "llm_cache_enabled": False}
            )
        )
        fixed = await fixture_llm.generate(task=LLMTask.FAST, prompt=prompt, output=Guess)
        assert fixed.value is not None and fixed.value.modality == "shall"
    finally:
        from compliancelens.llm.providers import mock

        mock._handlers.pop("smoke_test", None)


def test_schema_stub_handles_refs_and_optionals() -> None:
    class Child(BaseModel):
        n: int

    class Parent(BaseModel):
        child: Child
        maybe: str | None
        kids: list[Child]

    schema = Parent.model_json_schema()
    assert Parent.model_validate(schema_stub(schema))


def test_mock_provider_is_registered_for_mock_setting(settings: Settings) -> None:
    assert isinstance(build_llm(settings).provider, MockProvider)
