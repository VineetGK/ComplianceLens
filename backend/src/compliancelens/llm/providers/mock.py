"""Deterministic offline provider for tests and CI.

Resolution order for a call:

1. Fixture file ``<fixtures_dir>/<prompt_name>/<prompt_hash>.json`` if present.
2. A handler registered for the prompt name via :func:`register_mock_handler`
   (later phases register rule-based handlers, e.g. a regex requirement extractor).
3. A minimal schema-valid stub derived from the JSON schema.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from compliancelens.llm.cache import prompt_hash
from compliancelens.llm.types import RawCompletion, RenderedPrompt

MockHandler = Callable[[RenderedPrompt, dict[str, Any]], dict[str, Any]]

_handlers: dict[str, MockHandler] = {}


def register_mock_handler(prompt_name: str) -> Callable[[MockHandler], MockHandler]:
    """Decorator: route mock calls for ``prompt_name`` to the decorated function."""

    def deco(fn: MockHandler) -> MockHandler:
        _handlers[prompt_name] = fn
        return fn

    return deco


def schema_stub(schema: dict[str, Any], root: dict[str, Any] | None = None) -> Any:
    """Build the smallest value that satisfies ``schema`` (first enum, empty list…)."""
    root = root or schema
    if "$ref" in schema:
        ref = schema["$ref"].split("/")[-1]
        return schema_stub(root["$defs"][ref], root)
    for combinator in ("anyOf", "oneOf", "allOf"):
        if combinator in schema:
            options = [o for o in schema[combinator] if o.get("type") != "null"]
            return schema_stub(options[0] if options else schema[combinator][0], root)
    if "enum" in schema:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    kind = schema.get("type")
    if kind == "object":
        return {k: schema_stub(v, root) for k, v in schema.get("properties", {}).items()}
    if kind == "array":
        return []
    if kind == "string":
        return "mock"
    if kind == "integer":
        return int(schema.get("minimum", 0))
    if kind == "number":
        return float(schema.get("minimum", 0.0))
    if kind == "boolean":
        return False
    return None


class MockProvider:
    """Never touches the network; output depends only on the prompt."""

    name = "mock"

    def __init__(self, fixtures_dir: Path | None = None) -> None:
        self.fixtures_dir = fixtures_dir
        self.calls: list[RenderedPrompt] = []

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
        self.calls.append(prompt)
        payload = self._resolve(prompt, schema)
        text = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True)
        tokens_in = sum(len(t.content.split()) for t in prompt.turns) + len(prompt.system.split())
        return RawCompletion(
            text=text,
            model=model,
            input_tokens=tokens_in,
            output_tokens=len(text.split()),
            finish_reason="stop",
        )

    def _resolve(self, prompt: RenderedPrompt, schema: dict[str, Any]) -> Any:
        if self.fixtures_dir is not None:
            path = self.fixtures_dir / prompt.name / f"{prompt_hash(prompt)}.json"
            if path.exists():
                return path.read_text(encoding="utf-8")
        handler = _handlers.get(prompt.name)
        if handler is not None:
            return handler(prompt, schema)
        return schema_stub(schema)
