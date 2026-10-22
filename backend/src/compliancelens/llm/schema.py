"""JSON-schema helpers shared by providers."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from pydantic import BaseModel

# Keywords some constrained decoders reject; Pydantic still enforces them afterwards.
_UNSUPPORTED_KEYWORDS = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "pattern",
    "format",
    "default",
    "title",
}


def output_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema for ``model`` (validation mode)."""
    return model.model_json_schema(mode="validation")


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy where every object forbids extra keys and requires all keys.

    Constraint keywords are stripped because constrained decoders differ in support;
    the authoritative check is the Pydantic validation that runs on every response.
    """
    out = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in list(node):
                if key in _UNSUPPORTED_KEYWORDS:
                    node.pop(key)
            if node.get("type") == "object" and isinstance(node.get("properties"), dict):
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for key, value in node.items():
                # Keys of these mappings are field / definition names, not keywords.
                if key in ("properties", "$defs", "definitions") and isinstance(value, dict):
                    for sub in value.values():
                        walk(sub)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(out)
    return out


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def extract_json_text(text: str) -> str:
    """Best-effort: strip markdown fences / prose around a single JSON object."""
    stripped = text.strip()
    fenced = _FENCE.match(stripped)
    if fenced:
        return fenced.group(1)
    if stripped.startswith(("{", "[")):
        return stripped
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def schema_json(schema: dict[str, Any]) -> str:
    """Compact deterministic serialization (for prompts and cache keys)."""
    return json.dumps(schema, sort_keys=True, separators=(",", ":"))
