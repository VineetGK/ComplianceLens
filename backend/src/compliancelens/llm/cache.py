"""Content-addressed disk cache for LLM completions, so re-runs cost nothing."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from compliancelens.llm.types import RawCompletion, RenderedPrompt


def prompt_hash(prompt: RenderedPrompt) -> str:
    """Stable hash of the rendered prompt content (used in audit records)."""
    payload = prompt.model_dump(mode="json")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cache_key(
    *,
    provider: str,
    model: str,
    prompt: RenderedPrompt,
    schema: dict[str, Any],
    params: dict[str, Any],
) -> str:
    """hash(provider + model + prompt + output schema + generation params)."""
    payload = {
        "provider": provider,
        "model": model,
        "prompt": prompt.model_dump(mode="json"),
        "schema": schema,
        "params": params,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class DiskCache:
    """One JSON file per key under ``root/ab/abcdef….json``; writes are atomic."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> RawCompletion | None:
        """Return the cached completion or ``None``; corrupt entries are ignored."""
        path = self._path(key)
        try:
            return RawCompletion.model_validate_json(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return None

    def set(self, key: str, value: RawCompletion) -> None:
        """Persist ``value`` under ``key``."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(value.model_dump_json())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
