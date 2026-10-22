from __future__ import annotations

import os
from pathlib import Path

import pytest

# Tests must never reach a real provider, whatever the developer's .env says.
os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "mock"
os.environ["TRACING_ENABLED"] = "false"

from compliancelens.config import Settings
from compliancelens.llm.prompts import PromptRegistry

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="mock",
        embedding_provider="mock",
        tracing_enabled=False,
        llm_cache_dir=tmp_path / "llm-cache",
    )


@pytest.fixture
def prompts() -> PromptRegistry:
    return PromptRegistry(PROMPTS_DIR)


@pytest.fixture
def prompt_dir(tmp_path: Path) -> Path:
    """An isolated prompts directory with the real partials plus test templates."""
    root = tmp_path / "prompts"
    (root / "_partials").mkdir(parents=True)
    for partial in (PROMPTS_DIR / "_partials").iterdir():
        (root / "_partials" / partial.name).write_text(partial.read_text(encoding="utf-8"))
    return root
