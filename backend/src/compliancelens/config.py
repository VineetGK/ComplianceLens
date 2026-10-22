"""Application settings loaded from environment / `.env` via pydantic-settings.

Secrets are typed as ``SecretStr`` so they never appear in ``repr()`` or logs.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMTask(StrEnum):
    """Logical LLM workloads; each maps to a configurable model per provider."""

    EXTRACTION = "extraction"
    MAPPING = "mapping"
    COVERAGE = "coverage"
    FAST = "fast"


LLMProviderName = Literal["nvidia", "anthropic", "mock"]
EmbeddingProviderName = Literal["nvidia", "local", "mock"]
NvidiaStructuredMode = Literal["guided_json", "json_schema", "tool"]


class Settings(BaseSettings):
    """All runtime configuration. Model IDs live here, never in code paths."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---------------------------------------------------------------
    llm_provider: LLMProviderName = "nvidia"

    nvidia_api_key: SecretStr | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model_extraction: str = "meta/llama-3.3-70b-instruct"
    nvidia_model_mapping: str = "meta/llama-3.3-70b-instruct"
    nvidia_model_coverage: str = "meta/llama-3.3-70b-instruct"
    nvidia_model_fast: str = "meta/llama-3.1-8b-instruct"
    nvidia_structured_mode: NvidiaStructuredMode = "guided_json"

    anthropic_api_key: SecretStr | None = None
    anthropic_model_extraction: str = "claude-haiku-5-5"
    anthropic_model_mapping: str = "claude-sonnet-5-5"
    anthropic_model_coverage: str = "claude-sonnet-5-5"
    anthropic_model_fast: str = "claude-haiku-5-5"
    anthropic_effort: Literal["low", "medium", "high"] = "low"

    mock_model: str = "mock-1"
    mock_fixtures_dir: Path | None = None

    llm_rpm: int = Field(default=30, ge=1)
    llm_max_retries: int = Field(default=5, ge=0)
    llm_backoff_base_seconds: float = Field(default=1.0, gt=0)
    llm_backoff_max_seconds: float = Field(default=30.0, gt=0)
    llm_max_output_tokens: int = Field(default=8000, ge=256)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_cache_enabled: bool = True
    llm_cache_dir: Path = Path(".cache/llm")

    # --- Embeddings --------------------------------------------------------
    embedding_provider: EmbeddingProviderName = "nvidia"
    nvidia_embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = Field(default=32, ge=1)

    # --- Database ----------------------------------------------------------
    database_url: str = (
        "postgresql+psycopg://compliancelens:compliancelens@localhost:5432/compliancelens"
    )

    # --- Tracing -----------------------------------------------------------
    tracing_enabled: bool = True
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project_name: str = "compliancelens"

    # --- App ---------------------------------------------------------------
    log_level: str = "INFO"
    prompts_dir: Path = Path(__file__).resolve().parents[2] / "prompts"

    def model_for(self, task: LLMTask) -> str:
        """Return the configured model ID for ``task`` under the active provider."""
        if self.llm_provider == "mock":
            return self.mock_model
        field = f"{self.llm_provider}_model_{task.value}"
        value: str = getattr(self, field)
        return value

    def secret_values(self) -> list[str]:
        """Raw secret strings, used only by the log redactor."""
        out: list[str] = []
        for secret in (self.nvidia_api_key, self.anthropic_api_key):
            if secret is not None and secret.get_secret_value():
                out.append(secret.get_secret_value())
        return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings."""
    return Settings()
