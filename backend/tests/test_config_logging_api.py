from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from compliancelens.config import LLMTask, Settings
from compliancelens.llm.errors import ProviderConfigError
from compliancelens.llm.factory import build_provider
from compliancelens.logging_setup import REDACTED, SecretRedactingFilter, redact


def test_model_for_uses_active_provider() -> None:
    s = Settings(
        _env_file=None, llm_provider="nvidia", nvidia_model_fast="meta/llama-3.1-8b-instruct"
    )
    assert s.model_for(LLMTask.FAST) == "meta/llama-3.1-8b-instruct"
    assert s.model_for(LLMTask.MAPPING) == "meta/llama-3.3-70b-instruct"
    a = Settings(_env_file=None, llm_provider="anthropic")
    assert a.model_for(LLMTask.EXTRACTION) == "claude-haiku-5-5"
    assert a.model_for(LLMTask.MAPPING) == "claude-sonnet-5-5"


def test_secrets_hidden_from_repr() -> None:
    s = Settings(_env_file=None, nvidia_api_key=SecretStr("nvapi-supersecretvalue"))
    assert "supersecret" not in repr(s)
    assert s.secret_values() == ["nvapi-supersecretvalue"]


@pytest.mark.parametrize("provider", ["nvidia", "anthropic"])
def test_missing_key_is_a_clear_error(provider: str) -> None:
    with pytest.raises(ProviderConfigError, match="API_KEY"):
        build_provider(Settings(_env_file=None, llm_provider=provider))  # type: ignore[arg-type]


def test_redact_known_and_configured_secrets() -> None:
    text = "key=nvapi-abcdefghijklmnop other=sk-ant-api03-zzzzzzzzzzzz mine=hunter2hunter2"
    out = redact(text, ["hunter2hunter2"])
    assert "nvapi-abc" not in out and "sk-ant-api03" not in out and "hunter2" not in out
    assert out.count(REDACTED) == 3


def test_log_filter_redacts_formatted_args(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("test.redact")
    logger.addFilter(SecretRedactingFilter(["my-secret-token"]))
    with caplog.at_level(logging.INFO, logger="test.redact"):
        logger.info("calling with %s", "my-secret-token")
    assert "my-secret-token" not in caplog.text
    assert REDACTED in caplog.text


def test_health_reports_provider_without_db(monkeypatch: pytest.MonkeyPatch) -> None:
    from compliancelens.api import main

    class Boom:
        def connect(self) -> None:
            raise RuntimeError("no db")

    monkeypatch.setattr(main, "get_engine", lambda: Boom())
    with TestClient(main.app) as client:
        body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["llm_provider"] == "mock"
    assert "key" not in str(body).lower()
