"""Provider adapters exercised against a fake HTTP transport (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2 as httpx
import pytest

from compliancelens.embeddings.mock import HashingEmbeddings
from compliancelens.embeddings.nvidia import NvidiaEmbeddings
from compliancelens.llm.errors import FatalLLMError, LLMRefusalError, RetryableLLMError
from compliancelens.llm.providers.anthropic_provider import AnthropicProvider
from compliancelens.llm.providers.nvidia import NvidiaProvider
from compliancelens.llm.types import RenderedPrompt, Turn

SCHEMA = {
    "type": "object",
    "properties": {"modality": {"type": "string", "enum": ["shall", "may"]}},
    "required": ["modality"],
}
PROMPT = RenderedPrompt(
    name="smoke_test",
    version="v1",
    system="SYSTEM RULES",
    context="REG CONTEXT",
    turns=[Turn(role="user", content="<<<UNTRUSTED_DOC id=1>>>x<<<END_UNTRUSTED_DOC id=1>>>")],
)

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler, seen: list[httpx.Request]) -> httpx.AsyncClient:
    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(wrapped))


def _chat_ok(
    content: str | None = '{"modality": "shall"}', tool_args: str | None = None
) -> Handler:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_args is not None:
        message["tool_calls"] = [
            {"id": "c1", "type": "function", "function": {"name": "Guess", "arguments": tool_args}}
        ]
    body = {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "meta/llama-3.3-70b-instruct",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 8, "total_tokens": 128},
    }
    return lambda _: httpx.Response(200, json=body)


async def _nvidia_call(
    handler: Handler, mode: str = "guided_json"
) -> tuple[Any, list[httpx.Request]]:
    seen: list[httpx.Request] = []
    provider = NvidiaProvider(
        api_key="nvapi-test-key-123456",
        base_url="https://nim.test/v1",
        mode=mode,
        http_client=_client(handler, seen),
    )  # type: ignore[arg-type]
    raw = await provider.complete_json(
        model="meta/llama-3.3-70b-instruct",
        prompt=PROMPT,
        schema_name="Guess",
        schema=SCHEMA,
        max_tokens=512,
    )
    return raw, seen


async def test_nvidia_guided_json_request_shape() -> None:
    raw, seen = await _nvidia_call(_chat_ok())
    req = seen[0]
    body = json.loads(req.content)
    assert req.url.path == "/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer nvapi-test-key-123456"
    assert body["temperature"] == 0.0
    assert body["nvext"]["guided_json"]["additionalProperties"] is False
    assert body["messages"][0]["role"] == "system"
    assert "REG CONTEXT" in body["messages"][0]["content"]
    assert body["messages"][1]["content"].startswith("<<<UNTRUSTED_DOC")
    assert raw.text == '{"modality": "shall"}'
    assert (raw.input_tokens, raw.output_tokens) == (120, 8)


async def test_nvidia_json_schema_and_tool_modes() -> None:
    _, seen = await _nvidia_call(_chat_ok(), mode="json_schema")
    assert json.loads(seen[0].content)["response_format"]["type"] == "json_schema"

    raw, seen = await _nvidia_call(
        _chat_ok(content=None, tool_args='{"modality": "may"}'), mode="tool"
    )
    body = json.loads(seen[0].content)
    assert body["tool_choice"]["function"]["name"] == "Guess"
    assert raw.text == '{"modality": "may"}'


@pytest.mark.parametrize(
    ("status", "headers", "exc", "retry_after"),
    [
        (429, {"retry-after": "4"}, RetryableLLMError, 4.0),
        (503, {}, RetryableLLMError, None),
        (401, {}, FatalLLMError, None),
        (400, {}, FatalLLMError, None),
    ],
)
async def test_nvidia_error_mapping(
    status: int, headers: dict[str, str], exc: type[Exception], retry_after: float | None
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers=headers, json={"error": {"message": "nope"}})

    with pytest.raises(exc) as info:
        await _nvidia_call(handler)
    if retry_after is not None:
        assert info.value.retry_after == retry_after  # type: ignore[attr-defined]


def _anthropic_ok(stop_reason: str = "end_turn") -> Handler:
    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": '{"modality": "shall"}'}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 50,
            "output_tokens": 9,
            "cache_read_input_tokens": 2000,
            "cache_creation_input_tokens": 0,
        },
    }
    return lambda _: httpx.Response(200, json=body)


async def _anthropic_call(handler: Handler) -> tuple[Any, list[httpx.Request]]:
    seen: list[httpx.Request] = []
    provider = AnthropicProvider(
        api_key="sk-ant-test-0000000000", http_client=_client(handler, seen)
    )
    raw = await provider.complete_json(
        model="claude-haiku-5-5", prompt=PROMPT, schema_name="Guess", schema=SCHEMA, max_tokens=512
    )
    return raw, seen


async def test_anthropic_request_uses_caching_and_structured_output() -> None:
    raw, seen = await _anthropic_call(_anthropic_ok())
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/messages"
    assert [b["text"] for b in body["system"]] == ["SYSTEM RULES", "REG CONTEXT"]
    assert "cache_control" not in body["system"][0]
    assert body["system"][1]["cache_control"] == {"type": "ephemeral"}
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["effort"] == "low"
    assert "temperature" not in body
    assert body["messages"] == [{"role": "user", "content": PROMPT.turns[0].content}]
    assert raw.cache_read_tokens == 2000 and raw.input_tokens == 50


async def test_anthropic_refusal_and_errors() -> None:
    with pytest.raises(LLMRefusalError):
        await _anthropic_call(_anthropic_ok(stop_reason="refusal"))

    def overloaded(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            529, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
        )

    with pytest.raises(RetryableLLMError):
        await _anthropic_call(overloaded)


async def test_nvidia_embeddings_use_input_type() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        n = len(json.loads(request.content)["input"])
        data = [
            {"object": "embedding", "index": i, "embedding": [0.1 * (i + 1)] * 4} for i in range(n)
        ]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": data,
                "model": "m",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    emb = NvidiaEmbeddings(
        api_key="nvapi-x",
        base_url="https://nim.test/v1",
        model="nvidia/nv-embedqa-e5-v5",
        batch_size=2,
        http_client=_client(handler, seen),
    )
    vectors = await emb.embed_passages(["a", "b", "c"])
    await emb.embed_query("q")

    bodies = [json.loads(r.content) for r in seen]
    assert len(vectors) == 3 and emb.dim == 1024
    assert [b["input_type"] for b in bodies] == ["passage", "passage", "query"]
    assert [len(b["input"]) for b in bodies] == [2, 1, 1]


async def test_hashing_embeddings_are_deterministic_and_lexical() -> None:
    emb = HashingEmbeddings()
    a, b, c = await emb.embed_passages(
        [
            "contractor shall submit monthly status reports",
            "monthly status reports shall be submitted by the contractor",
            "offeror must hold a facility security clearance",
        ]
    )

    def cos(x: list[float], y: list[float]) -> float:
        return sum(i * j for i, j in zip(x, y, strict=True))

    assert a == (await emb.embed_passages(["contractor shall submit monthly status reports"]))[0]
    assert cos(a, b) > cos(a, c)
    assert emb.model_name == "mock/hashing-384"
