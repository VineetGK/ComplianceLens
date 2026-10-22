"""NVIDIA ``nv-embedqa-e5-v5`` embeddings via the OpenAI-compatible endpoint."""

from __future__ import annotations

from typing import Literal

import httpx2
import openai
from openai import AsyncOpenAI

from compliancelens.llm.backoff import BackoffPolicy, retry_async
from compliancelens.llm.errors import FatalLLMError, RetryableLLMError
from compliancelens.llm.rate_limiter import RateLimiter

KNOWN_DIMS = {"nvidia/nv-embedqa-e5-v5": 1024}

InputType = Literal["passage", "query"]


class NvidiaEmbeddings:
    """Asymmetric retrieval embeddings: ``input_type`` differs for passages vs queries."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        limiter: RateLimiter | None = None,
        backoff: BackoffPolicy | None = None,
        batch_size: int = 32,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self.model_name = model
        self.dim = KNOWN_DIMS.get(model, 1024)
        self._limiter = limiter
        self._backoff = backoff or BackoffPolicy()
        self._batch_size = batch_size
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, max_retries=0, http_client=http_client
        )

    async def _embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        async def attempt() -> list[list[float]]:
            if self._limiter is not None:
                await self._limiter.acquire()
            try:
                resp = await self._client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                    encoding_format="float",
                    extra_body={"input_type": input_type, "truncate": "END"},
                )
            except openai.RateLimitError as exc:
                raise RetryableLLMError("nvidia embeddings 429") from exc
            except openai.InternalServerError as exc:
                raise RetryableLLMError(f"nvidia embeddings {exc.status_code}") from exc
            except openai.APIConnectionError as exc:
                raise RetryableLLMError("nvidia embeddings connection") from exc
            except openai.APIStatusError as exc:
                raise FatalLLMError(f"nvidia embeddings {exc.status_code}: {exc.message}") from exc
            return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

        vectors, _ = await retry_async(attempt, self._backoff)
        return vectors

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            out.extend(await self._embed(texts[i : i + self._batch_size], "passage"))
        return out

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], "query"))[0]
