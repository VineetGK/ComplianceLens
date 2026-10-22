"""Exponential backoff with full jitter for transient provider errors."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from compliancelens.llm.errors import RetryableLLMError

T = TypeVar("T")
Sleep = Callable[[float], Awaitable[None]]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackoffPolicy:
    """Retry settings. ``max_retries`` counts retries after the first attempt."""

    max_retries: int = 5
    base_seconds: float = 1.0
    max_seconds: float = 30.0

    def delay(self, attempt: int, rng: random.Random, retry_after: float | None) -> float:
        """Delay before retry number ``attempt`` (1-based)."""
        ceiling = min(self.max_seconds, self.base_seconds * (2 ** (attempt - 1)))
        jittered = rng.uniform(0, ceiling)
        if retry_after is not None:
            return min(self.max_seconds, max(retry_after, jittered))
        return jittered


async def retry_async[T](
    fn: Callable[[], Awaitable[T]],
    policy: BackoffPolicy,
    *,
    sleep: Sleep = asyncio.sleep,
    rng: random.Random | None = None,
) -> tuple[T, int]:
    """Run ``fn`` retrying on :class:`RetryableLLMError`.

    Returns ``(result, attempts)``. Re-raises the last error once retries are spent.
    """
    rng = rng or random.Random()  # noqa: S311 - jitter, not crypto
    attempt = 0
    while True:
        attempt += 1
        try:
            return await fn(), attempt
        except RetryableLLMError as exc:
            if attempt > policy.max_retries:
                raise
            delay = policy.delay(attempt, rng, exc.retry_after)
            log.warning("retryable LLM error (attempt %d): %s; sleeping %.1fs", attempt, exc, delay)
            await sleep(delay)
