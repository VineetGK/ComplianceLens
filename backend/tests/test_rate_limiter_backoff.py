from __future__ import annotations

import random

import pytest

from compliancelens.llm.backoff import BackoffPolicy, retry_async
from compliancelens.llm.errors import FatalLLMError, RetryableLLMError
from compliancelens.llm.rate_limiter import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


async def test_limiter_allows_rpm_then_waits_for_window() -> None:
    clock = FakeClock()
    limiter = RateLimiter(3, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        assert await limiter.acquire() == 0.0
    waited = await limiter.acquire()
    assert waited == pytest.approx(60.0)
    assert clock.now == pytest.approx(60.0)


async def test_limiter_sliding_window_frees_oldest_slot() -> None:
    clock = FakeClock()
    limiter = RateLimiter(2, clock=clock, sleep=clock.sleep)
    await limiter.acquire()  # t=0
    clock.now = 30.0
    await limiter.acquire()  # t=30
    waited = await limiter.acquire()  # must wait until t=60 (first stamp expires)
    assert waited == pytest.approx(30.0)


def test_limiter_rejects_zero_rpm() -> None:
    with pytest.raises(ValueError):
        RateLimiter(0)


async def test_retry_succeeds_after_transient_errors() -> None:
    clock = FakeClock()
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RetryableLLMError("429")
        return "ok"

    result, attempts = await retry_async(
        flaky,
        BackoffPolicy(max_retries=5, base_seconds=1, max_seconds=8),
        sleep=clock.sleep,
        rng=random.Random(0),
    )
    assert (result, attempts) == ("ok", 3)
    assert len(clock.sleeps) == 2
    assert all(0 <= s <= 8 for s in clock.sleeps)


async def test_retry_honours_retry_after() -> None:
    clock = FakeClock()
    calls = 0

    async def limited() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableLLMError("429", retry_after=7.0)
        return "ok"

    await retry_async(
        limited, BackoffPolicy(max_seconds=30), sleep=clock.sleep, rng=random.Random(0)
    )
    assert clock.sleeps[0] >= 7.0


async def test_retry_gives_up_and_reraises() -> None:
    clock = FakeClock()

    async def always() -> str:
        raise RetryableLLMError("503")

    with pytest.raises(RetryableLLMError):
        await retry_async(always, BackoffPolicy(max_retries=2), sleep=clock.sleep)
    assert len(clock.sleeps) == 2


async def test_fatal_errors_are_not_retried() -> None:
    clock = FakeClock()

    async def bad() -> str:
        raise FatalLLMError("401")

    with pytest.raises(FatalLLMError):
        await retry_async(bad, BackoffPolicy(), sleep=clock.sleep)
    assert clock.sleeps == []
