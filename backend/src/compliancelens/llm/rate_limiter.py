"""Client-side sliding-window rate limiter (requests per minute)."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


class RateLimiter:
    """Allow at most ``rpm`` acquisitions in any rolling ``window`` seconds.

    Shared across coroutines; acquisitions are serialized by an asyncio lock so
    bursts queue up instead of all hitting the API and getting 429s.
    """

    def __init__(
        self,
        rpm: int,
        *,
        window: float = 60.0,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if rpm < 1:
            raise ValueError("rpm must be >= 1")
        self.rpm = rpm
        self.window = window
        self._clock = clock
        self._sleep = sleep
        self._stamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Wait until a slot is free; return the seconds spent waiting."""
        waited = 0.0
        async with self._lock:
            while True:
                now = self._clock()
                while self._stamps and now - self._stamps[0] >= self.window:
                    self._stamps.popleft()
                if len(self._stamps) < self.rpm:
                    self._stamps.append(now)
                    return waited
                delay = self.window - (now - self._stamps[0])
                waited += delay
                await self._sleep(delay)


_registry: dict[str, RateLimiter] = {}


def shared_limiter(key: str, rpm: int) -> RateLimiter:
    """Return a process-wide limiter for ``key`` (e.g. one per API key/provider).

    NVIDIA's free tier limits per account, so chat and embedding calls share one.
    """
    limiter = _registry.get(key)
    if limiter is None or limiter.rpm != rpm:
        limiter = RateLimiter(rpm)
        _registry[key] = limiter
    return limiter
