"""Client-side requests-per-minute / requests-per-day limiter (this process only)."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

from agent_replay.llm.provider import ModelProvider
from agent_replay.llm.types import ModelRequest, ModelResponse, QuotaExhausted

_MINUTE_S = 60.0
_DAY_S = 86_400.0


class RateLimiter:
    """Sleeps to honour RPM; raises QuotaExhausted instead of waiting out RPD."""

    def __init__(
        self,
        rpm: int,
        rpd: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rpm < 1 or rpd < 1:
            raise ValueError("rpm and rpd must be at least 1")
        self._rpm = rpm
        self._rpd = rpd
        self._clock = clock
        self._sleep = sleep
        self._minute: deque[float] = deque()
        self._day: deque[float] = deque()

    @staticmethod
    def _purge(window: deque[float], now: float, span: float) -> None:
        while window and now - window[0] >= span:
            window.popleft()

    def acquire(self) -> None:
        now = self._clock()
        self._purge(self._day, now, _DAY_S)
        if len(self._day) >= self._rpd:
            raise QuotaExhausted(
                f"Request quota exhausted: {self._rpd} requests in 24 h "
                "(AGENT_REPLAY_RPD). Resume later."
            )
        self._purge(self._minute, now, _MINUTE_S)
        if len(self._minute) >= self._rpm:
            self._sleep(self._minute[0] + _MINUTE_S - now)
            now = self._clock()
            self._purge(self._minute, now, _MINUTE_S)
        self._minute.append(now)
        self._day.append(now)


class RateLimitedProvider:
    """Acquires a limiter slot before every call to the wrapped provider."""

    def __init__(self, provider: ModelProvider, limiter: RateLimiter) -> None:
        self._provider = provider
        self._limiter = limiter

    def generate(self, request: ModelRequest) -> ModelResponse:
        self._limiter.acquire()
        return self._provider.generate(request)
