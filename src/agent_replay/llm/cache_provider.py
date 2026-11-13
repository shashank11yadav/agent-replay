"""Cache-first provider: serve from the response cache, fall back to a live provider.

A replay also passes the source run's own recorded responses (``recorded``, by request hash).
They are served before the shared cache: the cache is keyed by request alone, so a later
recording that sent the same request may have replaced the entry, but the source run's spans
still hold what it actually received.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from agent_replay.llm.budget import BudgetTracker
from agent_replay.llm.hashing import request_hash
from agent_replay.llm.provider import ModelProvider
from agent_replay.llm.types import CacheMissNoLiveModel, ModelRequest, ModelResponse


class ResponseCache(Protocol):
    def get(self, request_hash: str) -> dict[str, Any] | None: ...

    def put(self, request_hash: str, **kwargs: Any) -> None: ...


class CacheFirstProvider:
    """On a hit returns the stored response (cache_status 'hit'); on a miss calls ``live``.

    ``recorded`` responses are checked first when reading, then the cache.
    """

    def __init__(
        self,
        cache: ResponseCache,
        live: ModelProvider | None,
        *,
        read: bool = True,
        write: bool = True,
        overwrite: bool = False,
        tracker: BudgetTracker | None = None,
        clock: Callable[[], float] = time.monotonic,
        recorded: Mapping[str, dict[str, Any]] | None = None,
    ) -> None:
        self._cache = cache
        self._recorded = dict(recorded or {})
        self.live = live
        self._read = read
        self._write = write
        self._overwrite = overwrite
        self._tracker = tracker
        self._clock = clock

    def generate(self, request: ModelRequest) -> ModelResponse:
        key = request_hash(request)
        if self._read:
            cached = self._recorded.get(key)
            if cached is None:
                cached = self._cache.get(key)
            if cached is not None:
                response = ModelResponse.model_validate(cached)
                # attempts is set again by the client for this call.
                return response.model_copy(
                    update={"cache_status": "hit", "attempts": 1}
                )
        if self.live is None:
            raise CacheMissNoLiveModel(
                f"No cached response for request_hash={key} and no live model is available"
            )
        started = self._clock()
        response = self.live.generate(request)
        if response.latency_ms == 0:
            elapsed_ms = max(0, round((self._clock() - started) * 1000))
            response = response.model_copy(update={"latency_ms": elapsed_ms})
        if self._write:
            cost = (
                self._tracker.cost_of(response, request.model) if self._tracker else 0.0
            )
            extra: dict[str, Any] = {"overwrite": True} if self._overwrite else {}
            self._cache.put(
                key,
                provider=request.provider,
                model=request.model,
                request=request.model_dump(mode="json"),
                response=response.model_dump(mode="json"),
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=response.latency_ms,
                cost_usd=cost,
                **extra,
            )
        return response
