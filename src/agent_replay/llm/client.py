"""The single model client: retries with backoff and one structured log line per call."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from agent_replay.llm.budget import BudgetTracker
from agent_replay.llm.hashing import request_hash
from agent_replay.llm.provider import ModelProvider
from agent_replay.llm.types import (
    ModelRequest,
    ModelResponse,
    RateLimitError,
    TransientError,
)

logger = logging.getLogger("agent_replay.llm")

_RETRYABLE = (RateLimitError, TransientError)


class LLMClient:
    """Retries rate-limit and transient errors; everything else is raised at once.

    Spend is recorded by BudgetGuardProvider; the client only reads the shared
    tracker for the log line (spent_usd is None when no tracker is given).
    """

    def __init__(
        self,
        provider: ModelProvider,
        max_attempts: int = 3,
        backoff_base_s: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        tracker: BudgetTracker | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._provider = provider
        self._max_attempts = max_attempts
        self._backoff_base_s = backoff_base_s
        self._sleep = sleep
        self._tracker = tracker
        self._pricing = tracker if tracker is not None else BudgetTracker()

    def generate(self, request: ModelRequest) -> ModelResponse:
        slept_s = 0.0
        attempt = 1
        while True:
            try:
                response = self._provider.generate(request)
                break
            except _RETRYABLE as exc:
                if attempt >= self._max_attempts:
                    self._log(request, None, attempt, slept_s, type(exc).__name__)
                    raise
                delay = self._backoff_base_s * 2 ** (attempt - 1)
                self._sleep(delay)
                slept_s += delay
                attempt += 1
            except Exception as exc:
                self._log(request, None, attempt, slept_s, type(exc).__name__)
                raise
        response = response.model_copy(
            update={
                "attempts": attempt,
                "latency_ms": response.latency_ms + round(slept_s * 1000),
            }
        )
        self._log(request, response, attempt, slept_s, None)
        return response

    def _log(
        self,
        request: ModelRequest,
        response: ModelResponse | None,
        attempts: int,
        slept_s: float,
        error: str | None,
    ) -> None:
        # Only scalar fields and the exception class name: never exception text,
        # which may carry SDK details.
        fields = {
            "model": request.model,
            "request_hash": request_hash(request),
            "cache_status": response.cache_status if response else None,
            "input_tokens": response.input_tokens if response else 0,
            "output_tokens": response.output_tokens if response else 0,
            "latency_ms": response.latency_ms if response else round(slept_s * 1000),
            "attempts": attempts,
            "cost_usd": self._pricing.cost_of(response, request.model)
            if response
            else 0.0,
            "spent_usd": self._tracker.spent_usd if self._tracker else None,
            "error": error,
        }
        message = " ".join(f"{k}=%s" for k in fields)
        level = logging.INFO if error is None else logging.WARNING
        logger.log(level, "llm_call " + message, *fields.values(), extra=fields)
