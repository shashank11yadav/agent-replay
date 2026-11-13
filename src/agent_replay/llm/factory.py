"""Builds the provider chain: cache -> budget guard -> rate limiter -> provider.

Two entry points, never stacked:

- ``build_live_provider`` returns only the live inner provider (whatever the registered
  provider spec builds, rate-limited when the spec says so), with no cache and no budget
  guard. This is what ``runtime.runner.record_run`` / ``replay_run`` take: the runner adds
  cache, budget guard and LLMClient itself.
- ``build_provider`` returns the full cache-first chain for callers that do not go through
  the runner (e.g. the judge). Never pass its output to the runner: spend would be guarded
  and recorded twice, and two cache layers would run.
"""

from __future__ import annotations

from typing import Any, Literal

from agent_replay.config import Settings
from agent_replay.llm.budget import BudgetGuardProvider, BudgetTracker
from agent_replay.llm.cache_provider import CacheFirstProvider, ResponseCache
from agent_replay.llm.client import LLMClient
from agent_replay.llm.provider import ModelProvider
from agent_replay.llm.ratelimit import RateLimitedProvider, RateLimiter
from agent_replay.llm.registry import get_provider_spec

__all__ = [
    "build_live_provider",
    "build_provider",
    "cache_first_chain",
    "make_client",
]

_trackers: dict[tuple[object, ...], BudgetTracker] = {}
_limiters: dict[tuple[int, int], RateLimiter] = {}


def get_tracker(settings: Settings) -> BudgetTracker:
    """The one BudgetTracker for this process (per price configuration).

    Its table prices settings.model at the agent prices and settings.judge_model at the
    judge prices, so agent and judge calls share one spend at their own rates.
    """
    agent = (settings.price_input_per_m, settings.price_output_per_m)
    judge = (settings.judge_price_input_per_m, settings.judge_price_output_per_m)
    key = (agent, judge, settings.model, settings.judge_model)
    if key not in _trackers:
        _trackers[key] = BudgetTracker(
            *agent,
            model_prices={settings.judge_model: judge, settings.model: agent},
        )
    return _trackers[key]


def get_limiter(settings: Settings) -> RateLimiter:
    """The one RateLimiter for this process (per rpm/rpd), shared by agent and judge."""
    key = (settings.rpm, settings.rpd)
    if key not in _limiters:
        _limiters[key] = RateLimiter(*key)
    return _limiters[key]


def make_client(provider: ModelProvider, settings: Settings) -> LLMClient:
    """The LLMClient for agent and judge calls, with the retry settings and shared tracker."""
    return LLMClient(
        provider,
        max_attempts=settings.max_attempts,
        backoff_base_s=settings.backoff_base_s,
        tracker=get_tracker(settings),
    )


def _check_mode(mode: str) -> None:
    if mode not in ("record", "replay"):
        raise ValueError("mode must be 'record' or 'replay'")


def build_live_provider(
    settings: Settings, mode: Literal["record", "replay"]
) -> ModelProvider | None:
    """The live inner provider only, from the registered spec for ``settings.provider``.

    Returns whatever the spec builds (None in replay mode without a credential or tool;
    ValueError in record mode), wrapped in the shared rate limiter when the spec asks for it.
    An unknown provider name raises ValueError listing the registered names. No cache,
    no budget guard.
    """
    _check_mode(mode)
    spec = get_provider_spec(settings.provider)
    inner = spec.build(settings, mode)
    if inner is not None and spec.rate_limited:
        inner = RateLimitedProvider(inner, get_limiter(settings))
    return inner


def build_provider(
    settings: Settings, mode: Literal["record", "replay"], *, cache: Any
) -> CacheFirstProvider:
    """Full chain: CacheFirst -> BudgetGuard -> build_live_provider output."""
    _check_mode(mode)
    return cache_first_chain(
        settings, mode, build_live_provider(settings, mode), cache=cache
    )


def cache_first_chain(
    settings: Settings,
    mode: Literal["record", "replay"],
    inner: ModelProvider | None,
    *,
    cache: Any,
) -> CacheFirstProvider:
    """CacheFirst -> BudgetGuard -> ``inner`` (a live provider, or None for cache only)."""
    _check_mode(mode)
    live: ModelProvider | None = None
    if inner is not None:
        live = BudgetGuardProvider(
            inner, settings.run_budget_usd, get_tracker(settings)
        )
    response_cache: ResponseCache = cache
    return CacheFirstProvider(
        response_cache,
        live,
        read=mode == "replay",
        write=True,
        overwrite=mode == "record",
        tracker=get_tracker(settings),
    )
