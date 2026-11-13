"""CacheFirstProvider and build_provider: hits, misses, budget and the provider chain."""

from __future__ import annotations

import os
from typing import Any

import pytest

from agent_replay.llm.cache_provider import CacheFirstProvider
from agent_replay.llm.types import ModelRequest, ModelResponse

FAKE_KEY = "sk-test-0123456789abcdef"


class FakeCache:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.puts: list[tuple[str, dict[str, Any]]] = []

    def get(self, request_hash: str) -> dict[str, Any] | None:
        row = self.rows.get(request_hash)
        return None if row is None else row["response"]

    def put(self, request_hash: str, **kw: Any) -> None:
        self.puts.append((request_hash, kw))
        self.rows.setdefault(request_hash, kw)


class CountingProvider:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls = 0

    def generate(self, request: Any) -> Any:
        self.calls += 1
        return self.response.model_copy(deep=True)


def _req(text: str = "hi") -> Any:
    from agent_replay.llm.types import Message, ModelRequest

    return ModelRequest(
        provider="scripted",
        model="m",
        system="s",
        messages=[Message(role="user", text=text)],
    )


def _resp(text: str = "answer", latency: int = 123, **kw: Any) -> Any:
    from agent_replay.llm.types import ModelResponse

    return ModelResponse(
        text=text, input_tokens=100, output_tokens=50, latency_ms=latency, **kw
    )


def _seed(cache: FakeCache, req: Any, resp: Any) -> str:
    from agent_replay.llm.hashing import request_hash

    h = request_hash(req)
    cache.put(
        h,
        provider=req.provider,
        model=req.model,
        request=req.model_dump(mode="json"),
        response=resp.model_dump(mode="json"),
        latency_ms=resp.latency_ms,
    )
    cache.puts.clear()
    return h


def _cfp(cache: Any, live: Any, **kw: Any) -> Any:
    from agent_replay.llm.cache_provider import CacheFirstProvider

    return CacheFirstProvider(cache, live, **kw)


def _settings(**kw: Any) -> Any:
    from agent_replay.config import Settings

    return Settings(**kw)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in list(os.environ):
        if k.startswith("AGENT_REPLAY_") and k != "AGENT_REPLAY_TEST_DATABASE_URL":
            monkeypatch.delenv(k)
    # The default provider is claude_code: never let it find a real `claude` on PATH.
    monkeypatch.setenv("AGENT_REPLAY_CLAUDE_CLI", "")


@pytest.fixture
def fake_genai(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Replace the SDK client class so no real google-genai client is ever built."""
    from google import genai

    made: list[Any] = []

    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            made.append((a, k))
            self.models = None

    monkeypatch.setattr(genai, "Client", FakeClient)
    return made


# ---- CacheFirstProvider ----------------------------------------------------


def test_hit_returns_cached_response_and_latency_without_live_call() -> None:
    cache, req = FakeCache(), _req()
    _seed(cache, req, _resp("cached", latency=777))
    live = CountingProvider(_resp("live", latency=1))
    out = _cfp(cache, live, read=True).generate(req)
    assert out.cache_status == "hit"
    assert out.text == "cached"
    assert out.latency_ms == 777
    assert live.calls == 0


def test_miss_with_live_calls_once_stores_and_returns_live() -> None:
    from agent_replay.llm.hashing import request_hash

    cache, req = FakeCache(), _req()
    live = CountingProvider(_resp("fresh", latency=42))
    out = _cfp(cache, live, read=True).generate(req)
    assert live.calls == 1
    assert out.cache_status == "live"
    assert out.text == "fresh"
    assert len(cache.puts) == 1
    h, kw = cache.puts[0]
    assert h == request_hash(req)
    assert kw["latency_ms"] == 42
    assert kw["response"]["text"] == "fresh"
    assert kw["response"]["latency_ms"] == 42
    again = _cfp(cache, live, read=True).generate(req)
    assert again.cache_status == "hit" and again.latency_ms == 42
    assert live.calls == 1


def test_miss_without_live_raises_with_hash_and_no_other_call() -> None:
    from agent_replay.llm.hashing import request_hash
    from agent_replay.llm.types import CacheMissNoLiveModel

    cache, req = FakeCache(), _req()
    with pytest.raises(CacheMissNoLiveModel) as ei:
        _cfp(cache, None, read=True).generate(req)
    assert request_hash(req) in str(ei.value)
    assert cache.puts == []


def test_read_false_write_true_always_calls_live_and_writes() -> None:
    cache, req = FakeCache(), _req()
    _seed(cache, req, _resp("old", latency=5))
    live = CountingProvider(_resp("new", latency=9))
    p = _cfp(cache, live, read=False, write=True)
    a, b = p.generate(req), p.generate(req)
    assert live.calls == 2
    assert a.text == b.text == "new"
    assert a.cache_status == "live"
    assert len(cache.puts) >= 1


def test_hit_ignores_tool_call_signature() -> None:
    from agent_replay.llm.types import Message, ModelRequest, ToolCall

    def mk(sig: str | None) -> Any:
        return ModelRequest(
            provider="scripted",
            model="m",
            system="s",
            messages=[
                Message(role="user", text="q"),
                Message(
                    role="model",
                    tool_calls=[
                        ToolCall(id="call_1_0", name="t", args={"a": 1}, signature=sig)
                    ],
                ),
            ],
        )

    cache = FakeCache()
    _seed(cache, mk("AAAA"), _resp("c"))
    live = CountingProvider(_resp("live"))
    out = _cfp(cache, live, read=True).generate(mk("BBBB"))
    assert out.cache_status == "hit" and live.calls == 0


def test_real_repo_roundtrip(conn: Any) -> None:
    from agent_replay.db.cache_repo import CacheRepo

    repo, req = CacheRepo(conn), _req("db")
    live = CountingProvider(_resp("fresh", latency=31))
    assert _cfp(repo, live, read=True).generate(req).cache_status == "live"
    out = _cfp(repo, None, read=True).generate(req)
    assert (out.cache_status, out.text, out.latency_ms) == ("hit", "fresh", 31)
    assert live.calls == 1


# ---- budget interplay -------------------------------------------------------


def test_budget_zero_cache_hit_still_served() -> None:
    from agent_replay.llm.budget import BudgetGuardProvider

    cache, req = FakeCache(), _req()
    _seed(cache, req, _resp("cached", latency=8))
    live = CountingProvider(_resp("live"))
    out = _cfp(cache, BudgetGuardProvider(live, 0.0), read=True).generate(req)
    assert out.cache_status == "hit" and out.text == "cached"
    assert live.calls == 0


def test_budget_zero_miss_raises_and_writes_nothing() -> None:
    from agent_replay.llm.budget import BudgetGuardProvider
    from agent_replay.llm.types import BudgetExceeded

    cache = FakeCache()
    live = CountingProvider(_resp("live"))
    p = _cfp(cache, BudgetGuardProvider(live, 0.0), read=True)
    with pytest.raises(BudgetExceeded):
        p.generate(_req())
    assert live.calls == 0
    assert cache.puts == [] and cache.rows == {}


# ---- build_provider ---------------------------------------------------------


def test_gemini_with_key_chain_and_settings(fake_genai: list[Any]) -> None:
    from agent_replay.llm.budget import BudgetGuardProvider
    from agent_replay.llm.cache_provider import CacheFirstProvider
    from agent_replay.llm.factory import build_provider
    from agent_replay.llm.gemini import GeminiProvider
    from agent_replay.llm.ratelimit import RateLimitedProvider

    s = _settings(
        provider="gemini",
        gemini_api_key=FAKE_KEY,
        rpm=7,
        rpd=77,
        run_budget_usd=0.42,
        price_input_per_m=1.25,
        price_output_per_m=3.5,
    )
    p = build_provider(s, mode="record", cache=FakeCache())
    assert isinstance(p, CacheFirstProvider)
    guard = p.live
    assert isinstance(guard, BudgetGuardProvider)
    limited = guard._provider
    assert isinstance(limited, RateLimitedProvider)
    assert isinstance(limited._provider, GeminiProvider)
    assert limited._limiter._rpm == 7 and limited._limiter._rpd == 77
    assert guard._budget_usd == 0.42
    assert guard.tracker.price_input_per_m == 1.25
    assert guard.tracker.price_output_per_m == 3.5


def test_budget_refusal_does_not_acquire_rate_limiter(
    monkeypatch: pytest.MonkeyPatch, fake_genai: list[Any]
) -> None:
    from agent_replay.llm.factory import build_provider
    from agent_replay.llm.ratelimit import RateLimiter
    from agent_replay.llm.types import BudgetExceeded

    acquired: list[int] = []
    monkeypatch.setattr(RateLimiter, "acquire", lambda self: acquired.append(1))
    p = build_provider(
        _settings(provider="gemini", gemini_api_key=FAKE_KEY, run_budget_usd=0),
        mode="record",
        cache=FakeCache(),
    )
    with pytest.raises(BudgetExceeded):
        p.generate(_req())
    assert acquired == []


class _Live:
    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(text="ok", input_tokens=1, output_tokens=1)


class _Cache:
    def __init__(self) -> None:
        self.puts: list[dict[str, Any]] = []

    def get(self, request_hash: str) -> dict[str, Any] | None:
        raise AssertionError("must not read")

    def put(self, request_hash: str, **kwargs: Any) -> None:
        self.puts.append(kwargs)


def test_live_latency_measured_and_cached() -> None:
    ticks = iter([10.0, 10.25])
    cache = _Cache()
    p = CacheFirstProvider(
        cache, _Live(), read=False, overwrite=True, clock=lambda: next(ticks)
    )
    req = ModelRequest(provider="x", model="m", system="s", messages=[])
    r = p.generate(req)
    assert r.latency_ms == 250
    assert cache.puts[0]["latency_ms"] == 250
    assert cache.puts[0]["overwrite"] is True
