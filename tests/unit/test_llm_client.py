"""Rate limiter, budget guard, retrying client and its log line.

Cost fixtures: 1_000_000 input tokens cost $0.30 at the default prices.
"""

from __future__ import annotations

import logging

import pytest
from support import ROOT

from agent_replay.llm import (
    BudgetExceeded,
    Message,
    ModelRequest,
    ModelResponse,
    PermanentModelError,
    QuotaExhausted,
    RateLimitError,
    ScriptedProvider,
    TransientError,
)
from agent_replay.llm.budget import (
    BudgetGuardProvider,
    BudgetTracker,
    estimate_cost_usd,
)
from agent_replay.llm.client import LLMClient
from agent_replay.llm.ratelimit import RateLimitedProvider, RateLimiter

KEY = "AIzaSyFAKE-SECRET-KEY-1234567890"
SRC = ROOT / "src"
M = 1_000_000


def req() -> ModelRequest:
    return ModelRequest(
        provider="fake",
        model="m",
        system="s",
        messages=[Message(role="user", text="hi")],
    )


def resp(cost_in_m: float = 0.0, **kw) -> ModelResponse:
    """Response costing cost_in_m * 0.30 USD (input tokens in millions)."""
    return ModelResponse(text="ok", input_tokens=int(cost_in_m * M), **kw)


def usd(x: float) -> ModelResponse:
    return resp(x / 0.30)


class FakeTime:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


class Flaky:
    """Provider raising queued exceptions, then returning a response."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


# ---- rate limiter ----------------------------------------------------------------


def test_third_acquire_within_minute_sleeps_until_oldest_is_60s_old():
    t = FakeTime()
    rl = RateLimiter(rpm=2, rpd=100, clock=t.clock, sleep=t.sleep)
    rl.acquire()
    t.now += 10
    rl.acquire()
    assert t.sleeps == []
    t.now += 5  # 15 s after first, oldest is 15 s old
    rl.acquire()
    assert len(t.sleeps) == 1
    assert t.sleeps[0] == pytest.approx(45.0, abs=0.5)
    assert t.now == pytest.approx(1060.0, abs=0.5)


def test_rpd_exhaustion_raises_quota_without_sleep():
    t = FakeTime()
    rl = RateLimiter(rpm=1000, rpd=3, clock=t.clock, sleep=t.sleep)
    for _ in range(3):
        rl.acquire()
    with pytest.raises(QuotaExhausted):
        rl.acquire()
    assert t.sleeps == []


def test_rate_limited_provider_acquires_once_per_call():
    class Counting:
        n = 0

        def acquire(self) -> None:
            self.n += 1

    lim = Counting()
    inner = ScriptedProvider([resp(), resp()])
    p = RateLimitedProvider(inner, lim)  # type: ignore[arg-type]
    p.generate(req())
    assert lim.n == 1 and len(inner.requests) == 1
    p.generate(req())
    assert lim.n == 2 and len(inner.requests) == 2


# ---- cost ------------------------------------------------------------------------


def test_estimate_cost():
    assert estimate_cost_usd(
        input_tokens=3000,
        output_tokens=500,
        price_input_per_m=0.30,
        price_output_per_m=2.50,
    ) == pytest.approx(0.00215)
    assert estimate_cost_usd(
        input_tokens=M, output_tokens=M, price_input_per_m=0.30, price_output_per_m=2.50
    ) == pytest.approx(2.80)


# ---- budget guard ----------------------------------------------------------------


def test_budget_projection_blocks_second_060_call():
    inner = ScriptedProvider([usd(0.60), usd(0.60)])
    g = BudgetGuardProvider(inner, budget_usd=1.00)
    assert g.generate(req()).text == "ok"
    with pytest.raises(BudgetExceeded):
        g.generate(req())
    assert len(inner.requests) == 1


def test_budget_three_030_calls_then_blocked():
    inner = ScriptedProvider([usd(0.30)] * 4)
    g = BudgetGuardProvider(inner, budget_usd=1.00)
    for _ in range(3):
        g.generate(req())
    with pytest.raises(BudgetExceeded):
        g.generate(req())
    assert len(inner.requests) == 3


def test_budget_overspend_response_returned_then_refused():
    inner = ScriptedProvider([usd(1.20), usd(0.01)])
    g = BudgetGuardProvider(inner, budget_usd=1.00)
    assert g.generate(req()).input_tokens == 4 * M
    with pytest.raises(BudgetExceeded):
        g.generate(req())
    assert len(inner.requests) == 1


def test_budget_zero_refuses_first_call():
    inner = ScriptedProvider([usd(0.0)])
    g = BudgetGuardProvider(inner, budget_usd=0)
    with pytest.raises(BudgetExceeded):
        g.generate(req())
    assert inner.requests == []


def test_rate_limit_error_leaves_spend_unchanged():
    tr = BudgetTracker()
    inner = Flaky([usd(0.30), RateLimitError("429")])
    g = BudgetGuardProvider(inner, budget_usd=5.00, tracker=tr)
    g.generate(req())
    before = tr.spent_usd
    assert before == pytest.approx(0.30)
    with pytest.raises(RateLimitError):
        g.generate(req())
    assert tr.spent_usd == before


def test_shared_tracker_counts_both_guards():
    tr = BudgetTracker()
    a = BudgetGuardProvider(ScriptedProvider([usd(0.60)]), budget_usd=1.00, tracker=tr)
    inner_b = ScriptedProvider([usd(0.60)])
    b = BudgetGuardProvider(inner_b, budget_usd=1.00, tracker=tr)
    a.generate(req())
    with pytest.raises(BudgetExceeded):
        b.generate(req())
    assert inner_b.requests == []


# ---- client retries --------------------------------------------------------------


@pytest.mark.parametrize("exc", [QuotaExhausted("q"), BudgetExceeded(1.0, 1.0)])
def test_no_retry_for_quota_or_budget(exc):
    t = FakeTime()
    p = Flaky([exc, resp()])
    c = LLMClient(p, max_attempts=5, backoff_base_s=2, sleep=t.sleep)
    with pytest.raises(type(exc)):
        c.generate(req())
    assert p.calls == 1 and t.sleeps == []


@pytest.mark.parametrize("exc_t", [RateLimitError, TransientError])
def test_retries_with_exponential_backoff(exc_t):
    t = FakeTime()
    p = Flaky([exc_t("a"), exc_t("b"), resp()])
    c = LLMClient(p, max_attempts=4, backoff_base_s=2, sleep=t.sleep)
    r = c.generate(req())
    assert p.calls == 3
    assert t.sleeps == [2, 4]
    assert r.attempts == 3


@pytest.mark.parametrize("exc_t", [RateLimitError, TransientError])
def test_exhaustion_raises_last_error(exc_t):
    t = FakeTime()
    p = Flaky([exc_t("first"), exc_t("second"), exc_t("third"), resp()])
    c = LLMClient(p, max_attempts=3, backoff_base_s=1, sleep=t.sleep)
    with pytest.raises(exc_t, match="third"):
        c.generate(req())
    assert p.calls == 3
    assert t.sleeps == [1, 2]


def test_permanent_error_immediate():
    t = FakeTime()
    p = Flaky([PermanentModelError("bad"), resp()])
    c = LLMClient(p, max_attempts=5, backoff_base_s=2, sleep=t.sleep)
    with pytest.raises(PermanentModelError):
        c.generate(req())
    assert p.calls == 1 and t.sleeps == []


def test_attempts_and_latency_include_backoff():
    t = FakeTime()
    p = Flaky([TransientError("x"), TransientError("y"), resp(latency_ms=120)])
    c = LLMClient(p, max_attempts=3, backoff_base_s=2, sleep=t.sleep)
    r = c.generate(req())
    assert r.attempts == 3
    assert r.latency_ms == 120 + 6000


# ---- logging ---------------------------------------------------------------------


def _llm_records(caplog):
    return [r for r in caplog.records if r.name == "agent_replay.llm"]


def test_one_log_record_per_call_with_fields(caplog):
    t = FakeTime()
    tr = BudgetTracker()
    inner = ScriptedProvider([resp(1.0, output_tokens=1000, latency_ms=50), usd(0.30)])
    c = LLMClient(
        BudgetGuardProvider(inner, budget_usd=10, tracker=tr),
        max_attempts=3,
        backoff_base_s=2,
        sleep=t.sleep,
        tracker=tr,
    )
    with caplog.at_level(logging.DEBUG, logger="agent_replay.llm"):
        c.generate(req())
    recs = _llm_records(caplog)
    assert len(recs) == 1
    r = recs[0]
    for f in (
        "model",
        "request_hash",
        "cache_status",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "attempts",
        "cost_usd",
        "spent_usd",
    ):
        assert hasattr(r, f), f
    assert r.model == "m"
    assert r.input_tokens == M and r.output_tokens == 1000
    assert r.cache_status == "live"
    assert r.attempts == 1
    assert r.cost_usd == pytest.approx(0.30 + 1000 * 2.5 / M)
    assert r.spent_usd == pytest.approx(r.cost_usd)
    with caplog.at_level(logging.DEBUG, logger="agent_replay.llm"):
        c.generate(req())
    assert len(_llm_records(caplog)) == 2


def test_cache_hit_log_zero_cost_and_spend_unchanged(caplog):
    t = FakeTime()
    tr = BudgetTracker()
    inner = ScriptedProvider([usd(0.30), resp(2.0, cache_status="hit")])
    c = LLMClient(
        BudgetGuardProvider(inner, budget_usd=10, tracker=tr),
        max_attempts=3,
        backoff_base_s=2,
        sleep=t.sleep,
        tracker=tr,
    )
    c.generate(req())
    spent = tr.spent_usd
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="agent_replay.llm"):
        c.generate(req())
    (r,) = _llm_records(caplog)
    assert r.cache_status == "hit"
    assert r.cost_usd == 0
    assert tr.spent_usd == spent
    assert r.spent_usd == pytest.approx(spent)


def test_logs_never_contain_api_key(caplog, monkeypatch):
    monkeypatch.setenv("AGENT_REPLAY_GEMINI_API_KEY", KEY)
    t = FakeTime()
    p = Flaky([TransientError("boom"), resp()])
    c = LLMClient(p, max_attempts=3, backoff_base_s=1, sleep=t.sleep)
    with caplog.at_level(logging.DEBUG):
        c.generate(req())
    assert caplog.records
    for r in caplog.records:
        assert KEY not in r.getMessage()
        assert KEY not in str(r.__dict__)


def test_client_does_not_import_google_genai():
    src = (SRC / "agent_replay" / "llm" / "client.py").read_text()
    assert "google.genai" not in src
    assert "from google import genai" not in src
