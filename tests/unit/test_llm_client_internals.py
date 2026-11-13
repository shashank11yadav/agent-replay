"""Unit tests for the rate limiter, budget tracker and client internals."""

import logging

import pytest

from agent_replay.llm import (
    Message,
    ModelRequest,
    ModelResponse,
    PermanentModelError,
    QuotaExhausted,
    ScriptedProvider,
    TransientError,
)
from agent_replay.llm.budget import BudgetGuardProvider, BudgetTracker
from agent_replay.llm.client import LLMClient
from agent_replay.llm.ratelimit import RateLimiter


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


def _req() -> ModelRequest:
    return ModelRequest(
        provider="p", model="m", system="s", messages=[Message(role="user", text="hi")]
    )


def test_limiter_rejects_nonpositive_rpm():
    with pytest.raises(ValueError):
        RateLimiter(rpm=0, rpd=10)


def test_limiter_window_frees_after_60s_without_sleep():
    t = FakeTime()
    rl = RateLimiter(rpm=1, rpd=10, clock=t.clock, sleep=t.sleep)
    rl.acquire()
    t.now += 60
    rl.acquire()
    assert t.sleeps == []


def test_limiter_day_window_rolls_after_24h():
    t = FakeTime()
    rl = RateLimiter(rpm=10, rpd=1, clock=t.clock, sleep=t.sleep)
    rl.acquire()
    with pytest.raises(QuotaExhausted):
        rl.acquire()
    t.now += 86_400
    rl.acquire()


def test_refused_acquire_is_not_counted():
    t = FakeTime()
    rl = RateLimiter(rpm=10, rpd=1, clock=t.clock, sleep=t.sleep)
    rl.acquire()
    for _ in range(3):
        with pytest.raises(QuotaExhausted):
            rl.acquire()
    t.now += 86_400
    rl.acquire()


def test_tracker_hit_costs_zero_and_tracks_largest():
    tr = BudgetTracker()
    assert tr.cost_of(ModelResponse(input_tokens=10**6, cache_status="hit")) == 0
    tr.record(0.2)
    tr.record(0.5)
    tr.record(0.1)
    assert tr.spent_usd == pytest.approx(0.8)
    assert tr.largest_call_usd == pytest.approx(0.5)


def test_guard_does_not_raise_largest_on_hit():
    tr = BudgetTracker()
    inner = ScriptedProvider([ModelResponse(input_tokens=10**7, cache_status="hit")])
    BudgetGuardProvider(inner, budget_usd=1.0, tracker=tr).generate(_req())
    assert tr.spent_usd == 0 and tr.largest_call_usd == 0


def test_client_logs_failure_without_exception_text(caplog):
    class Boom:
        def generate(self, request: ModelRequest) -> ModelResponse:
            raise PermanentModelError("secret-detail")

    c = LLMClient(Boom(), max_attempts=3, backoff_base_s=1, sleep=lambda s: None)
    with (
        caplog.at_level(logging.DEBUG, logger="agent_replay.llm"),
        pytest.raises(PermanentModelError),
    ):
        c.generate(_req())
    (rec,) = [r for r in caplog.records if r.name == "agent_replay.llm"]
    assert rec.error == "PermanentModelError"
    assert rec.attempts == 1
    assert "secret-detail" not in str(rec.__dict__)


def test_client_logs_once_after_retries(caplog):
    t = FakeTime()
    outcomes: list[Exception | ModelResponse] = [TransientError("x"), ModelResponse()]

    class P:
        def generate(self, request: ModelRequest) -> ModelResponse:
            o = outcomes.pop(0)
            if isinstance(o, Exception):
                raise o
            return o

    c = LLMClient(P(), max_attempts=3, backoff_base_s=1, sleep=t.sleep)
    with caplog.at_level(logging.DEBUG, logger="agent_replay.llm"):
        c.generate(_req())
    recs = [r for r in caplog.records if r.name == "agent_replay.llm"]
    assert len(recs) == 1 and recs[0].attempts == 2 and recs[0].error is None


def test_client_rejects_zero_attempts():
    with pytest.raises(ValueError):
        LLMClient(ScriptedProvider([]), max_attempts=0)
