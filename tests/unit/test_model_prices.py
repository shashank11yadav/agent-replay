"""Per-model prices in one shared process budget.

Pinned costs for 1000 input and 500 output tokens:
gemini/scripted default (0.30, 2.50) -> 0.00155
claude-haiku-5-5 (0.10, 0.50)        -> 0.00035
claude-sonnet-5-5 (2.00, 10.00)      -> 0.007
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import ValidationError

from agent_replay.config import Settings
from agent_replay.llm import ModelRequest, ModelResponse, ScriptedProvider, factory
from agent_replay.llm.budget import BudgetGuardProvider, BudgetTracker
from agent_replay.llm.cache_provider import CacheFirstProvider
from agent_replay.llm.client import LLMClient
from agent_replay.llm.types import BudgetExceeded
from agent_replay.runtime.contexts import RecordingContext
from agent_replay.tracing import new_tracer

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
GEMINI = "gemini-3.5-flash-lite"

GEMINI_COST = 0.00155
HAIKU_COST = 0.00035
SONNET_COST = 0.007


@pytest.fixture(autouse=True)
def clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key in list(os.environ):
        if key.startswith("AGENT_REPLAY_") and key not in (
            "AGENT_REPLAY_GEMINI_API_KEY",
            "AGENT_REPLAY_CLAUDE_CLI",
        ):
            monkeypatch.delenv(key)
    monkeypatch.setattr(factory, "_trackers", {})
    yield


def resp(i: int = 1000, o: int = 500, status: str = "live") -> ModelResponse:
    return ModelResponse(text="x", input_tokens=i, output_tokens=o, cache_status=status)  # type: ignore[arg-type]


def req(model: str, provider: str = "claude_code") -> ModelRequest:
    return ModelRequest(provider=provider, model=model, system="s", messages=[])


def table_tracker() -> BudgetTracker:
    return BudgetTracker(  # type: ignore[call-arg]
        0.10,
        0.50,
        model_prices={HAIKU: (0.10, 0.50), SONNET: (2.00, 10.00)},
    )


def claude_settings(**kw: Any) -> Settings:
    return Settings(provider="claude_code", **kw)


# ---- BudgetTracker.cost_of ------------------------------------------------------


def test_cost_of_uses_the_models_own_prices() -> None:
    t = table_tracker()
    assert t.cost_of(resp(), SONNET) == pytest.approx(SONNET_COST)  # type: ignore[call-arg]
    assert t.cost_of(resp(), HAIKU) == pytest.approx(HAIKU_COST)  # type: ignore[call-arg]


def test_cost_of_unknown_model_or_none_uses_default_pair() -> None:
    t = BudgetTracker(
        0.30,
        2.50,
        model_prices={SONNET: (2.00, 10.00)},  # type: ignore[call-arg]
    )
    assert t.cost_of(resp(), "not-in-table") == pytest.approx(GEMINI_COST)  # type: ignore[call-arg]
    assert t.cost_of(resp(), None) == pytest.approx(GEMINI_COST)  # type: ignore[call-arg]


def test_one_argument_cost_of_unchanged() -> None:
    assert BudgetTracker().cost_of(resp()) == pytest.approx(GEMINI_COST)
    assert BudgetTracker(1.0, 2.0).cost_of(resp(1_000_000, 1_000_000)) == (
        pytest.approx(3.0)
    )
    # one-arg call ignores the table: default pair
    assert table_tracker().cost_of(resp()) == pytest.approx(HAIKU_COST)


def test_cache_hits_cost_zero_for_any_model() -> None:
    t = table_tracker()
    hit = resp(status="hit")
    assert t.cost_of(hit) == 0.0
    assert t.cost_of(hit, SONNET) == 0.0  # type: ignore[call-arg]
    assert t.cost_of(hit, "zzz") == 0.0  # type: ignore[call-arg]
    assert t.cost_of(hit, None) == 0.0  # type: ignore[call-arg]


def test_tracker_exposes_model_prices_table() -> None:
    t = table_tracker()
    assert dict(t.model_prices) == {HAIKU: (0.10, 0.50), SONNET: (2.00, 10.00)}  # type: ignore[attr-defined]


# ---- judge price settings ---------------------------------------------------------


def judge_prices(s: Settings) -> tuple[float, float]:
    return (s.judge_price_input_per_m, s.judge_price_output_per_m)  # type: ignore[attr-defined]


def test_claude_code_defaults() -> None:
    s = claude_settings()
    assert (s.price_input_per_m, s.price_output_per_m) == (0.10, 0.50)
    assert judge_prices(s) == (2.00, 10.00)


def test_gemini_and_scripted_judge_prices_equal_agent_prices() -> None:
    for provider in ("gemini", "scripted"):
        s = Settings(provider=provider)
        assert judge_prices(s) == (0.30, 2.50)
        assert judge_prices(s) == (s.price_input_per_m, s.price_output_per_m)


def test_judge_equals_agent_model_follows_explicit_agent_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_REPLAY_PRICE_INPUT_PER_M", "9")
    monkeypatch.setenv("AGENT_REPLAY_PRICE_OUTPUT_PER_M", "11")
    assert judge_prices(Settings(provider="gemini")) == (9.0, 11.0)


def test_env_judge_prices_win_over_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_REPLAY_JUDGE_PRICE_INPUT_PER_M", "7")
    monkeypatch.setenv("AGENT_REPLAY_JUDGE_PRICE_OUTPUT_PER_M", "8")
    assert judge_prices(claude_settings()) == (7.0, 8.0)
    assert judge_prices(Settings()) == (7.0, 8.0)


def test_only_one_judge_price_set_overrides_only_that_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_REPLAY_JUDGE_PRICE_OUTPUT_PER_M", "8")
    assert judge_prices(claude_settings()) == (2.00, 8.0)
    monkeypatch.delenv("AGENT_REPLAY_JUDGE_PRICE_OUTPUT_PER_M")
    monkeypatch.setenv("AGENT_REPLAY_JUDGE_PRICE_INPUT_PER_M", "7")
    assert judge_prices(claude_settings()) == (7.0, 10.00)


def test_judge_model_in_table_uses_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_REPLAY_JUDGE_MODEL", HAIKU)
    monkeypatch.setenv("AGENT_REPLAY_MODEL", SONNET)
    s = claude_settings()
    assert (s.price_input_per_m, s.price_output_per_m) == (2.00, 10.00)
    assert judge_prices(s) == (0.10, 0.50)


def test_judge_model_unknown_and_different_uses_provider_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # not the agent prices: the provider's default_prices (claude_code default 0.10/0.50)
    monkeypatch.setenv("AGENT_REPLAY_MODEL", SONNET)
    monkeypatch.setenv("AGENT_REPLAY_JUDGE_MODEL", "zzz")
    s = claude_settings()
    assert (s.price_input_per_m, s.price_output_per_m) == (2.00, 10.00)
    assert judge_prices(s) == (0.10, 0.50)


def test_judge_unknown_model_with_registered_dummy_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_replay.llm import registry

    registry.register(
        registry.ProviderSpec(
            name="dummy36",
            build=lambda settings, mode: None,
            needs_credential_to_record=False,
            default_model="m",
            default_judge_model=None,
            default_prices=(5.0, 6.0),
            model_prices={"m": (1.0, 2.0), "j": (3.0, 4.0)},
        )
    )
    try:
        monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "dummy36")
        s = Settings()
        assert judge_prices(s) == (1.0, 2.0)  # judge == model, table hit
        monkeypatch.setenv("AGENT_REPLAY_JUDGE_MODEL", "j")
        assert judge_prices(Settings()) == (3.0, 4.0)
        monkeypatch.setenv("AGENT_REPLAY_JUDGE_MODEL", "other")
        assert judge_prices(Settings()) == (5.0, 6.0)
        monkeypatch.setenv("AGENT_REPLAY_PRICE_INPUT_PER_M", "9")
        monkeypatch.setenv("AGENT_REPLAY_JUDGE_MODEL", "m")
        assert judge_prices(Settings()) == (9.0, 2.0)
    finally:
        registry.unregister("dummy36")


@pytest.mark.parametrize(
    "var",
    [
        "AGENT_REPLAY_JUDGE_PRICE_INPUT_PER_M",
        "AGENT_REPLAY_JUDGE_PRICE_OUTPUT_PER_M",
    ],
)
def test_negative_judge_price_is_a_validation_error(
    var: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(var, "-0.01")
    with pytest.raises(ValidationError):
        Settings()


def test_zero_judge_price_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_REPLAY_JUDGE_PRICE_INPUT_PER_M", "0")
    monkeypatch.setenv("AGENT_REPLAY_JUDGE_PRICE_OUTPUT_PER_M", "0")
    assert judge_prices(claude_settings()) == (0.0, 0.0)


# ---- get_tracker -------------------------------------------------------------------


def test_get_tracker_shared_for_equal_configuration() -> None:
    s = claude_settings()
    assert factory.get_tracker(s) is factory.get_tracker(claude_settings())
    assert factory.get_tracker(s) is not factory.get_tracker(
        Settings(provider="gemini")
    )


def test_get_tracker_table_maps_models_to_prices() -> None:
    t = factory.get_tracker(claude_settings())
    assert t.cost_of(resp(), HAIKU) == pytest.approx(HAIKU_COST)  # type: ignore[call-arg]
    assert t.cost_of(resp(), SONNET) == pytest.approx(SONNET_COST)  # type: ignore[call-arg]
    # default pair is the agent prices
    assert t.cost_of(resp(), None) == pytest.approx(HAIKU_COST)  # type: ignore[call-arg]
    assert t.cost_of(resp(), "other") == pytest.approx(HAIKU_COST)  # type: ignore[call-arg]
    assert t.cost_of(resp()) == pytest.approx(HAIKU_COST)
    table = dict(t.model_prices)  # type: ignore[attr-defined]
    assert table[HAIKU] == (0.10, 0.50)
    assert table[SONNET] == (2.00, 10.00)


def test_get_tracker_table_follows_overridden_models_and_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_REPLAY_MODEL", "a")
    monkeypatch.setenv("AGENT_REPLAY_JUDGE_MODEL", "j")
    monkeypatch.setenv("AGENT_REPLAY_PRICE_INPUT_PER_M", "1")
    monkeypatch.setenv("AGENT_REPLAY_PRICE_OUTPUT_PER_M", "2")
    monkeypatch.setenv("AGENT_REPLAY_JUDGE_PRICE_INPUT_PER_M", "3")
    monkeypatch.setenv("AGENT_REPLAY_JUDGE_PRICE_OUTPUT_PER_M", "4")
    t = factory.get_tracker(Settings())
    r = resp(1_000_000, 1_000_000)
    assert t.cost_of(r, "a") == pytest.approx(3.0)  # type: ignore[call-arg]
    assert t.cost_of(r, "j") == pytest.approx(7.0)  # type: ignore[call-arg]
    assert t.cost_of(r, None) == pytest.approx(3.0)  # type: ignore[call-arg]


def test_get_tracker_differs_when_only_judge_prices_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = factory.get_tracker(claude_settings())
    monkeypatch.setenv("AGENT_REPLAY_JUDGE_PRICE_INPUT_PER_M", "5")
    b = factory.get_tracker(claude_settings())
    assert a is not b
    assert b.cost_of(resp(1_000_000, 0), SONNET) == pytest.approx(5.0)  # type: ignore[call-arg]


# ---- every consumer prices by request.model -----------------------------------------


def test_budget_guard_records_spend_by_request_model() -> None:
    t = factory.get_tracker(claude_settings())
    g = BudgetGuardProvider(ScriptedProvider([resp(), resp()]), 100.0, t)
    g.generate(req(SONNET))
    assert t.spent_usd == pytest.approx(SONNET_COST)
    g.generate(req(HAIKU))
    assert t.spent_usd == pytest.approx(SONNET_COST + HAIKU_COST)
    assert t.largest_call_usd == pytest.approx(SONNET_COST)


class RecCache:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def get(self, request_hash: str) -> dict[str, Any] | None:
        return None

    def put(self, request_hash: str, **kwargs: Any) -> None:
        self.rows.append(kwargs)


def test_cache_row_cost_usd_priced_by_request_model() -> None:
    t = factory.get_tracker(claude_settings())
    cache = RecCache()
    p = CacheFirstProvider(cache, ScriptedProvider([resp(), resp()]), tracker=t)
    p.generate(req(SONNET))
    p.generate(req(HAIKU))
    assert [r["model"] for r in cache.rows] == [SONNET, HAIKU]
    assert cache.rows[0]["cost_usd"] == pytest.approx(SONNET_COST)
    assert cache.rows[1]["cost_usd"] == pytest.approx(HAIKU_COST)


def test_client_log_line_cost_usd_priced_by_request_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    t = factory.get_tracker(claude_settings())
    c = LLMClient(ScriptedProvider([resp(), resp()]), tracker=t)
    with caplog.at_level(logging.INFO, logger="agent_replay.llm"):
        c.generate(req(SONNET))
        c.generate(req(HAIKU))
    costs = [r.cost_usd for r in caplog.records if hasattr(r, "cost_usd")]  # type: ignore[attr-defined]
    assert costs == [pytest.approx(SONNET_COST), pytest.approx(HAIKU_COST)]


def test_run_context_span_and_run_cost_priced_by_request_model() -> None:
    t = factory.get_tracker(claude_settings())
    tracer, _ = new_tracer()
    ctx = RecordingContext(tracer, ScriptedProvider([resp(), resp()]), {}, tracker=t)
    ctx.call_model(req(SONNET))
    ctx.call_model(req(HAIKU))
    costs = [s["cost_usd"] for s in ctx.steps.values() if s["kind"] == "model"]
    assert costs == [pytest.approx(SONNET_COST), pytest.approx(HAIKU_COST)]
    # the runner stores run cost_usd as the sum of span costs
    assert sum(costs) == pytest.approx(SONNET_COST + HAIKU_COST)


# ---- combined spend and BudgetExceeded -----------------------------------------------


def test_agent_and_judge_calls_share_one_spent_usd_and_budget() -> None:
    t = factory.get_tracker(claude_settings())
    heavy = resp(1_000_000, 0)  # haiku $0.10, sonnet $2.00
    inner = ScriptedProvider([heavy, heavy, heavy])
    agent_side = BudgetGuardProvider(inner, 2.05, t)
    judge_side = BudgetGuardProvider(inner, 2.05, t)
    assert agent_side.tracker is judge_side.tracker

    agent_side.generate(req(HAIKU))
    assert t.spent_usd == pytest.approx(0.10)
    judge_side.generate(req(SONNET))
    assert t.spent_usd == pytest.approx(2.10)

    with pytest.raises(BudgetExceeded) as exc:
        agent_side.generate(req(HAIKU))
    assert exc.value.spent_usd == pytest.approx(2.10)
    assert len(inner.requests) == 2


def test_judge_spend_alone_is_priced_at_judge_rate_against_budget() -> None:
    # Priced at the haiku rate this call would be $0.10 and leave room; at the
    # sonnet rate it is $2.00 and the next call must be refused.
    t = factory.get_tracker(claude_settings())
    inner = ScriptedProvider([resp(1_000_000, 0)] * 3)
    g = BudgetGuardProvider(inner, 1.0, t)
    g.generate(req(SONNET))
    assert t.spent_usd == pytest.approx(2.00)
    with pytest.raises(BudgetExceeded):
        g.generate(req(HAIKU))


# ---- gemini / scripted values unchanged ----------------------------------------------


@pytest.mark.parametrize("provider", ["gemini", "scripted"])
def test_gemini_and_scripted_costs_unchanged(provider: str) -> None:
    s = Settings(provider=provider)
    assert s.model == s.judge_model == GEMINI
    t = factory.get_tracker(s)
    assert t.cost_of(resp()) == pytest.approx(GEMINI_COST)

    cache = RecCache()
    CacheFirstProvider(cache, ScriptedProvider([resp()]), tracker=t).generate(
        req(GEMINI, provider)
    )
    assert cache.rows[0]["cost_usd"] == pytest.approx(GEMINI_COST)

    g = BudgetGuardProvider(ScriptedProvider([resp(), resp()]), 10.0, t)
    g.generate(req(GEMINI, provider))
    g.generate(req(GEMINI, provider))  # judge call on the same model
    assert t.spent_usd == pytest.approx(2 * GEMINI_COST)

    tracer, _ = new_tracer()
    ctx = RecordingContext(tracer, ScriptedProvider([resp()]), {}, tracker=t)
    ctx.call_model(req(GEMINI, provider))
    (row,) = ctx.steps.values()
    assert row["cost_usd"] == pytest.approx(GEMINI_COST)
