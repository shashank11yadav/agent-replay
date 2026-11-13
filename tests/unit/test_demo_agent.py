"""Demo agent, its synthetic tools and data, prompts and the example task."""

from __future__ import annotations

import importlib
import json
import re
from typing import Any

import pytest
from support import ROOT

from agent_replay.adapters.base import ToolResult
from agent_replay.llm import ModelRequest, ModelResponse, ScriptedProvider, ToolCall
from agent_replay.taskspec import TaskSpec, load_tasks

DEMO = ROOT / "src" / "agent_replay" / "demo"


def _tools() -> Any:
    return importlib.import_module("agent_replay.demo.tools")


def _agent_mod() -> Any:
    return importlib.import_module("agent_replay.demo.agent")


def _records(name: str) -> list[dict[str, Any]]:
    raw = json.loads((DEMO / "data" / name).read_text())
    if (
        isinstance(raw, dict)
        and len(raw) == 1
        and isinstance(next(iter(raw.values())), (list, dict))
    ):
        raw = next(iter(raw.values()))
    if isinstance(raw, dict):
        return [{"_key": k, **v} if isinstance(v, dict) else v for k, v in raw.items()]
    return list(raw)


def _idof(rec: dict[str, Any]) -> str:
    for k in ("order_id", "id", "product_id", "_key"):
        if k in rec:
            return str(rec[k])
    raise AssertionError(f"no id key in {rec}")


# --- tools ----------------------------------------------------------------


def test_get_order_returns_order_and_unknown_is_not_found() -> None:
    t = _tools()
    recs = {_idof(r): r for r in _records("orders.json")}
    assert "A1001" in recs
    got = t.get_order("A1001")
    expected = {k: v for k, v in recs["A1001"].items() if k != "_key"}
    assert {k: v for k, v in got.items() if k != "_key"} == expected
    assert t.get_order("ZZZ-nope") == {"error": "not_found"}


def test_search_products_case_insensitive_sorted_by_id() -> None:
    t = _tools()
    prods = _records("products.json")
    matches = sorted((p for p in prods if "lamp" in str(p["name"]).lower()), key=_idof)
    assert len(matches) >= 2, "data must contain at least two lamp products"
    for q in ("lamp", "LAMP", "Lamp"):
        res = t.search_products(q)
        items = res["products"] if isinstance(res, dict) else res
        assert [_idof(p) for p in items] == [_idof(p) for p in matches]
    res = t.search_products("lamp")
    items = res["products"] if isinstance(res, dict) else res
    ids = [_idof(p) for p in items]
    assert ids == sorted(ids)


def _rate(rates: Any, a: str, b: str) -> float:
    if isinstance(rates, dict) and "rates" in rates:
        rates = rates["rates"]
    if isinstance(rates, dict):
        if a in rates and isinstance(rates[a], dict):
            return float(rates[a][b])
        for key in (f"{a}_{b}", f"{a}->{b}", f"{a}{b}", f"{a}/{b}"):
            if key in rates:
                return float(rates[key])
        return float(rates[b]) / float(rates[a])
    raise AssertionError("unsupported rates shape")


def test_convert_currency_uses_fixed_rate_and_rounds() -> None:
    t = _tools()
    rates = json.loads((DEMO / "data" / "rates.json").read_text())
    res = t.convert_currency(100, "USD", "EUR")
    assert isinstance(res, dict)
    nums = [v for v in res.values() if isinstance(v, float)]
    assert round(_rate(rates, "USD", "EUR") * 100, 2) in nums
    for v in nums:
        assert round(v, 2) == v
    odd = t.convert_currency(33.333, "USD", "EUR")
    assert all(round(v, 2) == v for v in odd.values() if isinstance(v, float))


def test_convert_currency_unknown_currency() -> None:
    t = _tools()
    assert t.convert_currency(100, "USD", "XXX") == {"error": "unknown_currency"}
    assert t.convert_currency(100, "XXX", "EUR") == {"error": "unknown_currency"}


def test_each_tool_has_schema_and_output_model() -> None:
    from pydantic import BaseModel

    t = _tools()
    from agent_replay.llm import ToolSchema

    names = {"get_order", "search_products", "convert_currency"}
    assert set(t.TOOL_SCHEMAS) == names
    assert set(t.OUTPUT_MODELS) == names
    for n in names:
        s = t.TOOL_SCHEMAS[n]
        assert isinstance(s, ToolSchema) and s.name == n and s.description
        assert s.parameters.get("type") == "object"
        assert s.parameters.get("properties")
        assert isinstance(t.OUTPUT_MODELS[n], type)
        assert issubclass(t.OUTPUT_MODELS[n], BaseModel)
    # real tool output validates against its model
    t.OUTPUT_MODELS["get_order"].model_validate(t.get_order("A1001"))


# --- agent basics ---------------------------------------------------------


def test_agent_name_and_load_prompt() -> None:
    a = _agent_mod()
    assert a.DemoAgent().name == "demo-agent"
    text = (DEMO / "prompts" / "structured.txt").read_text()
    assert text.strip()
    assert a.load_prompt("structured") == text
    assert (DEMO / "prompts" / "minimal.txt").read_text().strip()
    assert a.load_prompt("minimal") == (DEMO / "prompts" / "minimal.txt").read_text()


def test_load_prompt_unknown_variant_lists_available() -> None:
    a = _agent_mod()
    with pytest.raises(Exception) as ei:
        a.load_prompt("nonexistent_variant")
    msg = str(ei.value)
    assert "minimal" in msg and "structured" in msg


# --- agent loop with a fake context ----------------------------------------


class FakeCtx:
    def __init__(
        self,
        responses: list[ModelResponse | Exception],
        tool_result: ToolResult | None = None,
    ) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.tool_result = tool_result or ToolResult(
            ok=True, data={"order_id": "A1001"}, raw=None
        )

    def call_model(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request.model_copy(deep=True))
        r = self.responses[len(self.requests) - 1]
        if isinstance(r, Exception):
            raise r
        return r

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        self.tool_calls.append((name, dict(args)))
        return self.tool_result


def _task(max_steps: int = 5, **kw: Any) -> TaskSpec:
    d: dict[str, Any] = {
        "id": "t1",
        "prompt": "Where is order A1001?",
        "category": "lookup",
        "expected_text": ["x"],
        "must_call_tools": ["get_order"],
        "max_steps": max_steps,
        "rubric": "r",
    }
    d.update(kw)
    return TaskSpec(**d)


def _call(
    name: str = "get_order", args: dict[str, Any] | None = None, i: str = "x"
) -> ToolCall:
    return ToolCall(
        id=i, name=name, args=args if args is not None else {"order_id": "A1001"}
    )


def _agent(**settings: Any) -> Any:
    a = _agent_mod()
    if settings:
        from agent_replay.config import Settings

        return a.DemoAgent(settings=Settings.model_validate(settings))
    return a.DemoAgent()


def _tool_msgs(req: ModelRequest) -> list[Any]:
    return [m for m in req.messages if m.role == "tool"]


def test_run_one_tool_call_then_text_answer() -> None:
    ctx = FakeCtx(
        [
            ModelResponse(tool_calls=[_call()]),
            ModelResponse(text="It shipped."),
        ]
    )
    out = _agent().run(_task(), ctx)
    assert ctx.tool_calls == [("get_order", {"order_id": "A1001"})]
    assert out.status == "ok"
    assert out.final_output == "It shipped."
    assert len(ctx.requests) == 2
    first = ctx.requests[0]
    assert {t.name for t in first.tools} == {
        "get_order",
        "search_products",
        "convert_currency",
    }
    assert first.messages[0].role == "user"
    assert first.messages[0].text == "Where is order A1001?"
    assert first.system.strip()


def test_tool_call_ids_are_deterministic() -> None:
    def go() -> list[ModelRequest]:
        ctx = FakeCtx(
            [
                ModelResponse(
                    tool_calls=[
                        _call(i="a"),
                        _call("search_products", {"query": "lamp"}, "b"),
                    ]
                ),
                ModelResponse(text="done"),
            ]
        )
        _agent().run(_task(), ctx)
        return ctx.requests

    r1, r2 = go(), go()
    assert r1[1].model_dump() == r2[1].model_dump()
    req = r1[1]
    model_msg = [m for m in req.messages if m.role == "model"][-1]
    tool_msg = _tool_msgs(req)[-1]
    ids = [c.id for c in model_msg.tool_calls]
    assert re.fullmatch(r"call_[01]_0", ids[0])
    step = ids[0].split("_")[1]
    assert ids == [f"call_{step}_0", f"call_{step}_1"]
    assert [r.call_id for r in tool_msg.tool_results] == ids
    assert [r.name for r in tool_msg.tool_results] == ["get_order", "search_products"]


def test_malformed_tool_output_sent_as_error_and_no_raise() -> None:
    ctx = FakeCtx(
        [ModelResponse(tool_calls=[_call()]), ModelResponse(text="sorry")],
        ToolResult(ok=True, data={"unexpected_field": True}, raw=None),
    )
    out = _agent().run(_task(), ctx)
    assert out.status == "ok"
    tr = _tool_msgs(ctx.requests[1])[-1].tool_results[0]
    assert tr.response == {"error": "malformed_tool_output"}


def test_valid_tool_output_passed_through() -> None:
    from agent_replay.demo import tools as t

    data = t.get_order("A1001")
    ctx = FakeCtx(
        [ModelResponse(tool_calls=[_call()]), ModelResponse(text="ok")],
        ToolResult(ok=True, data=data, raw=json.dumps(data)),
    )
    _agent().run(_task(), ctx)
    tr = _tool_msgs(ctx.requests[1])[-1].tool_results[0]
    assert tr.response == data


def test_tool_error_result_passed_to_model() -> None:
    ctx = FakeCtx(
        [ModelResponse(tool_calls=[_call()]), ModelResponse(text="unavailable")],
        ToolResult(
            ok=False, data={"error": "tool_timeout"}, raw=None, error_type="timeout"
        ),
    )
    out = _agent().run(_task(), ctx)
    assert out.status == "ok" and out.final_output == "unavailable"
    tr = _tool_msgs(ctx.requests[1])[-1].tool_results[0]
    assert tr.response == {"error": "tool_timeout"}


def test_max_steps_from_task_when_lower() -> None:
    ctx = FakeCtx([ModelResponse(tool_calls=[_call()]) for _ in range(20)])
    out = _agent(max_steps=10).run(_task(max_steps=2), ctx)
    assert len(ctx.requests) == 2
    assert out.status == "error"
    assert out.error == "max_steps_exceeded"


def test_max_steps_from_settings_when_lower() -> None:
    ctx = FakeCtx([ModelResponse(tool_calls=[_call()]) for _ in range(20)])
    out = _agent(max_steps=3).run(_task(max_steps=9), ctx)
    assert len(ctx.requests) == 3
    assert out.status == "error"
    assert out.error == "max_steps_exceeded"


def test_call_model_exception_becomes_error_outcome() -> None:
    class Boom(Exception):
        pass

    ctx = FakeCtx([Boom("secret detail")])
    out = _agent().run(_task(), ctx)
    assert out.status == "error"
    assert out.error is not None and "Boom" in out.error


# --- example task ----------------------------------------------------------


def test_example_task_passes_deterministic_checks_with_scripted_provider() -> None:
    from agent_replay.demo import tools as t

    sk = DEMO / "example"
    jsonls = list(sk.glob("*.jsonl"))
    jsons = list(sk.glob("*.json"))
    assert len(jsonls) == 1 and len(jsons) == 1
    tasks = load_tasks(jsonls[0])
    assert len(tasks) == 1
    task = tasks[0]
    provider = ScriptedProvider.from_file(jsons[0])

    class Ctx:
        def __init__(self) -> None:
            self.called: list[str] = []
            self.models = 0

        def call_model(self, request: ModelRequest) -> ModelResponse:
            self.models += 1
            return provider.generate(request)

        def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
            self.called.append(name)
            fn = getattr(t, name)
            data = fn(**args)
            return ToolResult(ok="error" not in data, data=data, raw=json.dumps(data))

    ctx = Ctx()
    out = _agent().run(task, ctx)
    assert out.status == "ok", out.error
    assert out.final_output
    low = out.final_output.lower()
    assert all(s.lower() in low for s in task.expected_text)
    assert set(task.must_call_tools) <= set(ctx.called)
    assert ctx.models <= task.max_steps
