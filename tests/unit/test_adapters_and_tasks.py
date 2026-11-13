"""Adapter interface types and the task spec loader."""

from __future__ import annotations

import importlib
import json
import typing
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agent_replay.taskspec import load_tasks


def _base() -> Any:
    return importlib.import_module("agent_replay.adapters.base")


def _ts() -> Any:
    return importlib.import_module("agent_replay.taskspec")


def _row(i: str = "t1", **kw: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": i,
        "prompt": f"prompt {i}",
        "category": "lookup",
        "expected_text": ["a", "b"],
        "must_call_tools": ["get_order"],
        "max_steps": 5,
        "rubric": "be right",
    }
    d.update(kw)
    return d


def _write(p: Path, rows: list[Any]) -> Path:
    p.write_text(
        "\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n"
    )
    return p


# --- RunContext / AgentAdapter protocols ---------------------------------


def test_run_context_is_protocol_with_call_model_and_call_tool() -> None:
    b = _base()
    assert getattr(b.RunContext, "_is_protocol", False)
    assert callable(b.RunContext.call_model)
    assert callable(b.RunContext.call_tool)
    import inspect

    assert list(inspect.signature(b.RunContext.call_model).parameters) == [
        "self",
        "request",
    ]
    assert list(inspect.signature(b.RunContext.call_tool).parameters) == [
        "self",
        "name",
        "args",
    ]


def test_agent_adapter_is_protocol_with_name_and_run() -> None:
    import inspect

    b = _base()
    assert getattr(b.AgentAdapter, "_is_protocol", False)
    assert "name" in typing.get_type_hints(b.AgentAdapter) or hasattr(
        b.AgentAdapter, "name"
    )
    assert list(inspect.signature(b.AgentAdapter.run).parameters) == [
        "self",
        "task",
        "ctx",
    ]


def test_structural_adapter_works_without_inheritance() -> None:
    b = _base()
    ts = _ts()

    class Dummy:
        name = "dummy"

        def run(self, task: Any, ctx: Any) -> Any:
            return b.AgentOutcome(status="ok", final_output=task.prompt)

    a: Any = Dummy()
    task = ts.TaskSpec(**_row())
    out = a.run(task, None)
    assert out.final_output == "prompt t1"
    assert out.status == "ok"


# --- AgentOutcome ---------------------------------------------------------


def test_agent_outcome_fields_and_optional_defaults() -> None:
    b = _base()
    o = b.AgentOutcome(status="ok")
    assert o.final_output is None
    assert o.error is None
    assert o.status == "ok"
    e = b.AgentOutcome(status="error", error="boom", final_output=None)
    assert e.error == "boom"
    f = b.AgentOutcome(status="ok", final_output="answer")
    assert f.final_output == "answer"


def test_agent_outcome_rejects_unknown_status() -> None:
    b = _base()
    with pytest.raises(ValidationError):
        b.AgentOutcome(status="weird")


# --- ToolResult -----------------------------------------------------------


def test_tool_result_fields() -> None:
    b = _base()
    r = b.ToolResult(ok=True, data={"x": 1}, raw=None)
    assert r.ok is True
    assert r.data == {"x": 1}
    assert r.raw is None
    assert r.error_type is None
    bad = b.ToolResult(ok=False, data=None, raw="garbled{", error_type="malformed")
    assert bad.ok is False
    assert bad.data is None
    assert bad.raw == "garbled{"
    assert bad.error_type == "malformed"


def test_tool_result_requires_ok() -> None:
    b = _base()
    with pytest.raises(ValidationError):
        b.ToolResult(data={}, raw=None)


def test_adapters_package_reexports() -> None:
    pkg = importlib.import_module("agent_replay.adapters")
    b = _base()
    for n in ("RunContext", "AgentAdapter", "AgentOutcome", "ToolResult"):
        assert getattr(pkg, n) is getattr(b, n)


# --- TaskSpec -------------------------------------------------------------


def test_task_spec_fields() -> None:
    t = _ts().TaskSpec(**_row("x9"))
    assert t.id == "x9"
    assert t.prompt == "prompt x9"
    assert t.category == "lookup"
    assert t.expected_text == ["a", "b"]
    assert t.must_call_tools == ["get_order"]
    assert t.max_steps == 5
    assert t.rubric == "be right"


def test_task_spec_rejects_bad_types() -> None:
    ts = _ts()
    with pytest.raises(ValidationError):
        ts.TaskSpec(**_row(max_steps="many"))
    with pytest.raises(ValidationError):
        ts.TaskSpec(**_row(expected_text="notalist"))


# --- load_tasks -----------------------------------------------------------


def test_load_tasks_preserves_file_order(tmp_path: Path) -> None:
    ts = _ts()
    p = _write(tmp_path / "tasks.jsonl", [_row("z"), _row("a"), _row("m")])
    tasks = ts.load_tasks(p)
    assert [t.id for t in tasks] == ["z", "a", "m"]
    assert all(isinstance(t, ts.TaskSpec) for t in tasks)
    assert tasks[1].max_steps == 5


def test_load_tasks_accepts_str_path(tmp_path: Path) -> None:
    ts = _ts()
    p = _write(tmp_path / "tasks.jsonl", [_row("a")])
    assert [t.id for t in ts.load_tasks(str(p))] == ["a"]


@pytest.mark.parametrize(
    "bad",
    [
        "{not json",
        json.dumps({"id": "b"}),  # missing required fields
        json.dumps(_row("b", max_steps="x")),  # wrong type
        json.dumps(_row("b")).replace('"max_steps": 5', '"max_steps": NaN'),
    ],
)
def test_load_tasks_invalid_line_names_file_and_line(tmp_path: Path, bad: str) -> None:
    ts = _ts()
    p = _write(tmp_path / "mytasks.jsonl", [_row("a"), _row("c"), bad, _row("d")])
    with pytest.raises(Exception) as ei:
        ts.load_tasks(p)
    msg = str(ei.value)
    assert "mytasks.jsonl" in msg
    assert "3" in msg
    assert "line" in msg.lower() or "mytasks.jsonl:3" in msg


def test_load_tasks_invalid_first_line_reports_line_1(tmp_path: Path) -> None:
    ts = _ts()
    p = _write(tmp_path / "f.jsonl", ["[]", _row("a")])
    with pytest.raises(Exception) as ei:
        ts.load_tasks(p)
    assert "f.jsonl" in str(ei.value)
    assert "1" in str(ei.value)


def test_load_tasks_duplicate_id_names_id(tmp_path: Path) -> None:
    ts = _ts()
    p = _write(tmp_path / "t.jsonl", [_row("a"), _row("dupe-id"), _row("dupe-id")])
    with pytest.raises(Exception) as ei:
        ts.load_tasks(p)
    assert "dupe-id" in str(ei.value)


# --- get_task -------------------------------------------------------------


def test_get_task_returns_match(tmp_path: Path) -> None:
    ts = _ts()
    tasks = ts.load_tasks(_write(tmp_path / "t.jsonl", [_row("a"), _row("b")]))
    assert ts.get_task(tasks, "b").id == "b"


def test_get_task_missing_raises_keyerror_naming_id(tmp_path: Path) -> None:
    ts = _ts()
    tasks = ts.load_tasks(_write(tmp_path / "t.jsonl", [_row("a")]))
    with pytest.raises(KeyError) as ei:
        ts.get_task(tasks, "nope-77")
    assert "nope-77" in str(ei.value)


def test_blank_lines_skipped_and_line_numbers_kept(tmp_path: Path) -> None:
    row = (
        '{"id":"a","prompt":"p","category":"c","expected_text":[],'
        '"must_call_tools":[],"max_steps":1,"rubric":"r"}'
    )
    p = tmp_path / "t.jsonl"
    p.write_text(f"{row}\n\nnot json\n")
    with pytest.raises(ValueError, match=r"t\.jsonl:3"):
        load_tasks(p)


def test_load_tasks_errors_do_not_echo_the_file_contents(tmp_path: Path) -> None:
    ts = _ts()
    secret = json.dumps({"id": "a", "prompt": "ssh-rsa AAAASECRETKEY"})
    p = _write(tmp_path / "t.jsonl", [secret])
    with pytest.raises(ValueError) as ei:
        ts.load_tasks(p)
    assert "SECRETKEY" not in str(ei.value)
    assert "t.jsonl:1" in str(ei.value)
