from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from agent_replay.db import runs_repo
from agent_replay.scoring.checks import CheckResult
from agent_replay.scoring.service import score_run
from agent_replay.taskspec import TaskSpec

TASK = TaskSpec(
    id="t",
    prompt="p",
    category="c",
    expected_text=["a"],
    must_call_tools=[],
    max_steps=5,
    rubric="r",
)


class FakeConn:
    def __init__(self) -> None:
        self.events: list[str] = []

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.events.append("begin")
        yield
        self.events.append("commit")


@pytest.fixture
def conn(monkeypatch: pytest.MonkeyPatch) -> FakeConn:
    fake = FakeConn()
    run = {"final_output": "a", "status": "ok", "model_calls": 1}
    monkeypatch.setattr(runs_repo, "get_run", lambda c, r: run)
    monkeypatch.setattr(runs_repo, "list_spans", lambda c, r: [])

    def upsert(c: Any, run_id: str, name: str, passed: Any, *a: Any, **k: Any) -> None:
        fake.events.append(f"upsert:{name}:{passed}")

    monkeypatch.setattr(runs_repo, "upsert_score", upsert)
    monkeypatch.setattr(
        runs_repo, "delete_judge_score", lambda c, r: fake.events.append("delete")
    )
    return fake


def test_judge_with_none_passed_is_stored_as_null(conn: FakeConn) -> None:
    summary = score_run(
        conn,  # type: ignore[arg-type]
        "r",
        TASK,
        judge=lambda t, r, s: CheckResult(None, {"verdict": "judge_error"}),
    )
    assert "upsert:judge:None" in conn.events
    assert "delete" not in conn.events
    assert summary["judge"]["passed"] is None
    assert summary["deterministic_pass"] is True


def test_rows_written_in_one_transaction(conn: FakeConn) -> None:
    score_run(conn, "r", TASK)  # type: ignore[arg-type]
    assert conn.events[0] == "begin" and conn.events[-1] == "commit"
    assert conn.events.count("begin") == 1


def test_no_judge_deletes_stale_judge_row(conn: FakeConn) -> None:
    score_run(conn, "r", TASK)  # type: ignore[arg-type]
    assert "delete" in conn.events
