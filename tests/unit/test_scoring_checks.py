from agent_replay.scoring.checks import completed, expected_text
from agent_replay.taskspec import TaskSpec

TASK = TaskSpec(
    id="t",
    prompt="p",
    category="c",
    expected_text=["A", "b"],
    must_call_tools=[],
    max_steps=1,
    rubric="r",
)


def test_expected_text_reports_missing_when_no_output() -> None:
    res = expected_text(TASK, {"final_output": None}, [])
    assert not res.passed
    assert res.detail["missing"] == ["A", "b"]


def test_completed_requires_ok() -> None:
    assert completed(TASK, {"status": "ok"}, []).passed
    assert not completed(TASK, {"status": "error"}, []).passed
