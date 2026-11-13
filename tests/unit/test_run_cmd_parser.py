"""Parser behaviour of the record/replay/score commands (no database)."""

from __future__ import annotations

import pytest

from agent_replay.cli import build_parser


def test_replay_rejects_invalid_uuid() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["replay", "--run", "not-a-uuid"])
    assert exc.value.code == 2


def test_replay_normalises_uuid_and_cache_only() -> None:
    rid = "12345678-1234-5678-1234-567812345678"
    args = build_parser().parse_args(["replay", "--run", rid.upper(), "--cache-only"])
    assert args.run == rid and args.cache_only is True


def test_record_requires_task_and_tasks() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["record", "--task", "x"])


def test_replay_collects_repeated_fault_flags() -> None:
    rid = "12345678-1234-5678-1234-567812345678"
    args = build_parser().parse_args(
        [
            "replay",
            "--run",
            rid,
            "--fault",
            "tool_timeout:get_order",
            "--fault",
            "rate_limit:1",
        ]
    )
    assert [f.spec for f in args.fault] == ["tool_timeout:get_order", "rate_limit:1"]


def test_replay_invalid_fault_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rid = "12345678-1234-5678-1234-567812345678"
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["replay", "--run", rid, "--fault", "explode:x"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    for kind in ("tool_timeout", "malformed_tool_output", "rate_limit"):
        assert kind in err


def test_replay_rejects_fault_on_unknown_tool(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_replay.cli import run_cmd

    rid = "12345678-1234-5678-1234-567812345678"
    args = build_parser().parse_args(
        ["replay", "--run", rid, "--fault", "tool_timeout:get_ordr"]
    )
    assert run_cmd.run(args) == 1  # rejected before any settings or database access
    err = capsys.readouterr().err
    assert "get_ordr" in err and "get_order" in err
