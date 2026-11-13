"""`agent-replay record`, `replay` and `score` commands."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import psycopg

from agent_replay.cli.common import (
    CommandError,
    add_fault_flag,
    build_judge,
    check_fault_targets,
    connect,
    load_settings,
    uuid_arg,
)
from agent_replay.codes import JUDGE_STOP_CODES, STOP_CODES
from agent_replay.config import Settings
from agent_replay.db import runs_repo
from agent_replay.demo.agent import DemoAgent, available_variants
from agent_replay.llm import BudgetExceeded, QuotaExhausted
from agent_replay.llm.factory import build_live_provider
from agent_replay.runtime.faults import Fault
from agent_replay.runtime.runner import record_run, replay_run
from agent_replay.scoring.service import score_run
from agent_replay.taskspec import TaskSpec, get_task, load_tasks

Row = dict[str, Any]


def add_parsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    rec = subparsers.add_parser(
        "record", help="record one agent run with the live model"
    )
    rec.add_argument("--task", required=True, help="task id")
    rec.add_argument("--tasks", required=True, help="tasks JSONL file")
    rec.add_argument(
        "--variant",
        default="minimal",
        choices=available_variants(),
        help="prompt variant",
    )

    rep = subparsers.add_parser("replay", help="replay a recorded run")
    rep.add_argument("--run", required=True, type=uuid_arg, help="source run id")
    rep.add_argument("--variant", choices=available_variants(), help="prompt variant")
    rep.add_argument("--tasks", help="tasks JSONL file (default: rebuild from the run)")
    rep.add_argument(
        "--cache-only", action="store_true", help="never call a live model"
    )
    add_fault_flag(rep)

    sc = subparsers.add_parser("score", help="score a stored run")
    sc.add_argument("--run", required=True, type=uuid_arg, help="run id")
    sc.add_argument("--tasks", required=True, help="tasks JSONL file")
    sc.add_argument(
        "--no-judge",
        action="store_true",
        help="write only the deterministic rows (no LLM judge)",
    )


def _load_task(path: str, task_id: str) -> TaskSpec:
    try:
        return get_task(load_tasks(path), task_id)
    except (OSError, ValueError, KeyError) as exc:
        raise CommandError(str(exc).strip("'\"")) from None


def _rebuilt_task(
    conn: psycopg.Connection[Row], run: Row, run_id: str, s: Settings
) -> TaskSpec:
    """The source run's task: its stored snapshot, else rebuilt from its first request."""
    spans = runs_repo.list_spans(conn, run_id)
    for span in spans:
        snapshot = (
            (span["payload"] or {}).get("task") if span["kind"] == "agent" else None
        )
        if snapshot is not None:
            try:
                return TaskSpec.model_validate(snapshot)
            except ValueError:
                break
    print(
        "warning: run has no stored task snapshot; rebuilding the task from the "
        "recorded request (max_steps from settings)",
        file=sys.stderr,
    )
    for span in spans:
        if span["kind"] != "model":
            continue
        try:
            prompt = span["payload"]["request"]["messages"][0]["text"]
        except (KeyError, IndexError, TypeError):
            break
        return TaskSpec(
            id=run["task_id"],
            prompt=prompt,
            category="",
            expected_text=[],
            must_call_tools=[],
            max_steps=s.max_steps,
            rubric="",
        )
    raise CommandError(f"run {run_id} has no recorded model request to replay")


def _report(conn: psycopg.Connection[Row], run_id: Any) -> int:
    """Print the run id; exit 1 when the run was skipped or stopped by a stop code."""
    run = runs_repo.get_run(conn, str(run_id))
    if run is not None and run["status"] != "ok":
        print(
            f"warning: run status {run['status']}"
            + (f" ({run['error_type']})" if run["error_type"] else ""),
            file=sys.stderr,
        )
    print(run_id)
    if run is None:
        return 0
    stopped = run["status"] == "error" and run["error_type"] in STOP_CODES
    return 1 if run["status"] == "skipped" or stopped else 0


def _record(args: argparse.Namespace) -> int:
    settings = load_settings()
    task = _load_task(args.tasks, args.task)
    try:
        provider = build_live_provider(settings, "record")
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    assert provider is not None
    with connect(settings) as conn:
        try:
            run_id = record_run(
                task,
                DemoAgent(args.variant, settings),
                args.variant,
                conn=conn,
                provider=provider,
                settings=settings,
            )
        except (BudgetExceeded, QuotaExhausted) as exc:
            raise CommandError(
                f"run stopped: {type(exc).__name__} (the run was stored)"
            ) from None
        conn.commit()
        return _report(conn, run_id)


def _replay(args: argparse.Namespace) -> int:
    faults: list[Fault] = getattr(args, "fault", [])
    check_fault_targets(faults)
    settings = load_settings()
    with connect(settings) as conn:
        run = runs_repo.get_run(conn, args.run)
        if run is None:
            raise CommandError(f"run not found: {args.run}")
        task = (
            _load_task(args.tasks, run["task_id"])
            if args.tasks
            else _rebuilt_task(conn, run, args.run, settings)
        )
        variant = args.variant or run["variant"] or "minimal"
        try:
            provider = (
                None if args.cache_only else build_live_provider(settings, "replay")
            )
            agent = DemoAgent(variant, settings)
        except ValueError as exc:
            raise CommandError(str(exc)) from None
        try:
            run_id = replay_run(
                args.run,
                conn=conn,
                task=task,
                agent=agent,
                variant=variant,
                provider=provider,
                settings=settings,
                faults=faults,
            )
        except (BudgetExceeded, QuotaExhausted) as exc:
            raise CommandError(
                f"run stopped: {type(exc).__name__} (the run was stored)"
            ) from None
        conn.commit()
        return _report(conn, run_id)


def _score(args: argparse.Namespace) -> int:
    settings = load_settings()
    with connect(settings) as conn:
        run = runs_repo.get_run(conn, args.run)
        if run is None:
            raise CommandError(f"run not found: {args.run}")
        task = _load_task(args.tasks, run["task_id"])
        judge = None if args.no_judge else build_judge(conn, settings)
        summary = score_run(conn, args.run, task, judge=judge)
        conn.commit()
    print(json.dumps(summary, indent=2))
    stop = ((summary.get("judge") or {}).get("detail") or {}).get("verdict")
    if stop in JUDGE_STOP_CODES:
        print(f"error: judge stopped: {stop}", file=sys.stderr)
        return 1
    return 0


def run(args: argparse.Namespace) -> int:
    handlers = {"record": _record, "replay": _replay, "score": _score}
    try:
        return handlers[args.command](args)
    except CommandError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except psycopg.Error as exc:
        print(f"error: database error ({type(exc).__name__})", file=sys.stderr)
        return 1
