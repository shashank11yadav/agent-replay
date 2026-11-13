"""`agent-replay suite record|replay|score`: run a whole task file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
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
from agent_replay.db import runs_repo
from agent_replay.demo.agent import DemoAgent, available_variants
from agent_replay.llm.factory import build_live_provider
from agent_replay.runtime.faults import Fault
from agent_replay.suite.runner import (
    SuiteOutcome,
    file_sha256,
    plan_replay,
    record_suite,
    replay_suite,
    score_suite,
    select_subset,
)
from agent_replay.taskspec import TaskSpec, load_tasks

Row = dict[str, Any]


def add_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    suite = subparsers.add_parser("suite", help="record, replay or score a whole suite")
    sub = suite.add_subparsers(dest="suite_command", metavar="<suite command>")
    sub.required = True

    rec = sub.add_parser("record", help="record every task (resumable)")
    rec.add_argument(
        "--tasks",
        help="tasks JSONL file (new suite; with --resume, overrides the stored path)",
    )
    rec.add_argument("--variant", choices=available_variants(), help="prompt variant")
    rec.add_argument("--label", help="suite label (new suite)")
    rec.add_argument(
        "--resume",
        type=uuid_arg,
        metavar="SUITE_RUN_ID",
        help="continue a stopped suite (tasks file and variant come from the suite)",
    )

    rep = sub.add_parser("replay", help="replay every run of a recorded suite")
    rep.add_argument(
        "--source", required=True, type=uuid_arg, help="source suite run id"
    )
    rep.add_argument("--variant", choices=available_variants(), help="prompt variant")
    rep.add_argument("--label", default="", help="suite label")
    rep.add_argument(
        "--tasks", help="tasks JSONL file (default: the source suite's stored path)"
    )
    rep.add_argument("--tasks-subset", help="comma-separated task ids to replay")
    add_fault_flag(rep)

    sc = sub.add_parser("score", help="score every run of a suite")
    sc.add_argument("--suite-run", required=True, type=uuid_arg, help="suite run id")
    sc.add_argument("--tasks", help="tasks JSONL file (default: stored path)")
    sc.add_argument("--no-judge", action="store_true", help="deterministic checks only")


def _load(path: str) -> list[TaskSpec]:
    try:
        return load_tasks(path)
    except (OSError, ValueError) as exc:
        raise CommandError(str(exc)) from None


def _suite(conn: psycopg.Connection[Row], suite_run_id: str) -> Row:
    suite = runs_repo.get_suite_run(conn, suite_run_id)
    if suite is None:
        raise CommandError(f"suite run not found: {suite_run_id}")
    return suite


def _sha(path: str) -> str:
    try:
        return file_sha256(path)
    except OSError as exc:
        raise CommandError(
            f"cannot read tasks file {path}: {exc.strerror or type(exc).__name__}"
        ) from None


def _stored_tasks_path(suite: Row) -> str:
    """The suite's stored tasks path, if it is safe to read without asking.

    The value may come from an imported fixture, so only a relative path that stays under
    the current directory is used; anything else needs an explicit --tasks.
    """
    path = suite["tasks_path"]
    if not path:
        raise CommandError(f"suite run {suite['id']} has no tasks file recorded")
    cwd = Path.cwd().resolve()
    if Path(path).is_absolute() or not Path(path).resolve().is_relative_to(cwd):
        raise CommandError(
            f"suite run {suite['id']} stores a tasks file path that is absolute or "
            "outside the current directory; pass --tasks"
        )
    return str(path)


def _suite_tasks(suite: Row, override: str | None = None) -> list[TaskSpec]:
    """The suite's tasks: ``override`` if given, else the stored path."""
    path = override or _stored_tasks_path(suite)
    if suite["tasks_sha256"] and _sha(path) != suite["tasks_sha256"]:
        print(
            f"warning: {path} changed since the suite was created (sha256 differs)",
            file=sys.stderr,
        )
    return _load(path)


def _report_stop(outcome: SuiteOutcome, resume_hint: bool) -> None:
    print(
        f"error: suite stopped: {outcome.stop_code}; {outcome.remaining} tasks remain",
        file=sys.stderr,
    )
    if resume_hint:
        print(
            f"resume with: agent-replay suite record --resume {outcome.suite_run_id}",
            file=sys.stderr,
        )


def _agent(variant: str, settings: Any) -> DemoAgent:
    return DemoAgent(variant, settings)


def _check_variant(variant: str, settings: Any) -> None:
    """Fail before the suite row is written when the variant's prompt file is gone."""
    try:
        _agent(variant, settings)
    except ValueError as exc:
        raise CommandError(str(exc)) from None


def _record(args: argparse.Namespace) -> int:
    if args.resume and (args.variant is not None or args.label is not None):
        raise CommandError("--resume takes no --variant or --label")
    if not args.resume and (args.tasks is None or args.variant is None):
        raise CommandError("suite record needs --tasks and --variant (or --resume ID)")
    settings = load_settings()
    try:
        provider = build_live_provider(settings, "record")
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    assert provider is not None
    with connect(settings) as conn:
        if args.resume:
            suite = _suite(conn, args.resume)
            if suite["mode"] != "record":
                raise CommandError("only a record suite can be resumed")
            tasks = _suite_tasks(suite, args.tasks)
            suite_id, variant = str(suite["id"]), suite["variant"]
            _check_variant(variant, settings)
            runs_repo.set_suite_status(conn, suite_id, "running")
        else:
            tasks = _load(args.tasks)
            variant = args.variant
            _check_variant(variant, settings)
            path = args.tasks
            suite_id = str(
                runs_repo.create_suite_run(
                    conn,
                    label=args.label or "",
                    variant=variant,
                    mode="record",
                    tasks_path=path,
                    tasks_sha256=_sha(path),
                    task_count=len(tasks),
                )
            )
        conn.commit()
        outcome = record_suite(
            conn,
            suite_id,
            tasks,
            variant=variant,
            provider=provider,
            settings=settings,
            make_agent=_agent,
        )
    if outcome.stopped:
        _report_stop(outcome, resume_hint=True)
        print(suite_id)
        return 1
    print(suite_id)
    return 0


def _replay(args: argparse.Namespace) -> int:
    faults: list[Fault] = args.fault
    check_fault_targets(faults)
    settings = load_settings()
    with connect(settings) as conn:
        source = _suite(conn, args.source)
        if source["status"] in ("stopped", "running"):
            print(
                f"warning: source suite {args.source} is {source['status']}; "
                "it may be incomplete",
                file=sys.stderr,
            )
        tasks = _suite_tasks(source, args.tasks)
        if args.tasks_subset:
            ids = [i.strip() for i in args.tasks_subset.split(",") if i.strip()]
            try:
                tasks = select_subset(tasks, ids)
            except ValueError as exc:
                raise CommandError(str(exc)) from None
        source_runs = runs_repo.list_suite_run_rows(conn, args.source)
        plan = plan_replay(tasks, source_runs)
        planned = {t.id for t, _ in plan}
        cut_short = [
            t.id
            for t in tasks
            if t.id not in planned and any(r["task_id"] == t.id for r in source_runs)
        ]
        if cut_short:
            print(
                "warning: not replayed (only stopped runs): " + ", ".join(cut_short),
                file=sys.stderr,
            )
        if not plan:
            raise CommandError("the source suite has no runs to replay")
        variant = args.variant or source["variant"] or "minimal"
        _check_variant(variant, settings)
        try:
            provider = build_live_provider(settings, "replay")
        except ValueError as exc:
            raise CommandError(str(exc)) from None
        tasks_path = args.tasks or source["tasks_path"]
        tasks_sha256 = _sha(args.tasks) if args.tasks else source["tasks_sha256"]
        suite_id = str(
            runs_repo.create_suite_run(
                conn,
                label=args.label,
                variant=variant,
                mode="replay",
                source_suite_run_id=args.source,
                faults=[f.spec for f in faults],
                tasks_path=tasks_path,
                tasks_sha256=tasks_sha256,
                task_count=len(plan),
            )
        )
        conn.commit()
        outcome = replay_suite(
            conn,
            suite_id,
            plan,
            variant=variant,
            provider=provider,
            settings=settings,
            faults=faults,
            make_agent=_agent,
        )
    if outcome.stopped:
        _report_stop(outcome, resume_hint=False)
        print(suite_id)
        return 1
    print(suite_id)
    return 0


def _score(args: argparse.Namespace) -> int:
    settings = load_settings()
    with connect(settings) as conn:
        suite = _suite(conn, args.suite_run)
        tasks = _suite_tasks(suite, args.tasks)
        judge = None if args.no_judge else build_judge(conn, settings)
        try:
            outcome = score_suite(conn, args.suite_run, tasks, judge)
        except KeyError as exc:
            raise CommandError(str(exc).strip("'\"")) from None
    print(f"scored {outcome.scored} runs")
    if outcome.judge_stops:
        print(
            f"error: judge stopped on {len(outcome.judge_stops)} runs: "
            f"{outcome.judge_stops[0]}",
            file=sys.stderr,
        )
        return 1
    return 0


def run(args: argparse.Namespace) -> int:
    handlers = {"record": _record, "replay": _replay, "score": _score}
    try:
        return handlers[args.suite_command](args)
    except CommandError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except psycopg.Error as exc:
        print(f"error: database error ({type(exc).__name__})", file=sys.stderr)
        return 1
