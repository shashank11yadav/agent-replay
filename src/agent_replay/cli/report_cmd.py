"""`agent-replay gate|gate-check|compare`: regression gate and reports."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg

from agent_replay import jsonio
from agent_replay.cli.common import CommandError, connect, load_settings, uuid_arg
from agent_replay.report.compare import compare
from agent_replay.report.gate import (
    SuiteNotFoundError,
    Verdict,
    gate,
    load_thresholds,
)
from agent_replay.report.markdown import render_report

COMMANDS = ("gate", "gate-check", "compare")


def add_parsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    g = subparsers.add_parser("gate", help="compare a candidate suite with a baseline")
    g.add_argument(
        "--baseline", required=True, type=uuid_arg, help="baseline suite run id"
    )
    g.add_argument(
        "--candidate", required=True, type=uuid_arg, help="candidate suite run id"
    )
    g.add_argument("--thresholds", required=True, help="thresholds JSON file")
    g.add_argument(
        "--out-dir", required=True, help="directory for report.md/verdict.json"
    )

    c = subparsers.add_parser(
        "gate-check", help="exit 1 when a verdict is FAIL or ERROR"
    )
    c.add_argument("verdict_file", help="verdict.json written by `gate`")

    cmp_ = subparsers.add_parser("compare", help="compare several suite runs")
    cmp_.add_argument(
        "--suite-run",
        action="append",
        default=[],
        required=True,
        type=uuid_arg,
        metavar="ID",
        help="suite run id (repeat for each suite to compare)",
    )
    cmp_.add_argument("--out", required=True, help="Markdown file to write")


def _gate(args: argparse.Namespace) -> int:
    baseline, candidate = args.baseline, args.candidate
    try:
        thresholds = load_thresholds(args.thresholds)
    except OSError as exc:
        raise CommandError(
            f"cannot read thresholds {args.thresholds}: {exc.strerror}"
        ) from None
    except ValueError as exc:
        raise CommandError(
            f"invalid thresholds file {args.thresholds}: {exc}"
        ) from None
    with connect(load_settings()) as conn:
        try:
            result = gate(conn, baseline, candidate, thresholds)
        except SuiteNotFoundError as exc:
            raise CommandError(str(exc)) from None
    report = render_report(result)
    out_dir = Path(args.out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.md").write_text(report)
        (out_dir / "verdict.json").write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
    except OSError as exc:
        raise CommandError(f"cannot write report: {exc.strerror}") from None
    print(f"verdict: {result.verdict.value}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(report + "\n")
        except OSError as exc:
            print(
                f"warning: cannot append to step summary: {exc.strerror}",
                file=sys.stderr,
            )
    return 0


def _gate_check(args: argparse.Namespace) -> int:
    try:
        data = jsonio.read_json(args.verdict_file)
        verdict = Verdict(data["verdict"])
    except OSError as exc:
        raise CommandError(f"cannot read {args.verdict_file}: {exc.strerror}") from None
    except (ValueError, KeyError, TypeError):
        raise CommandError(f"{args.verdict_file} holds no valid verdict") from None
    print(f"verdict: {verdict.value}")
    return 1 if verdict in (Verdict.FAIL, Verdict.ERROR) else 0


def _compare(args: argparse.Namespace) -> int:
    ids = args.suite_run
    with connect(load_settings()) as conn:
        try:
            text = compare(conn, ids)
        except SuiteNotFoundError as exc:
            raise CommandError(str(exc)) from None
    try:
        Path(args.out).write_text(text)
    except OSError as exc:
        raise CommandError(f"cannot write {args.out}: {exc.strerror}") from None
    return 0


def run(args: argparse.Namespace) -> int:
    handlers = {"gate": _gate, "gate-check": _gate_check, "compare": _compare}
    try:
        return handlers[args.command](args)
    except CommandError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except psycopg.Error as exc:
        print(f"error: database error ({type(exc).__name__})", file=sys.stderr)
        return 2
