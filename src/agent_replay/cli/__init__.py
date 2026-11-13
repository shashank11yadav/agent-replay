"""Command line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from agent_replay import __version__
from agent_replay.cli import db_cmd, fixtures_cmd, report_cmd, run_cmd, suite_cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-replay",
        description="Record, replay and regression-test LLM agent runs.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    db_cmd.add_parser(subparsers)
    run_cmd.add_parsers(subparsers)
    suite_cmd.add_parser(subparsers)
    report_cmd.add_parsers(subparsers)
    fixtures_cmd.add_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "db":
        return db_cmd.run(args)
    if args.command in ("record", "replay", "score"):
        return run_cmd.run(args)
    if args.command == "suite":
        return suite_cmd.run(args)
    if args.command == "fixtures":
        return fixtures_cmd.run(args)
    if args.command in report_cmd.COMMANDS:
        return report_cmd.run(args)
    return 2
