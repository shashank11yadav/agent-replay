"""`agent-replay fixtures export|import|verify`: suite runs as JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg

from agent_replay.cli.common import CommandError, connect, load_settings, uuid_arg
from agent_replay.fixtures import (
    FixtureError,
    clean,
    export_suite,
    import_bundle,
    load_bundle,
    verify_bundle,
)


def add_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    fx = subparsers.add_parser("fixtures", help="export, import or verify fixtures")
    sub = fx.add_subparsers(dest="fixtures_command", metavar="<fixtures command>")
    sub.required = True

    ex = sub.add_parser("export", help="write a suite run as JSONL files")
    ex.add_argument("--suite-run", required=True, type=uuid_arg, help="suite run id")
    ex.add_argument("--out", required=True, help="output directory")

    im = sub.add_parser("import", help="load a fixture directory into the database")
    im.add_argument("dir", help="fixture directory")

    ve = sub.add_parser("verify", help="check a fixture directory is replayable")
    ve.add_argument("dir", help="fixture directory")


def _export(args: argparse.Namespace) -> int:
    settings = load_settings()
    with connect(settings) as conn:
        try:
            manifest = export_suite(conn, args.suite_run, Path(args.out), settings)
        except OSError as exc:
            # A rename names the target second; prefer it over the temp file.
            path = exc.filename2 or exc.filename or args.out
            raise CommandError(f"cannot write {path}: {exc.strerror}") from None
    print(f"exported {manifest.run_count} runs to {args.out}")
    return 0


def _import(args: argparse.Namespace) -> int:
    bundle = load_bundle(Path(args.dir))
    settings = load_settings()
    with connect(settings) as conn:
        counts = import_bundle(conn, bundle)
    summary = ", ".join(f"{k} {v}" for k, v in counts.items())
    print(f"imported suite {bundle.manifest.suite_run_id} (new rows: {summary})")
    return 0


def _verify(args: argparse.Namespace) -> int:
    bundle = load_bundle(Path(args.dir))
    gaps = verify_bundle(bundle)
    if not gaps:
        print(f"ok: {len(bundle.runs)} runs complete")
        return 0
    for run_id, items in gaps.items():
        for item in items:
            print(f"run {run_id}: {item}")
    print(f"error: {len(gaps)} runs incomplete", file=sys.stderr)
    return 1


def run(args: argparse.Namespace) -> int:
    handlers = {"export": _export, "import": _import, "verify": _verify}
    try:
        return handlers[args.fixtures_command](args)
    except (CommandError, FixtureError) as exc:
        print(f"error: {clean(str(exc), 300)}", file=sys.stderr)
        return 1
    except psycopg.Error as exc:
        print(f"error: database error ({type(exc).__name__})", file=sys.stderr)
        return 1
