"""`agent-replay db` commands."""

from __future__ import annotations

import argparse
import sys
from urllib.parse import urlsplit

import psycopg

from agent_replay.config import Settings, redact_url
from agent_replay.db.migrate import MigrationError, apply_migrations


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    db = subparsers.add_parser("db", help="database commands")
    db_sub = db.add_subparsers(dest="db_command", metavar="<db-command>", required=True)
    db_sub.add_parser("migrate", help="apply pending SQL migrations")


def _shown_url(url: str) -> str:
    """Redacted URL for messages; never raises."""
    try:
        if not urlsplit(url).scheme:
            return "<unparseable URL>"
        return redact_url(url) or "<unparseable URL>"
    except Exception:  # noqa: BLE001 - any parse failure must not leak the URL
        return "<unparseable URL>"


def run(args: argparse.Namespace) -> int:
    if args.db_command != "migrate":  # pragma: no cover - argparse enforces choices
        return 2
    try:
        url = Settings.model_validate({}).database_url
    except ValueError:
        # Validation errors may quote the offending input, so print nothing from them.
        print("error: invalid settings (<unparseable URL>)", file=sys.stderr)
        return 1
    if not url:
        print("error: AGENT_REPLAY_DATABASE_URL is not set", file=sys.stderr)
        return 1
    try:
        applied = apply_migrations(url)
    except MigrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (psycopg.Error, ValueError) as exc:
        # Never echo str(exc) or chain it: parse errors can quote the raw URL.
        print(
            f"error: database error ({type(exc).__name__}) for {_shown_url(url)}",
            file=sys.stderr,
        )
        return 1
    if not applied:
        print("Schema is up to date.")
    else:
        print(f"Applied {len(applied)} migration(s): {', '.join(applied)}")
    return 0
