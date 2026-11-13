"""Helpers shared by the CLI commands: errors, argument types, settings and the judge."""

from __future__ import annotations

import argparse
import uuid
from typing import Any

import psycopg
from psycopg.rows import dict_row

from agent_replay.cli.db_cmd import _shown_url
from agent_replay.codes import RATE_LIMIT
from agent_replay.config import Settings
from agent_replay.db.cache_repo import CacheRepo
from agent_replay.demo import tools as demo_tools
from agent_replay.llm.factory import build_live_provider, cache_first_chain, make_client
from agent_replay.llm.fake import ScriptedProvider
from agent_replay.runtime.faults import ALL_TOOLS, Fault, parse_fault
from agent_replay.scoring.judge import make_judge
from agent_replay.scoring.service import Judge

Row = dict[str, Any]

FAULT_HELP = (
    "inject a fault (repeatable): tool_timeout:<tool|*>, "
    "malformed_tool_output:<tool|*>, rate_limit:<n>"
)


class CommandError(Exception):
    """A user-facing failure; the message is safe to print."""


def uuid_arg(value: str) -> str:
    """argparse type: a UUID in canonical lower-case form."""
    try:
        return str(uuid.UUID(value))
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid UUID: {value!r}") from None


def fault_arg(value: str) -> Fault:
    try:
        return parse_fault(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def add_fault_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fault",
        action="append",
        default=[],
        type=fault_arg,
        metavar="SPEC",
        help=FAULT_HELP,
    )


def load_settings() -> Settings:
    try:
        return Settings.model_validate({})
    except ValueError:
        # Validation errors may quote the offending input; print nothing from them.
        raise CommandError("invalid settings") from None


def connect(settings: Settings) -> psycopg.Connection[Row]:
    if not settings.database_url:
        raise CommandError("AGENT_REPLAY_DATABASE_URL is not set")
    try:
        return psycopg.connect(settings.database_url, row_factory=dict_row)
    except (psycopg.Error, ValueError) as exc:
        raise CommandError(
            f"database error ({type(exc).__name__}) for {_shown_url(settings.database_url)}"
        ) from None


def check_fault_targets(faults: list[Fault]) -> None:
    """A tool fault must name a demo tool (or ``*``); a typo would inject nothing."""
    for f in faults:
        if (
            f.kind != RATE_LIMIT
            and f.target != ALL_TOOLS
            and f.target not in demo_tools.TOOLS
        ):
            raise CommandError(
                f"fault {f.spec!r} names an unknown tool; "
                f"known tools: {', '.join(sorted(demo_tools.TOOLS))}, *"
            )


def build_judge(conn: psycopg.Connection[Row], settings: Settings) -> Judge:
    """Judge hook over LLMClient and the cache-first chain (no live model: cache hits only)."""
    try:
        live = build_live_provider(settings, "replay")
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    if isinstance(live, ScriptedProvider) and live.has_tool_calls:
        # An agent script would answer (and cache) judge calls with agent replies.
        live = None
    provider = cache_first_chain(settings, "replay", live, cache=CacheRepo(conn))
    return make_judge(make_client(provider, settings), settings)
