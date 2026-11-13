"""Checks that keep the Markdown docs honest about the code.

They do not pin wording: each one compares something a doc shows (a command, a flag,
an example value) with the code or with a rule that holds for every doc.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest
from support import ROOT

from agent_replay.cli import build_parser
from agent_replay.llm.claude_code import ClaudeCodeProvider
from agent_replay.llm.types import Message, ModelRequest

DOCS = sorted(
    p
    for p in [
        ROOT / "README.md",
        ROOT / "evals" / "RUNBOOK.md",
        ROOT / "deploy" / "DEPLOY.md",
        *(ROOT / "docs").rglob("*.md"),
    ]
    if p.is_file()
)
# Docs whose code snippets show `agent-replay` commands a reader is meant to run.
COMMAND_DOCS = [
    p
    for p in DOCS
    if p.name == "README.md" or p.name == "RUNBOOK.md" or p.parent.name == "docs"
]


def _subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _commands_in(text: str) -> set[tuple[str, str | None]]:
    """`agent-replay <command> [<sub>]` pairs shown in fenced or inline code."""
    snippets = re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    snippets += re.findall(r"`([^`\n]+)`", text)
    found: set[tuple[str, str | None]] = set()
    for snip in snippets:
        for m in re.finditer(
            r"agent-replay\s+([a-z][a-z-]*)(?:\s+([a-z][a-z-]*))?", snip
        ):
            found.add((m.group(1), m.group(2)))
    return found


def test_readme_shows_cli_commands() -> None:
    assert _commands_in((ROOT / "README.md").read_text(encoding="utf-8"))


@pytest.mark.parametrize("doc", COMMAND_DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_command_shown_in_a_doc_exists_in_the_cli(doc: Path) -> None:
    commands = _subcommands(build_parser())
    for cmd, sub in sorted(_commands_in(doc.read_text(encoding="utf-8")), key=str):
        assert cmd in commands, f"{doc.name}: `agent-replay {cmd}` is not a CLI command"
        nested = _subcommands(commands[cmd])
        if nested and sub is not None:
            assert sub in nested, (
                f"{doc.name}: `agent-replay {cmd} {sub}` does not exist"
            )


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_real_looking_key_in_docs(doc: Path) -> None:
    text = doc.read_text(encoding="utf-8")
    assert not re.search(r"AIza[0-9A-Za-z_\-]{20,}", text)
    for line in text.splitlines():
        m = re.search(r"AGENT_REPLAY_GEMINI_API_KEY\s*=\s*(\S+)", line)
        if m:
            value = m.group(1).strip("`\"'")
            assert value == "" or re.match(
                r"[<$_{\[]|your|xxx|\.\.\.", value, re.IGNORECASE
            ), line


def test_providers_doc_flag_table_covers_every_claude_code_flag() -> None:
    request = ModelRequest(
        provider="claude_code",
        model="claude-haiku-5-5",
        system="s",
        messages=[Message(role="user", text="hi")],
    )
    argv = ClaudeCodeProvider("claude", max_turns=3)._argv(request)
    flags = {a for a in argv[1:] if a.startswith("-")}
    assert {"-p", "--model", "--max-turns"} <= flags
    rows = [
        ln
        for ln in (ROOT / "docs" / "providers.md")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.startswith("|")
    ]
    for flag in sorted(flags):
        assert any(re.search(rf"`{re.escape(flag)}[` ]", row) for row in rows), (
            f"docs/providers.md has no flag-table row for {flag}"
        )
