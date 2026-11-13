"""Record a small scripted suite and drive the CLI against it."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import psycopg

from support import scrubbed_env

N = 3


def base_env(db_url: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = scrubbed_env()
    env.pop("GITHUB_STEP_SUMMARY", None)
    env.update(
        AGENT_REPLAY_DATABASE_URL=db_url,
        AGENT_REPLAY_GEMINI_API_KEY="",
        AGENT_REPLAY_BACKOFF_BASE_S="0",
        AGENT_REPLAY_RUN_BUDGET_USD="100",
    )
    env.update(extra or {})
    return env


def cli(
    db_url: str, *args: str, extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the CLI. AGENT_REPLAY_PROVIDER is NOT set unless `extra` sets it."""
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from agent_replay.cli import main; sys.exit(main())",
            *args,
        ],
        env=base_env(db_url, extra),
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


def ok(p: subprocess.CompletedProcess[str]) -> subprocess.CompletedProcess[str]:
    assert p.returncode == 0, (p.args[3:], p.stdout, p.stderr)
    return p


def last_line(p: subprocess.CompletedProcess[str]) -> str:
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    assert lines, f"empty stdout; stderr={p.stderr!r}"
    return lines[-1].strip()


def write_tasks(tmp: Path) -> Path:
    lines = [
        json.dumps(
            {
                "id": f"g-{i}",
                "prompt": f"What is the status of order A1001 (case {i})?",
                "category": "lookup",
                "expected_text": ["shipped"],
                "must_call_tools": ["get_order"],
                "max_steps": 4,
                "rubric": f"States that order A1001 has shipped (case {i}).",
            }
        )
        for i in range(1, N + 1)
    ]
    p = tmp / "tasks.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return p


def _script(tmp: Path, name: str, entries: list[Any]) -> Path:
    p = tmp / name
    p.write_text(json.dumps(entries))
    return p


def build_suite(db_url: str, tmp: Path, score: bool = True) -> tuple[str, Path]:
    """Record (and optionally score) a scripted suite; returns (suite id, tasks path)."""
    tmp.mkdir(parents=True, exist_ok=True)
    tasks = write_tasks(tmp)
    pair = [
        {
            "tool_calls": [
                {"id": "x", "name": "get_order", "args": {"order_id": "A1001"}}
            ]
        },
        {"text": "Order A1001 has shipped."},
    ]
    agent = _script(tmp, "agent.json", pair * (N + 3))
    judge = _script(
        tmp, "judge.json", [{"text": '{"pass": true, "reason": "ok"}'}] * (2 * N)
    )
    scripted = {"AGENT_REPLAY_PROVIDER": "scripted"}
    rec = ok(
        cli(
            db_url,
            "suite",
            "record",
            "--tasks",
            str(tasks),
            "--variant",
            "structured",
            "--label",
            "base",
            extra={**scripted, "AGENT_REPLAY_SCRIPT_PATH": str(agent)},
        )
    )
    sid = last_line(rec)
    if score:
        ok(
            cli(
                db_url,
                "suite",
                "score",
                "--suite-run",
                sid,
                "--tasks",
                str(tasks),
                extra={**scripted, "AGENT_REPLAY_SCRIPT_PATH": str(judge)},
            )
        )
    return sid, tasks


def export(
    db_url: str, sid: str, out: Path, extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return cli(
        db_url,
        "fixtures",
        "export",
        "--suite-run",
        sid,
        "--out",
        str(out),
        extra={"AGENT_REPLAY_PROVIDER": "scripted", **(extra or {})},
    )


def connect(url: str) -> Any:
    return psycopg.connect(url, autocommit=True)


def agent_hashes(c: Any) -> list[str]:
    rows = c.execute(
        "SELECT DISTINCT payload->>'request_hash' FROM spans "
        "WHERE kind = 'model' ORDER BY 1"
    ).fetchall()
    return [r[0] for r in rows if r[0]]


def judge_hashes(c: Any) -> list[str]:
    rows = c.execute(
        "SELECT DISTINCT detail->>'request_hash' FROM scores "
        "WHERE check_name = 'judge' ORDER BY 1"
    ).fetchall()
    return [r[0] for r in rows if r[0]]


def retag(c: Any, hashes: list[str], provider: str | None, model: str | None) -> None:
    """Rewrite provider/model columns of the given cache rows (None = keep)."""
    if provider is not None:
        c.execute(
            "UPDATE model_cache SET provider = %s WHERE request_hash = ANY(%s)",
            (provider, hashes),
        )
    if model is not None:
        c.execute(
            "UPDATE model_cache SET model = %s WHERE request_hash = ANY(%s)",
            (model, hashes),
        )


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
