"""Record then replay through the real CLI with a fake `claude` script."""

from __future__ import annotations

import re
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Any

from claude_cli_harness import FakeClaude, step_ok, step_text
from support import scrubbed_env

SKEL = Path(str(resources.files("agent_replay") / "demo" / "example"))
TASKS = SKEL / "task.jsonl"
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def cli(
    db_url: str, fake: FakeClaude, *args: str, cli_path: str | None = None
) -> subprocess.CompletedProcess[str]:
    env = scrubbed_env()
    env.update(fake.env())
    env["AGENT_REPLAY_DATABASE_URL"] = db_url
    env["AGENT_REPLAY_GEMINI_API_KEY"] = ""
    if cli_path is not None:
        env["AGENT_REPLAY_CLAUDE_CLI"] = cli_path
    code = "import sys; from agent_replay.cli import main; sys.exit(main())"
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def last_line(p: subprocess.CompletedProcess[str]) -> str:
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    assert lines, f"empty stdout; stderr={p.stderr!r}"
    return lines[-1].strip()


def test_record_with_claude_code_then_replay_serves_everything_from_cache(
    db_url: str, raw: Any, tmp_path: Path
) -> None:
    fake = FakeClaude(tmp_path)
    fake.script(
        step_ok({"tool_calls": [{"name": "get_order", "args": {"order_id": "A1001"}}]}),
        step_text("Order A1001 has shipped."),
    )
    rec = cli(
        db_url,
        fake,
        "record",
        "--task",
        "example-1",
        "--tasks",
        str(TASKS),
        "--variant",
        "minimal",
    )
    assert rec.returncode == 0, (rec.stdout, rec.stderr)
    rid = last_line(rec)
    assert _UUID.match(rid), rid
    assert len(fake.calls()) == 2

    status, mode, final = raw.execute(
        "SELECT status, mode, final_output FROM runs WHERE id=%s", (rid,)
    ).fetchone()
    assert (status, mode) == ("ok", "record")
    assert "shipped" in final

    spans = raw.execute(
        "SELECT attributes->>'gen_ai.provider.name', attributes->>'gen_ai.request.model', "
        "cache_status FROM spans WHERE run_id=%s AND kind='model' ORDER BY seq",
        (rid,),
    ).fetchall()
    assert len(spans) == 2
    assert all(s[0] == "claude_code" and s[1] == "claude-haiku-5-5" for s in spans)
    cached = raw.execute("SELECT DISTINCT provider, model FROM model_cache").fetchall()
    assert cached == [("claude_code", "claude-haiku-5-5")]

    calls_before = len(fake.calls())
    rep = cli(db_url, fake, "replay", "--run", rid, cli_path="")
    assert rep.returncode == 0, (rep.stdout, rep.stderr)
    new = last_line(rep)
    assert _UUID.match(new) and new != rid
    assert len(fake.calls()) == calls_before, "replay must not start the CLI"

    new_status, new_mode, src = raw.execute(
        "SELECT status, mode, source_run_id::text FROM runs WHERE id=%s", (new,)
    ).fetchone()
    assert (new_status, new_mode, src) == ("ok", "replay", rid)
    statuses = [
        r[0]
        for r in raw.execute(
            "SELECT cache_status FROM spans WHERE run_id=%s AND kind='model'", (new,)
        ).fetchall()
    ]
    assert len(statuses) == 2 and all(s == "hit" for s in statuses)


def test_record_without_a_cli_fails_naming_the_setting(
    db_url: str, tmp_path: Path
) -> None:
    fake = FakeClaude(tmp_path)
    p = cli(
        db_url,
        fake,
        "record",
        "--task",
        "example-1",
        "--tasks",
        str(TASKS),
        "--variant",
        "minimal",
        cli_path="",
    )
    assert p.returncode != 0
    assert "AGENT_REPLAY_CLAUDE_CLI" in p.stdout + p.stderr
    assert "traceback" not in (p.stdout + p.stderr).lower()
    assert fake.calls() == []
