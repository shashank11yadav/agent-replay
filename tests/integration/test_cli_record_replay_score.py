"""record -> replay -> score through the real CLI, offline."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import uuid
from importlib import resources
from pathlib import Path
from typing import Any

from support import scrubbed_env

SKEL = Path(str(resources.files("agent_replay") / "demo" / "example"))
TASKS = SKEL / "task.jsonl"
SCRIPT = SKEL / "script.json"
SENTINEL_KEY = "AIzaSENTINELKEY1234567890abcdef"
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def cli(
    db_url: str,
    *args: str,
    env_extra: dict[str, str] | None = None,
    scripted: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = scrubbed_env()
    env["AGENT_REPLAY_DATABASE_URL"] = db_url
    env["AGENT_REPLAY_GEMINI_API_KEY"] = ""
    if scripted:
        env["AGENT_REPLAY_PROVIDER"] = "scripted"
        env["AGENT_REPLAY_SCRIPT_PATH"] = str(SCRIPT)
    env.update(env_extra or {})
    code = "import sys; from agent_replay.cli import main; sys.exit(main())"
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def last_line(p: subprocess.CompletedProcess[str]) -> str:
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    assert lines, f"empty stdout; stderr={p.stderr!r}"
    return lines[-1].strip()


def record(db_url: str) -> str:
    p = cli(
        db_url,
        "record",
        "--task",
        "example-1",
        "--tasks",
        str(TASKS),
        "--variant",
        "structured",
    )
    assert p.returncode == 0, (p.stdout, p.stderr)
    rid = last_line(p)
    assert _UUID.match(rid), rid
    return rid


def replay(db_url: str, rid: str) -> str:
    p = cli(db_url, "replay", "--run", rid, "--cache-only")
    assert p.returncode == 0, (p.stdout, p.stderr)
    new = last_line(p)
    assert _UUID.match(new), new
    return new


def test_record_exits_0_and_prints_uuid_last_line(db_url: str, raw: Any) -> None:
    rid = record(db_url)
    row = raw.execute(
        "SELECT mode, status, task_id FROM runs WHERE id=%s", (rid,)
    ).fetchone()
    assert row == ("record", "ok", "example-1")


def test_replay_cache_only_prints_new_run_id(db_url: str) -> None:
    rid = record(db_url)
    new = replay(db_url, rid)
    assert new != rid


def test_replayed_run_row_and_spans(db_url: str, raw: Any) -> None:
    rid = record(db_url)
    new = replay(db_url, rid)
    mode, src = raw.execute(
        "SELECT mode, source_run_id::text FROM runs WHERE id=%s", (new,)
    ).fetchone()
    assert mode == "replay" and src == rid

    def counts(r: str) -> tuple[int, int]:
        return (
            raw.execute(
                "SELECT count(*) FROM spans WHERE run_id=%s AND kind='model'", (r,)
            ).fetchone()[0],
            raw.execute(
                "SELECT count(*) FROM spans WHERE run_id=%s AND kind='tool'", (r,)
            ).fetchone()[0],
        )

    assert counts(new) == counts(rid)
    assert counts(rid)[0] >= 2 and counts(rid)[1] >= 1
    statuses = [
        r[0]
        for r in raw.execute(
            "SELECT cache_status FROM spans WHERE run_id=%s AND kind='model'", (new,)
        ).fetchall()
    ]
    assert statuses and all(s == "hit" for s in statuses)


def test_score_prints_json_with_deterministic_pass_true(db_url: str, raw: Any) -> None:
    new = replay(db_url, record(db_url))
    p = cli(db_url, "score", "--run", new, "--tasks", str(TASKS))
    assert p.returncode == 0, (p.stdout, p.stderr)
    data = json.loads(p.stdout)
    assert data["deterministic_pass"] is True
    n = raw.execute("SELECT count(*) FROM scores WHERE run_id=%s", (new,)).fetchone()[0]
    assert n >= 4


def test_replay_unknown_run_not_found(db_url: str) -> None:
    p = cli(db_url, "replay", "--run", str(uuid.uuid4()))
    assert p.returncode != 0
    assert "run not found" in (p.stdout + p.stderr).lower()
    assert "traceback" not in p.stderr.lower()


def test_record_gemini_without_key_fails_without_network(db_url: str) -> None:
    dead = "http://127.0.0.1:9"  # any network attempt would fail differently / hang
    p = cli(
        db_url,
        "record",
        "--task",
        "example-1",
        "--tasks",
        str(TASKS),
        "--variant",
        "structured",
        scripted=False,
        env_extra={
            "AGENT_REPLAY_PROVIDER": "gemini",
            "HTTPS_PROXY": dead,
            "HTTP_PROXY": dead,
            "ALL_PROXY": dead,
        },
    )
    assert p.returncode != 0
    out = (p.stdout + p.stderr).lower()
    assert "key" in out and "required" in out
    assert "traceback" not in out
    for bad in ("connect", "proxy", "network"):
        assert bad not in out


def test_api_key_never_printed(db_url: str) -> None:
    env = {"AGENT_REPLAY_GEMINI_API_KEY": SENTINEL_KEY}
    outs: list[subprocess.CompletedProcess[str]] = []
    p = cli(
        db_url,
        "record",
        "--task",
        "example-1",
        "--tasks",
        str(TASKS),
        "--variant",
        "structured",
        env_extra=env,
    )
    assert p.returncode == 0, p.stderr
    outs.append(p)
    rid = last_line(p)
    p2 = cli(db_url, "replay", "--run", rid, "--cache-only", env_extra=env)
    assert p2.returncode == 0, p2.stderr
    outs.append(p2)
    outs.append(
        cli(
            db_url,
            "score",
            "--run",
            last_line(p2),
            "--tasks",
            str(TASKS),
            env_extra=env,
        )
    )
    outs.append(cli(db_url, "replay", "--run", str(uuid.uuid4()), env_extra=env))
    outs.append(cli(db_url, "replay", "--run", "not-a-uuid", env_extra=env))
    for o in outs:
        assert SENTINEL_KEY not in o.stdout + o.stderr


def test_db_password_not_printed_on_connection_error() -> None:
    url = "postgresql://postgres:s3cretPW9@127.0.0.1:1/nodb"
    p = cli(url, "replay", "--run", str(uuid.uuid4()))
    assert p.returncode != 0
    assert "s3cretPW9" not in p.stdout + p.stderr
    assert "traceback" not in p.stderr.lower()


def test_replay_that_ends_skipped_exits_1_and_still_prints_the_run_id(
    db_url: str, raw: Any
) -> None:
    rid = record(db_url)
    # Another prompt variant sends different requests, so --cache-only misses.
    p = cli(db_url, "replay", "--run", rid, "--variant", "minimal", "--cache-only")
    assert p.returncode == 1, (p.stdout, p.stderr)
    new = last_line(p)
    status = raw.execute("SELECT status FROM runs WHERE id = %s", (new,)).fetchone()
    assert status == ("skipped",)
