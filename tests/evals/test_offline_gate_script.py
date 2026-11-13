"""scripts/offline-gate.sh: the committed baseline PASSes and the committed regression FAILs."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from support import ROOT, pg, scrubbed_env

SCRIPT = ROOT / "scripts" / "offline-gate.sh"


def _database_exists(admin: str, name: str) -> bool:
    with pg.connect(admin) as conn:
        row = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (name,)
        ).fetchone()
    return row is not None


@pytest.mark.skipif(shutil.which("uv") is None, reason="the script runs the CLI via uv")
def test_offline_gate_script_reports_both_expected_verdicts() -> None:
    admin = pg.admin_url()
    pg.connect(admin).close()  # fail early when the server is unreachable
    env = scrubbed_env()
    env["AGENT_REPLAY_DATABASE_URL"] = admin
    p = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    assert "baseline replay:   PASS (expected PASS)" in p.stdout
    assert "prompt regression: FAIL (expected FAIL)" in p.stdout
    names = re.findall(r"throwaway database: (\S+)", p.stderr)
    assert len(names) == 1, p.stderr
    assert not _database_exists(admin, names[0]), (
        "the throwaway database was not dropped"
    )


def _run_script(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


# Stands in for uv: `suite replay` fails, everything else goes to the real uv.
FAILING_REPLAY_UV = """#!/bin/sh
case "$*" in
  *"agent-replay suite replay"*) echo "error: replay broke" >&2; exit 3;;
esac
exec "%s" "$@"
"""


@pytest.mark.skipif(shutil.which("uv") is None, reason="the script runs the CLI via uv")
def test_a_failing_replay_is_reported_and_the_database_dropped(tmp_path: Path) -> None:
    admin = pg.admin_url()
    pg.connect(admin).close()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shim = bindir / "uv"
    shim.write_text(FAILING_REPLAY_UV % shutil.which("uv"))
    shim.chmod(0o755)
    env = scrubbed_env()
    env["AGENT_REPLAY_DATABASE_URL"] = admin
    env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    p = _run_script(env)
    assert p.returncode == 1, p.stdout + p.stderr
    assert "baseline replay:   ERROR (replay stopped)" in p.stdout
    names = re.findall(r"throwaway database: (\S+)", p.stderr)
    assert len(names) == 1, p.stderr
    assert not _database_exists(admin, names[0])


@pytest.mark.skipif(shutil.which("uv") is None, reason="the script runs the CLI via uv")
def test_scratch_directory_is_removed_when_the_database_cannot_be_created(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    env = scrubbed_env()
    # Nothing listens on port 1, so creating the throwaway database fails at once.
    env["AGENT_REPLAY_DATABASE_URL"] = (
        "postgresql://postgres:postgres@127.0.0.1:1/postgres"
    )
    env["TMPDIR"] = str(scratch)
    p = _run_script(env)
    assert p.returncode != 0
    assert not list(scratch.glob("agent-replay-gate.*")), list(scratch.iterdir())
