"""The committed fixture bundles still verify."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from support import ROOT, scrubbed_env

from agent_replay.db import runs_repo
from agent_replay.fixtures import import_bundle, load_bundle

BUNDLES = [
    p
    for p in (ROOT / "evals" / "baseline", ROOT / "evals" / "regression")
    if p.is_dir()
]


@pytest.mark.parametrize("bundle", BUNDLES, ids=lambda p: p.name)
def test_fixtures_verify_exits_zero(bundle: Path) -> None:
    env = scrubbed_env()
    code = "import sys; from agent_replay.cli import main; sys.exit(main())"
    done = subprocess.run(
        [sys.executable, "-c", code, "fixtures", "verify", str(bundle)],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, (done.stdout, done.stderr)


def test_baseline_imports_into_a_database_that_has_its_own_spans(conn: Any) -> None:
    # A local database whose span ids overlap the ones in the exported files.
    run_id = runs_repo.save_run(
        conn,
        {"task_id": "local", "variant": "minimal", "mode": "record", "status": "ok"},
        [{"seq": i, "kind": "tool", "name": f"step {i}"} for i in range(200)],
    )
    bundle = load_bundle(ROOT / "evals" / "baseline")
    counts = import_bundle(conn, bundle)
    assert counts["runs"] == 30
    assert counts["spans"] == len(bundle.spans)
    assert len(runs_repo.list_spans(conn, str(run_id))) == 200
    assert set(import_bundle(conn, bundle).values()) == {0}
