"""The deliberate prompt regression recorded with claude_code.

Checked from files and offline commands only:

- the evals/regression/ manifest;
- the offline gate on a throwaway database with the CLI disabled;
- evals/results/regression.md, against the verdict computed here from the fixtures
  (no measured numbers are hard-coded).

No live model is called.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from support import ROOT, pg, scrubbed_env

from agent_replay.fixtures import FILES

EVALS = ROOT / "evals"
BASELINE = EVALS / "baseline"
REGRESSION = EVALS / "regression"
RUNBOOK = EVALS / "RUNBOOK.md"
RESULT = EVALS / "results" / "regression.md"
GATE_JSON = EVALS / "gate.json"


# --- helpers -----------------------------------------------------------------------------


def jload(p: Path) -> Any:
    return json.loads(p.read_text())


def base_manifest() -> dict[str, Any]:
    return jload(BASELINE / "manifest.json")


def reg_manifest() -> dict[str, Any]:
    assert (REGRESSION / "manifest.json").is_file(), (
        "evals/regression/manifest.json missing"
    )
    return jload(REGRESSION / "manifest.json")


def flat(text: str) -> str:
    """Lower-case text without blockquote markers and with folded whitespace."""
    no_quote = re.sub(r"(?m)^\s*>+ ?", "", text)
    return re.sub(r"\s+", " ", no_quote).lower()


def tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    out: list[tuple[list[str], list[list[str]]]] = []
    cur: list[list[str]] = []

    def flush() -> None:
        if len(cur) >= 2:
            body = [
                r
                for r in cur[1:]
                if not all(re.fullmatch(r":?-+:?", c) for c in r if c)
            ]
            out.append((cur[0], body))
        cur.clear()

    for line in text.splitlines():
        if line.strip().startswith("|"):
            inner = line.strip().strip("|")
            cur.append(
                [c.strip().replace("\\", "") for c in re.split(r"(?<!\\)\|", inner)]
            )
        else:
            flush()
    flush()
    return out


def numbers(cell: str) -> list[float]:
    return [float(m.replace(",", "")) for m in re.findall(r"\d[\d,]*\.?\d*", cell)]


def metric_numbers(
    text: str, words: tuple[str, ...], avoid: tuple[str, ...] = ()
) -> list[float]:
    """Numbers of a metric, whether the table has one row or one column per metric."""

    def hit(cell: str) -> bool:
        low = cell.lower()
        return all(w in low for w in words) and not any(a in low for a in avoid)

    found: list[float] = []
    for header, body in tables(text):
        for i, h in enumerate(header):
            if hit(h):
                for row in body:
                    if i < len(row):
                        found += numbers(row[i])
        for row in body:
            if row and hit(row[0]):
                for cell in row[1:]:
                    found += numbers(cell)
    return found


def has(values: list[float], expected: float, tol: float) -> bool:
    return any(
        abs(v - expected) <= tol or abs(abs(v) - abs(expected)) <= tol for v in values
    )


# --- database ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scratch_db() -> Iterator[str]:
    with pg.throwaway_database("ar_test_regression", migrate=False) as url:
        yield url


def env_for(url: str) -> dict[str, str]:
    env = scrubbed_env()
    env.pop("GITHUB_STEP_SUMMARY", None)
    env.update(
        AGENT_REPLAY_DATABASE_URL=url,
        AGENT_REPLAY_PROVIDER="claude_code",
        AGENT_REPLAY_CLAUDE_CLI="",
        AGENT_REPLAY_GEMINI_API_KEY="",
        AGENT_REPLAY_BACKOFF_BASE_S="0",
    )
    return env


def cli(url: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from agent_replay.cli import main; sys.exit(main())",
            *args,
        ],
        env=env_for(url),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )


def ok(p: subprocess.CompletedProcess[str]) -> subprocess.CompletedProcess[str]:
    assert p.returncode == 0, (p.args[3:], p.stdout, p.stderr)
    return p


@pytest.fixture(scope="module")
def gate_run(scratch_db: str, tmp_path_factory: Any) -> dict[str, Any]:
    """Import the baseline, then the regression, gate them offline, run gate-check."""
    from agent_replay.db.migrate import apply_migrations

    assert (REGRESSION / "manifest.json").is_file(), "evals/regression/ is missing"
    apply_migrations(scratch_db)
    ok(cli(scratch_db, "fixtures", "import", str(BASELINE)))
    ok(cli(scratch_db, "fixtures", "import", str(REGRESSION)))
    base_id = base_manifest()["suite_run_id"]
    cand_id = reg_manifest()["suite_run_id"]
    out_dir = tmp_path_factory.mktemp("regression_gate")
    gate = cli(
        scratch_db,
        "gate",
        "--baseline",
        base_id,
        "--candidate",
        cand_id,
        "--thresholds",
        str(GATE_JSON),
        "--out-dir",
        str(out_dir),
    )
    verdict_path = out_dir / "verdict.json"
    assert verdict_path.is_file(), (gate.stdout, gate.stderr)
    check = cli(scratch_db, "gate-check", str(verdict_path))
    return {
        "gate": gate,
        "check": check,
        "verdict": jload(verdict_path),
        "base_id": base_id,
        "cand_id": cand_id,
        "url": scratch_db,
    }


# --- regression export ------------------------------------------------------


def test_regression_export_has_all_fixture_files() -> None:
    for name in FILES:
        assert (REGRESSION / name).is_file(), f"evals/regression/{name} missing"


def test_regression_manifest_records_the_regressed_variant_and_thirty_runs() -> None:
    m = reg_manifest()
    assert m["variant"] == "structured_regressed"
    assert m["suite_run"]["variant"] == "structured_regressed"
    assert m["run_count"] == 30
    rows = [
        ln for ln in (REGRESSION / "runs.jsonl").read_text().splitlines() if ln.strip()
    ]
    assert len(rows) == 30


def test_regression_manifest_has_the_same_provider_and_models_as_the_baseline() -> None:
    b, m = base_manifest(), reg_manifest()
    for key in ("provider", "agent_model", "judge_model"):
        assert m[key] == b[key], key
        assert m[key], key
    assert m["provider"] == "claude_code"


def test_regression_source_suite_run_id_is_the_baseline_suite_run_id() -> None:
    b, m = base_manifest(), reg_manifest()
    assert m["suite_run"]["source_suite_run_id"] == b["suite_run_id"]
    assert m["suite_run_id"] != b["suite_run_id"]
    assert m["suite_run"]["mode"] == "replay"


def test_regression_replays_the_same_task_file_as_the_baseline() -> None:
    assert reg_manifest()["tasks_sha256"] == base_manifest()["tasks_sha256"]


# --- offline gate -----------------------------------------------------


def test_offline_gate_writes_verdict_fail(gate_run: dict[str, Any]) -> None:
    v = gate_run["verdict"]
    assert v["verdict"] == "FAIL", v
    assert v["baseline_id"] == gate_run["base_id"]
    assert v["candidate_id"] == gate_run["cand_id"]
    assert v["candidate"]["task_count"] == 30
    assert v["candidate"]["skipped_count"] == 0
    assert v["skipped_requests"] == 0


def test_offline_gate_check_exits_one(gate_run: dict[str, Any]) -> None:
    assert gate_run["check"].returncode == 1, (
        gate_run["check"].stdout,
        gate_run["check"].stderr,
    )


def test_offline_gate_trips_at_least_one_metric(gate_run: dict[str, Any]) -> None:
    assert [c for c in gate_run["verdict"]["checks"] if c["status"] == "FAIL"]


# --- regression.md ----------------------------------------------------------


def result_text() -> str:
    assert RESULT.is_file(), "evals/results/regression.md does not exist"
    return RESULT.read_text()


def test_regression_md_has_a_table_with_every_metric_for_before_and_after() -> None:
    text = result_text()
    assert tables(text), "no markdown table"
    for label, words, avoid in (
        ("pass rate", ("pass", "rate"), ("deterministic", "judge")),
        ("mean total tokens", ("token",), ()),
        ("p95 latency", ("p95",), ()),
        ("notional USD", ("usd",), ()),
    ):
        assert len(metric_numbers(text, words, avoid)) >= 2, (
            f"no before and after values for {label}"
        )
    low = text.lower()
    assert "notional" in low
    assert re.search(r"before|baseline", low) and re.search(
        r"after|regress|candidate", low
    )


def test_regression_md_pass_rate_matches_the_fixtures(gate_run: dict[str, Any]) -> None:
    v = gate_run["verdict"]
    nums = metric_numbers(result_text(), ("pass", "rate"), ("deterministic", "judge"))
    assert has(nums, v["baseline"]["pass_rate"] * 100, 0.06), nums
    assert has(nums, v["candidate"]["pass_rate"] * 100, 0.06), nums
    check = next(c for c in v["checks"] if c["metric"].lower().startswith("pass rate"))
    assert has(nums, check["delta"], 0.06), (check["delta"], nums)


def test_regression_md_tokens_match_the_fixtures(gate_run: dict[str, Any]) -> None:
    v = gate_run["verdict"]
    nums = metric_numbers(result_text(), ("token",))
    assert has(nums, v["baseline"]["mean_total_tokens"], 0.51), nums
    assert has(nums, v["candidate"]["mean_total_tokens"], 0.51), nums
    check = next(c for c in v["checks"] if "token" in c["metric"].lower())
    assert has(nums, check["delta"], 0.06), (check["delta"], nums)


def test_regression_md_p95_matches_the_fixtures(gate_run: dict[str, Any]) -> None:
    v = gate_run["verdict"]
    nums = metric_numbers(result_text(), ("p95",))
    assert has(nums, v["baseline"]["p95_latency_ms"], 0.5), nums
    assert has(nums, v["candidate"]["p95_latency_ms"], 0.5), nums
    check = next(c for c in v["checks"] if "p95" in c["metric"].lower())
    assert has(nums, check["delta"], 0.06), (check["delta"], nums)


def agent_usd(directory: Path) -> float:
    rows = [
        json.loads(ln)
        for ln in (directory / "runs.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    return float(sum(r["cost_usd"] or 0 for r in rows))


def test_regression_md_notional_usd_is_consistent_with_the_recorded_costs() -> None:
    """Agent calls alone are a lower bound; judge calls add less than the run budget."""
    nums = metric_numbers(result_text(), ("usd",))
    for directory in (BASELINE, REGRESSION):
        low = agent_usd(directory)
        assert any(low - 0.00006 <= n <= low + 1.5 for n in nums), (
            directory.name,
            low,
            nums,
        )
    assert agent_usd(BASELINE) > 0 and agent_usd(REGRESSION) > 0


def test_regression_md_states_the_thresholds_from_gate_json() -> None:
    text = result_text()
    low = text.lower()
    assert "threshold" in low
    all_numbers = numbers(text)
    for key, value in jload(GATE_JSON).items():
        assert has(all_numbers, float(value), 0.0001), (key, value)


def test_regression_md_names_the_metric_that_tripped(gate_run: dict[str, Any]) -> None:
    text = result_text()
    paragraphs = [
        p for p in re.split(r"\n\s*\n", text) if re.search(r"trip", p, re.IGNORECASE)
    ]
    assert paragraphs, "regression.md never says which metric tripped the gate"
    said = flat(" ".join(paragraphs))
    keywords = {"pass rate": "pass rate", "token": "token", "p95": "p95"}
    failed = [c for c in gate_run["verdict"]["checks"] if c["status"] == "FAIL"]
    assert failed
    for c in failed:
        word = next(w for k, w in keywords.items() if k in c["metric"].lower())
        assert word in said, (c["metric"], said)
    assert "fail" in flat(text)
