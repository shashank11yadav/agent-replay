"""Fixture manifests pin the provider and models.

Cache entries of a scripted suite are retagged directly in model_cache (provider/model
columns) to get distinguishable values, so the tests prove the manifest comes from the
exported cache entries and not from the current settings. The `claude_code` fixture of the
import test is made by rewriting an exported scripted fixture, so no CLI is needed (import
re-hashes each stored request, which still says 'scripted', so the files stay valid).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from support.suites import (
    agent_hashes,
    build_suite,
    cli,
    connect,
    export,
    judge_hashes,
    ok,
    read_jsonl,
    retag,
)

from agent_replay.fixtures import FILES

KEYS = ("provider", "agent_model", "judge_model")


def manifest(d: Path) -> dict:
    return json.loads((d / "manifest.json").read_text())


# --- line 1 -----------------------------------------------------------------------


def test_manifest_provider_and_models_come_from_exported_cache_entries(
    source_db_url: str, tmp_path: Path
) -> None:
    sid, _ = build_suite(source_db_url, tmp_path)
    with connect(source_db_url) as c:
        a, j = agent_hashes(c), judge_hashes(c)
        assert a and j and not set(a) & set(j)
        retag(c, a, "prov-x", "agent-m.1")
        retag(c, j, "prov-x", "judge-m_2")
    out = tmp_path / "fx"
    # Settings say something else: the manifest must not echo them.
    ok(
        export(
            source_db_url,
            sid,
            out,
            {
                "AGENT_REPLAY_MODEL": "settings-agent-model",
                "AGENT_REPLAY_JUDGE_MODEL": "settings-judge-model",
            },
        )
    )
    m = manifest(out)
    assert m.get("provider") == "prov-x", m
    assert m.get("agent_model") == "agent-m.1", m
    assert m.get("judge_model") == "judge-m_2", m
    # the retagged judge entries really were exported (judge_requests)
    exported = {r["request_hash"] for r in read_jsonl(out / "cache.jsonl")}
    assert set(m["judge_requests"].values()) <= exported


def test_judge_model_is_null_when_no_judge_entry_is_exported(
    source_db_url: str, tmp_path: Path
) -> None:
    sid, _ = build_suite(source_db_url, tmp_path, score=False)
    with connect(source_db_url) as c:
        retag(c, agent_hashes(c), "prov-x", "agent-m.1")
    out = tmp_path / "fx"
    ok(export(source_db_url, sid, out))
    m = manifest(out)
    assert m["judge_requests"] == {}
    assert "judge_model" in m and m["judge_model"] is None, m
    assert m.get("provider") == "prov-x" and m.get("agent_model") == "agent-m.1", m


# --- line 2 -----------------------------------------------------------------------


def _mixed_agent_model(c, a, j):
    retag(c, a[:1], None, "other-model")


def _mixed_agent_provider(c, a, j):
    retag(c, a[:1], "other-provider", None)


def _mixed_judge_model(c, a, j):
    retag(c, j[:1], None, "other-judge-model")


def _judge_provider_differs(c, a, j):
    retag(c, j, "other-provider", None)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_mixed_agent_model, id="agent-models-differ"),
        pytest.param(_mixed_agent_provider, id="agent-providers-differ"),
        pytest.param(_mixed_judge_model, id="judge-models-differ"),
        pytest.param(_judge_provider_differs, id="judge-provider-differs-from-agent"),
    ],
)
def test_export_refuses_mixed_providers_or_models_and_writes_nothing(
    mutate, source_db_url: str, tmp_path: Path
) -> None:
    sid, _ = build_suite(source_db_url, tmp_path)
    # Control: the untouched suite exports fine, so a refusal below is about the mix.
    ok(export(source_db_url, sid, tmp_path / "control"))
    with connect(source_db_url) as c:
        a, j = agent_hashes(c), judge_hashes(c)
        assert len(a) >= 2 and len(j) >= 2
        mutate(c, a, j)
    out = tmp_path / "fx"
    p = export(source_db_url, sid, out)
    assert p.returncode == 1, (p.stdout, p.stderr)
    assert p.stderr.strip(), "a clear message on stderr"
    assert "Traceback" not in p.stderr
    written = [n for n in FILES if (out / n).exists()]
    assert written == [], f"files written despite the refusal: {written}"


# --- line 3 -----------------------------------------------------------------------


def test_manifest_without_the_new_keys_still_imports_and_verifies(
    source_db_url: str, target_db_url: str, tmp_path: Path
) -> None:
    sid, _ = build_suite(source_db_url, tmp_path)
    out = tmp_path / "fx"
    ok(export(source_db_url, sid, out))
    m = manifest(out)
    assert m["format_version"] == 1
    assert all(k in m for k in KEYS), f"new keys missing from {sorted(m)}"
    old = tmp_path / "old"
    old.mkdir()
    for name in FILES:
        (old / name).write_bytes((out / name).read_bytes())
    for k in KEYS:
        m.pop(k)
    (old / "manifest.json").write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")
    ok(cli(target_db_url, "fixtures", "import", str(old)))
    ok(cli(target_db_url, "fixtures", "verify", str(old)))


def test_reexport_gives_identical_bytes_and_format_version_stays_1(
    source_db_url: str, tmp_path: Path
) -> None:
    sid, _ = build_suite(source_db_url, tmp_path)
    one, two = tmp_path / "one", tmp_path / "two"
    ok(export(source_db_url, sid, one))
    ok(export(source_db_url, sid, two))
    for name in FILES:
        assert (one / name).read_bytes() == (two / name).read_bytes(), name
    m = manifest(one)
    assert m["format_version"] == 1
    assert all(k in m for k in KEYS), sorted(m)


# --- line 4 -----------------------------------------------------------------------


def test_claude_code_fixture_imports_and_verifies_without_provider_setting(
    source_db_url: str, target_db_url: str, tmp_path: Path
) -> None:
    sid, _ = build_suite(source_db_url, tmp_path)
    out = tmp_path / "fx"
    ok(export(source_db_url, sid, out))
    rows = read_jsonl(out / "cache.jsonl")
    assert rows
    for r in rows:
        r["provider"] = "claude_code"
        r["model"] = "claude-sonnet-5-5"
    (out / "cache.jsonl").write_text(
        "".join(
            json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows
        )
    )
    m = manifest(out)
    if "provider" in m:  # keep the files self-consistent once the keys exist
        m.update(
            provider="claude_code",
            agent_model="claude-haiku-5-5",
            judge_model="claude-sonnet-5-5",
        )
        (out / "manifest.json").write_text(json.dumps(m, sort_keys=True, indent=2))
    # cli() sets no AGENT_REPLAY_PROVIDER (and no key, no CLI).
    ok(cli(target_db_url, "fixtures", "import", str(out)))
    ok(cli(target_db_url, "fixtures", "verify", str(out)))
    with connect(target_db_url) as c:
        n = c.execute(
            "SELECT count(*) FROM model_cache WHERE provider = 'claude_code'"
        ).fetchone()[0]
    assert n == len(rows)
