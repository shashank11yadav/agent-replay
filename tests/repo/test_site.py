"""The project page in site/: its data matches the recordings and every file it uses exists."""

from __future__ import annotations

import importlib.util
import json
import re

from support import ROOT
from support.workflow import load_yaml

SITE = ROOT / "site"
PAGES = ROOT / ".github" / "workflows" / "pages.yml"


def _builder():
    spec = importlib.util.spec_from_file_location(
        "site_data", ROOT / "scripts" / "site_data.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _data() -> dict:
    text = (SITE / "data.js").read_text(encoding="utf-8")
    return json.loads(text[text.index("=") + 1 :].rstrip().rstrip(";"))


def test_data_js_is_built_from_the_committed_fixtures() -> None:
    assert (SITE / "data.js").read_text(encoding="utf-8") == _builder().build(), (
        "site/data.js is stale: run `uv run python scripts/site_data.py`"
    )


def test_example_and_board_agree_with_the_regression_results() -> None:
    data = _data()
    assert data["task"]["id"] == "T11"
    assert all(c["passed"] for c in data["baseline"]["checks"])
    assert not all(c["passed"] for c in data["regressed"]["checks"])
    board = data["board"]
    assert len(board) == 30
    assert sum(t["baseline"] for t in board) == 22  # 73.3%
    assert sum(t["regressed"] for t in board) == 20  # 66.7%
    flipped = sorted(t["id"] for t in board if t["baseline"] and not t["regressed"])
    assert flipped == ["T11", "T15", "T19", "T20", "T23"]
    assert [g["status"] for g in data["gate"]] == ["FAIL", "PASS", "FAIL"]


def test_every_local_file_the_page_references_exists() -> None:
    html = (SITE / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)="([^"#:]+)"', html)
    assert refs, "no local references found"
    for ref in refs:
        if ref in ("./",):
            continue
        assert (SITE / ref).is_file(), ref


def test_page_credits_the_author_and_names_no_assistant() -> None:
    html = (SITE / "index.html").read_text(encoding="utf-8")
    assert 'href="https://shashankyadav.dev"' in html
    visible = re.sub(r"<[^>]+>", " ", html)
    assert not re.search(r"\b(claude|anthropic)\b", visible, re.IGNORECASE)


def test_pages_workflow_publishes_site_with_pinned_actions() -> None:
    wf = load_yaml(PAGES)
    job = wf["jobs"]["deploy"]
    assert wf["permissions"] == {"contents": "read"}
    assert job["permissions"] == {
        "contents": "read",
        "pages": "write",
        "id-token": "write",
    }
    steps = job["steps"]
    for step in steps:
        if "uses" in step:
            assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", step["uses"]), step["uses"]
    upload = next(
        s for s in steps if s["uses"].startswith("actions/upload-pages-artifact@")
    )
    assert upload["with"]["path"] == "site"
