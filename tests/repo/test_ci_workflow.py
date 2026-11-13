"""Shape of .github/workflows/ci.yml: triggers, permissions and the python and viewer jobs.

The workflow is parsed, not grepped, so reordering keys or comments does not matter.
"""

from __future__ import annotations

import re
from typing import Any

from support.workflow import CI, WF, load_yaml


def pinned(action: str) -> str:
    """The one `action@<commit sha>` ci.yml uses. Which commit is Dependabot's to bump."""
    refs = set(re.findall(rf"uses: ({re.escape(action)}@\S+)", CI.read_text()))
    assert len(refs) == 1, f"{action} must be used at exactly one ref, got {refs}"
    ref = refs.pop()
    assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", ref), (
        f"{ref} is not pinned to a commit"
    )
    return ref


CHECKOUT = pinned("actions/checkout")
SETUP_PYTHON = pinned("actions/setup-python")
SETUP_UV = pinned("astral-sh/setup-uv")
SETUP_NODE = pinned("actions/setup-node")


def load_ci() -> dict[str, Any]:
    return load_yaml(CI)


def _triggers(data: dict[str, Any]) -> set[str]:
    on = data.get("on")
    if isinstance(on, str):
        return {on}
    if isinstance(on, list):
        return set(on)
    assert isinstance(on, dict), f"bad `on`: {on!r}"
    return set(on)


def python_job(data: dict[str, Any]) -> dict[str, Any]:
    jobs = data.get("jobs")
    assert isinstance(jobs, dict) and jobs, "no jobs"
    job = jobs.get("python")
    assert isinstance(job, dict), f"no `python` job; jobs: {list(jobs)}"
    return job


def steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    s = job.get("steps")
    assert isinstance(s, list) and s
    return s


def test_triggers_push_to_main_and_pull_request_never_pull_request_target() -> None:
    data = load_ci()
    trig = _triggers(data)
    assert {"push", "pull_request"} <= trig
    assert data["on"]["push"] == {"branches": ["main"]}
    assert "pull_request_target" not in trig
    assert "pull_request_target" not in CI.read_text()


def test_every_action_is_pinned_by_sha_like_the_regression_gate() -> None:
    gate_pins = set(re.findall(r"uses: (\S+@[0-9a-f]{40})", WF.read_text()))
    for line in CI.read_text().splitlines():
        m = re.search(r"uses: (\S+)( # v\d+\.\d+\.\d+)?$", line.strip())
        if not m:
            continue
        assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", m.group(1)), line
        assert m.group(2), f"no version comment: {line}"
        if not m.group(1).startswith("actions/setup-node@"):
            assert m.group(1) in gate_pins, line


def test_top_level_permissions_contents_read_only() -> None:
    data = load_ci()
    assert data.get("permissions") == {"contents": "read"}


def test_python_job_action_versions_and_python_312() -> None:
    job = python_job(load_ci())
    uses = [s["uses"] for s in steps(job) if "uses" in s]
    assert CHECKOUT in uses
    assert SETUP_UV in uses
    assert SETUP_PYTHON in uses
    sp = next(s for s in steps(job) if s.get("uses") == SETUP_PYTHON)
    assert str(sp.get("with", {}).get("python-version")) == "3.12"


def test_python_job_postgres_service_and_test_db_url() -> None:
    job = python_job(load_ci())
    services = job.get("services")
    assert isinstance(services, dict)
    pg = next(
        (v for v in services.values() if str(v.get("image")) == "postgres:18.6-alpine"),
        None,
    )
    assert pg is not None, f"no postgres:18.6-alpine service: {services}"
    assert pg.get("env", {}).get("POSTGRES_PASSWORD")
    opts = str(pg.get("options"))
    assert "pg_isready" in opts
    assert "--health-cmd" in opts
    assert [str(p) for p in pg.get("ports", [])] == ["5432:5432"]

    url = None
    holders = [job.get("env") or {}] + [s.get("env") or {} for s in steps(job)]
    for env in holders:
        if "AGENT_REPLAY_TEST_DATABASE_URL" in env:
            url = str(env["AGENT_REPLAY_TEST_DATABASE_URL"])
    assert url, "AGENT_REPLAY_TEST_DATABASE_URL not set"
    assert url.startswith("postgresql://")
    assert "5432" in url
    assert "localhost" in url or "127.0.0.1" in url


def test_python_job_runs_exact_check_commands_as_separate_steps() -> None:
    job = python_job(load_ci())
    runs = [str(s["run"]).strip() for s in steps(job) if "run" in s]
    for cmd in (
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run mypy src",
        "uv run pytest -q",
    ):
        assert runs.count(cmd) == 1, f"{cmd!r} must be exactly one step; runs={runs}"
    order = [
        runs.index(c)
        for c in (
            "uv run ruff check .",
            "uv run ruff format --check .",
            "uv run mypy src",
            "uv run pytest -q",
        )
    ]
    assert order == sorted(order)


def test_no_secrets_and_no_gemini_key() -> None:
    load_ci()
    text = CI.read_text()
    assert not re.search(r"secrets\s*[.\[]", text)
    assert "secrets" not in text.lower()
    assert "AGENT_REPLAY_GEMINI_API_KEY" not in text
    assert "GEMINI" not in text.upper()


def _job_text(text: str, name: str) -> str:
    """Raw text of a top-level job (2-space indented key under `jobs:`)."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln == f"  {name}:"), None)
    assert start is not None, f"no `{name}` job in ci.yml"
    end = len(lines)
    for i in range(start + 1, len(lines)):
        ln = lines[i]
        if ln.strip() and not ln.startswith("   ") and not ln.lstrip().startswith("#"):
            end = i
            break
    return "\n".join(lines[start:end]).rstrip() + "\n"


def viewer_job() -> dict[str, Any]:
    data = load_ci()
    jobs = data.get("jobs")
    assert isinstance(jobs, dict)
    job = jobs.get("viewer")
    assert isinstance(job, dict), f"no `viewer` job; jobs: {list(jobs)}"
    return job


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    s = job.get("steps")
    assert isinstance(s, list) and s
    return s


def test_viewer_job_uses_pinned_checkout_and_setup_node_22_or_later() -> None:
    job = viewer_job()
    uses = [s["uses"] for s in _steps(job) if "uses" in s]
    assert CHECKOUT in uses
    assert SETUP_NODE in uses
    sn = next(s for s in _steps(job) if s.get("uses") == SETUP_NODE)
    version = str(sn.get("with", {}).get("node-version"))
    m = re.fullmatch(r"(\d+)(\.\d+)*(\.x)?", version)
    assert m and int(m.group(1)) >= 22, (
        f"viewer/package.json needs node >= 22, got {version}"
    )
    # setup-node must come before any run step
    idx = [i for i, s in enumerate(_steps(job)) if s.get("uses") == SETUP_NODE]
    first_run = next(i for i, s in enumerate(_steps(job)) if "run" in s)
    assert idx[0] < first_run


def test_viewer_job_runs_ci_typecheck_test_build_as_separate_ordered_steps() -> None:
    job = viewer_job()
    runs = [str(s["run"]).strip() for s in _steps(job) if "run" in s]
    cmds = [
        "npm --prefix viewer ci",
        "npm --prefix viewer run typecheck",
        "npm --prefix viewer test",
        "npm --prefix viewer run build",
    ]
    for c in cmds:
        assert runs.count(c) == 1, f"{c!r} must be exactly one step; runs={runs}"
    order = [runs.index(c) for c in cmds]
    assert order == sorted(order)


def test_top_level_permissions_contents_read_only_and_viewer_has_no_secrets() -> None:
    data = load_ci()
    assert data.get("permissions") == {"contents": "read"}
    job = viewer_job()
    perms = job.get("permissions")
    assert perms in (None, {"contents": "read"}), perms
    text = CI.read_text()
    assert "secrets" not in _job_text(text, "viewer").lower()
    assert "secrets" not in text.lower()
    assert "pull_request_target" not in text


PYTHON_STEPS = [
    {
        "name": "Check out",
        "uses": CHECKOUT,
        "with": {"persist-credentials": False},
    },
    {
        "name": "Set up Python",
        "uses": SETUP_PYTHON,
        "with": {"python-version": "3.12"},
    },
    {
        "name": "Set up uv",
        "uses": SETUP_UV,
        "with": {"enable-cache": True},
    },
    {"name": "Lint", "run": "uv run ruff check ."},
    {"name": "Format check", "run": "uv run ruff format --check ."},
    {"name": "Typecheck", "run": "uv run mypy src"},
    {"name": "Tests", "run": "uv run pytest -q"},
]


def test_python_job_steps_and_service_are_pinned() -> None:
    job = python_job(load_ci())
    assert job.get("runs-on") == "ubuntu-latest"
    assert job.get("timeout-minutes") == 15
    assert job.get("env") == {
        "UV_LOCKED": "1",
        "AGENT_REPLAY_TEST_DATABASE_URL": (
            "postgresql://postgres:postgres@localhost:5432/postgres"
        ),
    }
    pg = job["services"]["postgres"]
    assert pg["image"] == "postgres:18.6-alpine"
    assert pg["env"] == {"POSTGRES_PASSWORD": "postgres"}
    assert [str(p) for p in pg["ports"]] == ["5432:5432"]
    assert pg["options"] == (
        "--health-cmd pg_isready --health-interval 5s "
        "--health-timeout 5s --health-retries 10"
    )
    assert job["steps"] == PYTHON_STEPS


def test_docker_job_builds_the_image() -> None:
    jobs = load_ci()["jobs"]
    runs = [str(s.get("run") or "").strip() for s in jobs["docker"]["steps"]]
    assert "docker build ." in runs
    assert jobs["docker"].get("permissions") in (None, {"contents": "read"})
