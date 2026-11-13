"""Read-only FastAPI app: routes, errors, CORS and serving the viewer build.

List endpoints may return a bare JSON array or an object wrapping it under `items` or
`runs`; the tests accept both.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

ORIGIN = "https://viewer.example.test"
OTHER = "https://evil.example.test"
KEY = "AIzaDUMMYkey-api-5d2a9c"


def _settings(db_url: str, **kw: Any) -> Any:
    from agent_replay.config import Settings

    kw.setdefault("cors_origins", [ORIGIN])
    kw.setdefault("allowed_hosts", ["testserver"])
    return Settings(database_url=db_url, gemini_api_key=KEY, **kw)


def _span(
    seq: int, kind: str, name: str, payload: dict[str, Any], tokens: int = 5
) -> Any:
    return {
        "seq": seq,
        "kind": kind,
        "name": name,
        "payload": payload,
        "input_tokens": tokens,
        "output_tokens": tokens,
        "latency_ms": 10 * (seq + 1),
    }


def _mk_run(conn: Any, suite: Any, task: str, minute: int, spans: list[Any]) -> str:
    from agent_replay.db import runs_repo

    run = {
        "suite_run_id": suite,
        "task_id": task,
        "variant": "v1",
        "mode": "record",
        "status": "ok",
        "started_at": f"2026-01-01T00:{minute:02d}:00+00:00",
        "input_tokens": 10 + minute,
        "output_tokens": 5,
        "latency_ms": 100 + minute,
        "model_calls": 1,
        "tool_calls": 1,
    }
    return str(runs_repo.save_run(conn, run, spans))


@pytest.fixture
def seeded(conn: Any) -> dict[str, Any]:
    from agent_replay.db import runs_repo

    s1 = str(runs_repo.create_suite_run(conn, label="one", variant="v1", task_count=5))
    s2 = str(runs_repo.create_suite_run(conn, label="two", variant="v1", task_count=2))
    base = [
        _span(0, "model", "chat", {"text": "hello"}),
        _span(1, "tool", "get_order", {"args": {"id": "A1"}, "result": "shipped"}),
        _span(2, "model", "chat", {"text": "done"}),
    ]
    s1_runs = [_mk_run(conn, s1, f"t{i}", i, base) for i in range(5)]
    s2_runs = [_mk_run(conn, s2, f"u{i}", 10 + i, base) for i in range(2)]
    # A run that differs from the base first at seq 1.
    diverging = [
        base[0],
        _span(1, "tool", "get_order", {"args": {"id": "A2"}, "result": "missing"}),
        base[2],
    ]
    div = _mk_run(conn, s2, "div", 20, diverging)
    runs_repo.upsert_score(conn, s1_runs[0], "contains", True, {"why": "ok"})
    return {"s1": s1, "s2": s2, "s1_runs": s1_runs, "s2_runs": s2_runs, "div": div}


@pytest.fixture
def client(db_url: str, seeded: dict[str, Any]) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    from agent_replay.api import create_app

    with TestClient(create_app(_settings(db_url))) as c:
        yield c


def _items(body: Any) -> list[dict[str, Any]]:
    if isinstance(body, list):
        return body
    assert isinstance(body, dict), body
    for k in ("items", "runs"):
        if k in body:
            return body[k]
    raise AssertionError(f"no run list in {body!r}")


def _ids(client: Any, **params: Any) -> list[str]:
    r = client.get("/api/runs", params=params)
    assert r.status_code == 200, r.text
    return [str(x["id"]) for x in _items(r.json())]


# --- health / suites ---------------------------------------------------------------------


def test_health_ok(client: Any) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_suite_runs_newest_first(client: Any, seeded: dict[str, Any]) -> None:
    r = client.get("/api/suite-runs")
    assert r.status_code == 200
    rows = _items(r.json())
    ids = [str(x["id"]) for x in rows]
    assert ids == [seeded["s2"], seeded["s1"]]


def test_suite_summary_returns_metrics(client: Any, seeded: dict[str, Any]) -> None:
    r = client.get(f"/api/suite-runs/{seeded['s1']}/summary")
    assert r.status_code == 200
    assert r.json()["task_count"] == 5


# --- GET /api/runs: pagination and suite filter ------------------------------------------


def test_runs_lists_all_and_respects_limit(client: Any, seeded: dict[str, Any]) -> None:
    allr = _ids(client, limit=200)
    assert len(allr) == 8
    assert len(_ids(client, suite_run_id=seeded["s1"], limit=2)) == 2


def test_runs_pagination_no_overlap_no_gaps(
    client: Any, seeded: dict[str, Any]
) -> None:
    full = _ids(client, suite_run_id=seeded["s1"], limit=200)
    assert sorted(full) == sorted(seeded["s1_runs"])
    paged: list[str] = []
    for offset in (0, 2, 4, 6):
        paged += _ids(client, suite_run_id=seeded["s1"], limit=2, offset=offset)
    assert paged == full
    assert len(set(paged)) == len(paged) == 5
    assert _ids(client, suite_run_id=seeded["s1"], limit=2, offset=5) == []


def test_runs_filter_by_suite(client: Any, seeded: dict[str, Any]) -> None:
    got = _ids(client, suite_run_id=seeded["s2"], limit=200)
    assert sorted(got) == sorted([*seeded["s2_runs"], seeded["div"]])


@pytest.mark.parametrize(
    "params",
    [{"limit": 0}, {"limit": 201}, {"offset": -1}, {"suite_run_id": "not-a-uuid"}],
)
def test_runs_bad_query_is_422(client: Any, params: dict[str, Any]) -> None:
    assert client.get("/api/runs", params=params).status_code == 422


# --- GET /api/runs/{id} ------------------------------------------------------------------


def test_run_detail_has_spans_in_seq_order_and_scores(
    client: Any, seeded: dict[str, Any]
) -> None:
    rid = seeded["s1_runs"][0]
    r = client.get(f"/api/runs/{rid}")
    assert r.status_code == 200
    body = r.json()
    assert str(body["id"]) == rid
    assert body["task_id"] == "t0"
    assert [s["seq"] for s in body["spans"]] == [0, 1, 2]
    assert body["spans"][1]["name"] == "get_order"
    assert [s["check_name"] for s in body["scores"]] == ["contains"]
    assert body["scores"][0]["passed"] is True


def test_unknown_run_is_404_json(client: Any) -> None:
    r = client.get(f"/api/runs/{uuid.uuid4()}")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")
    assert isinstance(r.json(), dict) and r.json()


def test_malformed_run_id_is_422(client: Any) -> None:
    assert client.get("/api/runs/not-a-uuid").status_code == 422


# --- GET /api/diff -----------------------------------------------------------------------


def test_diff_identical_runs_has_no_divergence(
    client: Any, seeded: dict[str, Any]
) -> None:
    a, b = seeded["s1_runs"][0], seeded["s1_runs"][1]
    r = client.get("/api/diff", params={"left": a, "right": b})
    assert r.status_code == 200
    assert r.json()["first_divergence"] is None


def test_diff_reports_first_divergence_and_is_json_with_decimals_as_strings(
    client: Any, seeded: dict[str, Any]
) -> None:
    a, b = seeded["s1_runs"][0], seeded["div"]
    r = client.get("/api/diff", params={"left": a, "right": b})
    assert r.status_code == 200
    body = r.json()
    assert body["first_divergence"] == 1

    def no_decimal_like_floats(node: Any) -> None:
        # ADR: Decimal values are serialised as strings, never as JSON numbers with loss.
        if isinstance(node, dict):
            for v in node.values():
                no_decimal_like_floats(v)
        elif isinstance(node, list):
            for v in node:
                no_decimal_like_floats(v)

    no_decimal_like_floats(body)


def test_diff_serialises_decimal_as_string(
    db_url: str, seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force a Decimal into the diff model; the HTTP layer must emit it as a string."""
    from decimal import Decimal

    from fastapi.testclient import TestClient

    from agent_replay.api import create_app, routes

    real = routes.diff_runs

    def fake(left: Any, right: Any) -> Any:
        result = real(left, right)
        data = result.model_dump()
        data["cost_delta_usd"] = Decimal("0.0001230")
        return type("D", (), {"model_dump": lambda self, **kw: data})()

    monkeypatch.setattr(routes, "diff_runs", fake)
    with TestClient(create_app(_settings(db_url))) as c:
        r = c.get(
            "/api/diff",
            params={"left": seeded["s1_runs"][0], "right": seeded["s1_runs"][1]},
        )
    assert r.status_code == 200
    assert r.json()["cost_delta_usd"] == "0.0001230"


def test_diff_unknown_id_404_and_malformed_422(
    client: Any, seeded: dict[str, Any]
) -> None:
    a = seeded["s1_runs"][0]
    r = client.get("/api/diff", params={"left": a, "right": str(uuid.uuid4())})
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")
    assert (
        client.get("/api/diff", params={"left": a, "right": "nope"}).status_code == 422
    )
    assert (
        client.get("/api/diff", params={"left": "nope", "right": a}).status_code == 422
    )


# --- read-only ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_write_methods_return_405(
    client: Any, seeded: dict[str, Any], method: str
) -> None:
    rid = seeded["s1_runs"][0]
    for path in ("/api/runs", f"/api/runs/{rid}", "/api/suite-runs", "/api/health"):
        r = client.request(method, path)
        assert r.status_code == 405, (method, path, r.status_code)
    assert client.get(f"/api/runs/{rid}").status_code == 200  # nothing deleted


def test_no_write_routes_registered(db_url: str) -> None:
    from agent_replay.api import create_app

    app = create_app(_settings(db_url))
    for route in app.routes:
        methods = getattr(route, "methods", None) or set()
        path = getattr(route, "path", "")
        if path.startswith("/api"):
            assert methods <= {"GET", "HEAD", "OPTIONS"}, (path, methods)


# --- CORS --------------------------------------------------------------------------------


def test_cors_allows_configured_origin(client: Any) -> None:
    r = client.get("/api/health", headers={"Origin": ORIGIN})
    assert r.headers.get("access-control-allow-origin") == ORIGIN
    assert r.headers.get("access-control-allow-credentials") is None
    pre = client.options(
        "/api/runs",
        headers={"Origin": ORIGIN, "Access-Control-Request-Method": "GET"},
    )
    assert pre.status_code == 200
    assert pre.headers.get("access-control-allow-origin") == ORIGIN


def test_cors_rejects_other_origin(client: Any) -> None:
    r = client.get("/api/health", headers={"Origin": OTHER})
    assert "access-control-allow-origin" not in r.headers
    pre = client.options(
        "/api/runs",
        headers={"Origin": OTHER, "Access-Control-Request-Method": "GET"},
    )
    assert pre.headers.get("access-control-allow-origin") not in (OTHER, "*")


def test_cors_never_wildcard_and_only_get_allowed(client: Any) -> None:
    pre = client.options(
        "/api/runs",
        headers={"Origin": ORIGIN, "Access-Control-Request-Method": "DELETE"},
    )
    assert pre.headers.get("access-control-allow-origin") != "*"
    assert pre.status_code == 400  # DELETE is not an allowed CORS method


def test_cors_unconfigured_allows_nobody(db_url: str, seeded: dict[str, Any]) -> None:
    from fastapi.testclient import TestClient

    from agent_replay.api import create_app

    with TestClient(create_app(_settings(db_url, cors_origins=[]))) as c:
        r = c.get("/api/health", headers={"Origin": ORIGIN})
    assert "access-control-allow-origin" not in r.headers


# --- viewer mount, lifespan, secrets, asgi -------------------------------------------------


def test_viewer_dist_serves_index_and_api_still_works(
    db_url: str, tmp_path: Path
) -> None:
    from fastapi.testclient import TestClient

    from agent_replay.api import create_app

    (tmp_path / "index.html").write_text("<html>VIEWER-BUILD</html>")
    with TestClient(create_app(_settings(db_url, viewer_dist=tmp_path))) as c:
        root = c.get("/")
        assert root.status_code == 200 and "VIEWER-BUILD" in root.text
        assert c.get("/api/health").json() == {"status": "ok"}


def test_no_viewer_dist_root_is_404(client: Any) -> None:
    assert client.get("/").status_code == 404
    assert client.get("/api/health").status_code == 200


def test_asgi_module_exposes_app(monkeypatch: pytest.MonkeyPatch, db_url: str) -> None:
    import importlib

    monkeypatch.setenv("AGENT_REPLAY_DATABASE_URL", db_url)
    mod = importlib.import_module("agent_replay.api.asgi")
    assert hasattr(mod.app, "routes")


def test_error_bodies_do_not_leak_secrets(client: Any, db_url: str) -> None:
    pw = urlsplit(db_url).password or "postgres"
    for path in (
        f"/api/runs/{uuid.uuid4()}",
        "/api/runs/not-a-uuid",
        "/api/runs?limit=0",
        "/api/diff?left=x&right=y",
    ):
        text = client.get(path).text
        assert KEY not in text and db_url not in text
        assert f":{pw}@" not in text


def test_a_request_for_another_host_is_refused(db_url: str) -> None:
    from fastapi.testclient import TestClient

    from agent_replay.api import create_app

    app = create_app(_settings(db_url, allowed_hosts=["localhost", "127.0.0.1"]))
    with TestClient(app, base_url="http://localhost") as c:
        assert c.get("/api/health").status_code == 200
        rebound = c.get("/api/health", headers={"Host": "attacker.example"})
        assert rebound.status_code == 400


def test_responses_carry_the_security_headers(db_url: str, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from agent_replay.api import create_app

    (tmp_path / "index.html").write_text("<!doctype html><title>v</title>")
    with TestClient(create_app(_settings(db_url, viewer_dist=tmp_path))) as c:
        for path in ("/api/health", "/", "/api/runs/not-a-uuid"):
            headers = c.get(path).headers
            assert headers["x-content-type-options"] == "nosniff"
            assert headers["referrer-policy"] == "no-referrer"
            csp = headers["content-security-policy"]
            assert "default-src 'self'" in csp
            assert f"connect-src 'self' {ORIGIN}" in csp
            assert "frame-ancestors 'none'" in csp
            assert "script-src" not in csp  # scripts fall back to 'self'


def test_api_connections_have_a_statement_timeout(db_url: str) -> None:
    from agent_replay.db.pool import get_pool

    with get_pool(_settings(db_url)) as pool, pool.connection() as conn:
        row = conn.execute("SHOW statement_timeout").fetchone()
    assert row == {"statement_timeout": "5s"}


def test_runs_list_carries_the_task_pass_verdict(
    client: Any, conn: Any, seeded: dict[str, Any]
) -> None:
    from agent_replay.db import runs_repo

    passed, failed, checks_only = seeded["s1_runs"][1:4]
    for rid, judge in ((passed, True), (failed, False), (checks_only, None)):
        runs_repo.upsert_score(conn, rid, "contains", True, {})
        if judge is not None:
            runs_repo.upsert_score(conn, rid, "judge", judge, {}, kind="judge")
    r = client.get("/api/runs", params={"suite_run_id": seeded["s1"]})
    assert r.status_code == 200, r.text
    verdicts = {x["id"]: x["passed"] for x in _items(r.json())}
    assert verdicts[passed] is True
    assert verdicts[failed] is False
    assert verdicts[checks_only] is None  # no judge row: not fully scored
    assert verdicts[seeded["s1_runs"][4]] is None  # no scores at all
    assert {"task_id", "status", "input_tokens"} <= set(_items(r.json())[0])
