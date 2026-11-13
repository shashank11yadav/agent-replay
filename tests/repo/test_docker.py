"""Dockerfile, .dockerignore, docker-compose.yml and .env.example.

Static checks only (no `docker compose up`). Compose structure is read from
`docker compose config --format json` (PyYAML is not a dependency); the raw
text is checked where interpolation would hide the thing under test.
The live `GET /` + `/api/health` check is done separately, against a running stack.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
from typing import Any

import pytest
from support import ROOT

DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
COMPOSE = ROOT / "docker-compose.yml"
ENV_EXAMPLE = ROOT / ".env.example"
KEY = "AGENT_REPLAY_GEMINI_API_KEY"

HAVE_DOCKER = shutil.which("docker") is not None
needs_docker = pytest.mark.skipif(not HAVE_DOCKER, reason="docker CLI not installed")


# ---------------------------------------------------------------- Dockerfile


def _instructions() -> list[tuple[str, str]]:
    assert DOCKERFILE.is_file(), f"{DOCKERFILE} does not exist"
    out: list[tuple[str, str]] = []
    buf = ""
    for raw in DOCKERFILE.read_text().splitlines():
        line = raw.strip()
        if not buf and (not line or line.startswith("#")):
            continue
        if buf and line.startswith("#"):
            continue
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        buf += line
        word, _, rest = buf.partition(" ")
        out.append((word.upper(), rest.strip()))
        buf = ""
    return out


def _global_args() -> dict[str, str]:
    """Defaults of the ARGs declared before the first FROM (usable in FROM lines)."""
    args: dict[str, str] = {}
    for op, rest in _instructions():
        if op == "FROM":
            break
        if op == "ARG" and "=" in rest:
            name, _, value = rest.partition("=")
            args[name.strip()] = value.strip().strip('"')
    return args


def _stages() -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    args = _global_args()
    for op, rest in _instructions():
        if op == "FROM":
            rest = re.sub(
                r"\$\{(\w+)\}", lambda m: args.get(m.group(1), m.group(0)), rest
            )
            toks = [t for t in rest.split() if not t.startswith("--")]
            alias = (
                toks[2].lower() if len(toks) >= 3 and toks[1].upper() == "AS" else None
            )
            stages.append({"image": toks[0], "alias": alias, "ins": []})
        elif stages:
            stages[-1]["ins"].append((op, rest))
    return stages


def _runs(stage: dict[str, Any]) -> list[str]:
    return [r for op, r in stage["ins"] if op == "RUN"]


def _copy_from(stage: dict[str, Any]) -> list[tuple[str, list[str], str]]:
    """(source, [paths], dest) for each COPY --from=..."""
    res = []
    for op, rest in stage["ins"]:
        if op != "COPY":
            continue
        m = re.search(r"--from=(\S+)", rest)
        if not m:
            continue
        args = [t for t in rest.split() if not t.startswith("--")]
        res.append((m.group(1), args[:-1], args[-1]))
    return res


def _final() -> dict[str, Any]:
    st = _stages()
    assert st, "no FROM in Dockerfile"
    return st[-1]


def test_dockerfile_three_stage_build() -> None:
    st = _stages()
    assert len(st) == 3, f"want exactly 3 stages, got {[s['image'] for s in st]}"
    node, builder, final = st
    assert node["image"] == "node:22.23.4-slim"
    runs = " && ".join(_runs(node))
    assert re.search(r"\bnpm ci\b", runs)
    assert re.search(r"\bnpm run build\b", runs)
    assert runs.index("npm ci") < runs.index("npm run build")

    assert re.fullmatch(r"python:3\.12\.\d+-slim", builder["image"]), builder["image"]
    uv_copies = [
        c for c in _copy_from(builder) if c[0] == "ghcr.io/astral-sh/uv:0.13.6"
    ]
    assert uv_copies, "builder must COPY --from=ghcr.io/astral-sh/uv:0.13.6"
    assert any("/uv" in p for _, paths, _ in uv_copies for p in paths)
    brun = " && ".join(_runs(builder))
    first = "uv sync --locked --no-install-project --no-editable"
    second = "uv sync --locked --no-editable"
    assert first in brun, "missing dependency-layer uv sync"
    assert second in brun.replace(first, "", 1), "missing project uv sync"
    assert brun.index(first) < brun.rindex(second)

    # Same interpreter at build and run time, or the venv may not load.
    assert final["image"] == builder["image"]
    sources = {src for src, _, _ in _copy_from(final)}
    assert builder["alias"] and builder["alias"] in sources | {"0", "1"}
    venv = [
        c
        for c in _copy_from(final)
        if any(p.rstrip("/").endswith("/.venv") for p in c[1])
    ]
    assert venv and venv[0][1][0].rstrip("/") == "/app/.venv"
    dist = [c for c in _copy_from(final) if any("viewer/dist" in p for p in c[1])]
    assert dist, "final stage must copy viewer/dist from the node stage"
    assert dist[0][0] in (node["alias"], "0")


def test_no_latest_or_untagged_images() -> None:
    images: list[str] = []
    aliases = set()
    for s in _stages():
        if s["image"].lower() not in aliases:
            images.append(s["image"])
        if s["alias"]:
            aliases.add(s["alias"])
        images += [
            c[0] for c in _copy_from(s) if not c[0].isdigit() and c[0] not in aliases
        ]
    assert COMPOSE.is_file(), f"{COMPOSE} does not exist"
    images += re.findall(
        r"^\s*image:\s*['\"]?([^\s'\"#]+)", COMPOSE.read_text(), re.MULTILINE
    )
    assert len(images) >= 5
    for img in images:
        name = img.rsplit("/", 1)[-1]
        assert "@sha256:" in img or ":" in name, f"untagged image {img}"
        assert not img.endswith(":latest"), f"latest image {img}"


def test_final_stage_non_root_and_uvicorn_cmd_from_venv() -> None:
    final = _final()
    users = [r for op, r in final["ins"] if op == "USER"]
    assert users, "final stage has no USER"
    assert users[-1].split(":")[0].strip().lower() not in ("root", "0")
    cmds = [r for op, r in final["ins"] if op in ("CMD", "ENTRYPOINT")]
    assert cmds, "final stage has no CMD"
    cmd = cmds[-1]
    toks = json.loads(cmd) if cmd.startswith("[") else cmd.split()
    assert toks[-5:] == [
        "agent_replay.api.asgi:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ], toks
    exe = toks[0]
    path_envs = " ".join(r for op, r in final["ins"] if op == "ENV" and "PATH" in r)
    assert exe == "/app/.venv/bin/uvicorn" or (
        exe == "uvicorn" and "/app/.venv/bin" in path_envs
    ), f"uvicorn not taken from the venv: {toks}"
    if exe == "uvicorn":
        assert "/app/.venv/bin" in path_envs


def test_final_stage_viewer_dist_env_points_at_copied_dist() -> None:
    final = _final()
    env = {}
    for op, rest in final["ins"]:
        if op == "ENV":
            for m in re.finditer(r"(\w+)=(\"[^\"]*\"|\S+)", rest):
                env[m.group(1)] = m.group(2).strip('"')
            if "=" not in rest.split()[0]:
                k, _, v = rest.partition(" ")
                env[k] = v.strip().strip('"')
    val = env.get("AGENT_REPLAY_VIEWER_DIST")
    assert val, f"AGENT_REPLAY_VIEWER_DIST not set in final stage: {env}"
    workdir = "/"
    dests = []
    for op, rest in final["ins"]:
        if op == "WORKDIR":
            workdir = rest if rest.startswith("/") else f"{workdir.rstrip('/')}/{rest}"
        if op == "COPY" and "viewer/dist" in rest:
            dest = rest.split()[-1]
            if not dest.startswith("/"):
                dest = f"{workdir.rstrip('/')}/{dest}"
            dests.append(dest.rstrip("/"))
    assert dests, "no COPY of viewer/dist in final stage"
    assert val.rstrip("/") in dests, f"{val} is not the copy destination {dests}"
    assert val.startswith("/")


# ------------------------------------------------------------- .dockerignore


def _covered(path: str, patterns: list[str]) -> bool:
    hit = False
    for p in patterns:
        neg = p.startswith("!")
        p = p.lstrip("!").lstrip("/").rstrip("/")
        cands = [p] + ([p[3:]] if p.startswith("**/") else [])
        if any(fnmatch.fnmatchcase(path, c) for c in cands):
            hit = not neg
    return hit


def test_dockerignore_excludes_secrets_and_build_output() -> None:
    assert DOCKERIGNORE.is_file(), f"{DOCKERIGNORE} does not exist"
    pats = [
        ln.strip()
        for ln in DOCKERIGNORE.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    for path in (
        ".env",
        ".env.local",
        ".env.production",
        ".git",
        "node_modules",
        "viewer/node_modules",
        "viewer/dist",
    ):
        assert _covered(path, pats), f".dockerignore does not exclude {path}: {pats}"
    assert not any(p.startswith("!") and "env" in p for p in pats)


# ------------------------------------------------------------------- compose


def _compose_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != KEY}
    return env


def _config() -> dict[str, Any]:
    assert COMPOSE.is_file()
    proc = subprocess.run(
        ["docker", "compose", "config", "--format", "json"],
        cwd=ROOT,
        env=_compose_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _cmd_text(svc: dict[str, Any]) -> str:
    parts: list[str] = []
    for k in ("entrypoint", "command"):
        v = svc.get(k)
        if isinstance(v, list):
            parts += [str(x) for x in v]
        elif v:
            parts.append(str(v))
    return " ".join(parts)


def test_compose_no_version_key_and_exact_services() -> None:
    assert COMPOSE.is_file()
    assert not re.search(r"^version\s*:", COMPOSE.read_text(), re.MULTILINE)
    if not HAVE_DOCKER:
        pytest.skip("docker CLI not installed")
    cfg = _config()
    assert "version" not in cfg
    assert set(cfg["services"]) == {"db", "migrate", "seed", "api"}


@needs_docker
def test_compose_depends_on_chain() -> None:
    svcs = _config()["services"]

    def dep(a: str, b: str) -> str | None:
        return (svcs[a].get("depends_on") or {}).get(b, {}).get("condition")

    assert dep("migrate", "db") == "service_healthy"
    assert dep("seed", "migrate") == "service_completed_successfully"
    assert dep("api", "seed") == "service_completed_successfully"
    # db has a healthcheck for service_healthy to mean something
    assert svcs["db"].get("healthcheck", {}).get("test")


@needs_docker
def test_seed_imports_baseline_and_regression_only_if_dirs_exist() -> None:
    text = _cmd_text(_config()["services"]["seed"])
    for d in ("evals/baseline", "evals/regression"):
        assert f"agent-replay fixtures import {d}" in text, text
        assert re.search(rf"(\[|test)\s+-d\s+{re.escape(d)}\b", text), (
            f"no directory-exists guard for {d}: {text}"
        )
        # the guard must precede its import
        assert re.search(
            rf"-d\s+{re.escape(d)}\b.*fixtures import {re.escape(d)}", text
        )


@needs_docker
def test_api_publishes_8000_and_serves_default_uvicorn_command() -> None:
    api = _config()["services"]["api"]
    ports = api.get("ports") or []
    assert any(
        int(p.get("target", 0)) == 8000 and int(p.get("published", 0)) == 8000
        for p in ports
    ), ports
    assert "build" in api, "api must be built from the Dockerfile"
    # no separate viewer service: the image's default command serves everything
    cmd = _cmd_text(api)
    if cmd:
        assert "agent_replay.api.asgi:app" in cmd and "8000" in cmd


@needs_docker
def test_compose_config_exits_zero_with_api_key_unset() -> None:
    env = _compose_env()
    assert KEY not in env
    proc = subprocess.run(
        ["docker", "compose", "config"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr


def test_api_service_gets_no_model_key_and_answers_local_hosts() -> None:
    assert COMPOSE.is_file()
    text = COMPOSE.read_text()
    assert KEY not in text
    assert re.search(
        r"^\s*AGENT_REPLAY_ALLOWED_HOSTS:\s*localhost,127\.0\.0\.1\s*$",
        text,
        re.MULTILINE,
    )


def test_db_volume_mounted_at_var_lib_postgresql() -> None:
    assert COMPOSE.is_file()
    text = COMPOSE.read_text()
    assert re.search(r"^\s*-\s*\S+:/var/lib/postgresql\s*$", text, re.MULTILINE)
    assert "/var/lib/postgresql/data" not in text


# ------------------------------------------------------------------- secrets

_SECRET_SHAPES = re.compile(r"AIza[0-9A-Za-z_\-]{20,}|ghp_\w{20,}|sk-[A-Za-z0-9]{20,}")


def test_no_secret_values_in_compose_or_env_example() -> None:
    assert COMPOSE.is_file() and ENV_EXAMPLE.is_file(), (
        "compose or .env.example missing"
    )
    for f in (COMPOSE, ENV_EXAMPLE):
        assert not _SECRET_SHAPES.search(f.read_text()), (
            f"secret-shaped value in {f.name}"
        )
    for ln in COMPOSE.read_text().splitlines():
        m = re.match(r"^\s*(\w*(?:KEY|TOKEN|SECRET)\w*)\s*[:=]\s*(.*?)\s*$", ln)
        if m:
            val = m.group(2).strip("'\"")
            assert val == "" or re.fullmatch(r"\$\{\w+:?-\}", val), (
                f"{m.group(1)} has a literal value in compose: {val!r}"
            )


def test_env_example_lists_names_with_empty_values() -> None:
    assert ENV_EXAMPLE.is_file(), f"{ENV_EXAMPLE} does not exist"
    seen: dict[str, str] = {}
    for ln in ENV_EXAMPLE.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        m = re.fullmatch(r"(?:export\s+)?([A-Z][A-Z0-9_]*)=(.*)", s)
        assert m, f"not NAME=value: {ln!r}"
        seen[m.group(1)] = m.group(2).strip()
    assert KEY in seen
    assert all(v in ("", '""', "''") for v in seen.values()), seen
