# ADR-001: Stack and project layout

- Date: 2026-11-02
- Status: accepted

## Context
The project needs a Python backend, Postgres, a small React/TypeScript viewer, GitHub Actions
and a one-command local run with docker compose. Settings come from the environment through one
config module. pydantic-settings 2.16.2 and google-genai 2.34.0 need Python 3.11 or newer, and
Vitest 5 needs Node ^22.12.

## Decision
- Python 3.12, uv, `pyproject.toml` only, one package `src/agent_replay/`.
- CLI on stdlib `argparse`, one console script `agent-replay`. Each command group lives in its
  own `cli/*_cmd.py` module that exposes `add_parser()` (or `add_parsers()`) and `run()`.
- Settings: pydantic-settings `BaseSettings` with `env_prefix='AGENT_REPLAY_'`,
  `extra='forbid'` and no `env_file`. Tests then cannot pick up a developer's real key from a
  `.env` file; variables are exported in the shell or passed by compose.
- Viewer: Vite `react-ts` template in `viewer/`, Vitest, Testing Library and jsdom, Node 22.
- Checks: `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy src`, `npm --prefix viewer test`, `npm --prefix viewer run typecheck`.

## Alternatives considered
- Typer or Click: another dependency, and argparse is enough for about ten subcommands.
- A monorepo tool for the viewer: unnecessary for one small app.
- Reading `.env` in settings: convenient, but test isolation would then depend on precedence
  rules between the file and the environment.

## Consequences
- Adding a command touches `cli/__init__.py` by two lines (parser and dispatch).
- Dev tools are pinned: pytest 9.2.0, ruff 0.17.3, mypy 2.4.1. httpx 0.28.1 is a pinned runtime
  dependency, because `llm/gemini.py` catches its transport errors (FastAPI's `TestClient` uses
  it too); the server is uvicorn 0.54.0. The API is started with the uvicorn CLI rather than a
  `serve` subcommand, which keeps the CLI free of server options.
- `make check` runs the same checks plus the viewer build (the `Makefile` lists the other
  shortcuts).
