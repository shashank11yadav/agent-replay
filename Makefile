# Developer shortcuts. `make check` runs what CI runs (except the Docker build: `make image`).
# The Python tests need Postgres at AGENT_REPLAY_TEST_DATABASE_URL (default: localhost:5432);
# `make db` starts the compose one.

# Host port of the compose db; docker compose reads it too.
AGENT_REPLAY_DB_PORT ?= 5432
export AGENT_REPLAY_DB_PORT
GATE_DATABASE_URL ?= postgresql://postgres:postgres@127.0.0.1:$(AGENT_REPLAY_DB_PORT)/postgres

.PHONY: setup db lint fmt typecheck test test-fast viewer-test viewer-build check image up down gate-demo

setup:
	uv sync
	npm --prefix viewer ci

db:
	docker compose up -d --wait db

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy src
	npm --prefix viewer run typecheck

test:
	uv run pytest -q

# Unit and provider tests only, about 20 seconds (the Claude Code end-to-end test still
# needs Postgres).
test-fast:
	uv run pytest -q tests/unit tests/providers

viewer-test:
	npm --prefix viewer test

viewer-build:
	npm --prefix viewer run build

check: lint typecheck test viewer-test viewer-build

image:
	docker build -t agent-replay:local .

up:
	docker compose up -d --build --wait

down:
	docker compose down

# Offline gate on the committed fixtures: baseline PASS, prompt regression FAIL.
# Uses a throwaway database on the server in GATE_DATABASE_URL; never calls a model.
gate-demo:
	AGENT_REPLAY_DATABASE_URL="$(GATE_DATABASE_URL)" scripts/offline-gate.sh
