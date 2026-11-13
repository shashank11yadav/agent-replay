# ADR-006: Hosting: docker compose first, optional free-tier hosting

- Date: 2026-11-02
- Status: accepted

## Context
A reader should be able to see the baseline and regression runs in the viewer without an API key
or a cloud account. Free hosting tiers sleep, expire or change terms.

## Decision
- The primary way to run it is `docker compose up`: `db` (postgres:18.6-alpine with a
  healthcheck, published on `127.0.0.1` at `AGENT_REPLAY_DB_PORT`, default 5432), a one-shot
  `migrate`, a one-shot `seed` that imports the committed fixtures, and `api` (with a
  healthcheck on `/api/health`), which also serves the built viewer (ADR-009). No API key is
  needed to browse.
- Hosting is optional. `deploy/DEPLOY.md` describes a free-tier setup: the API on a Render web
  service, Postgres on Neon, the viewer on GitHub Pages built with `VITE_API_URL`.

## Alternatives considered
- Render free Postgres: expires 30 days after creation.
- Fly.io or Koyeb: both need a card on file.

## Consequences
- A hosted copy can sleep twice (API and database), so the viewer shows a "waking up" state and
  retries once.
