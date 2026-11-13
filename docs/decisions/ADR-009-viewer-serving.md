# ADR-009: Viewer serving: FastAPI StaticFiles mount in one container

- Date: 2026-11-02
- Status: accepted

## Context
`docker compose up` must show the viewer with no key. Vite's docs say `vite preview` is for
previewing a build locally, not a production server
([static deploy guide](https://vite.dev/guide/static-deploy.html)). FastAPI documents mounting
`StaticFiles` ([static files](https://fastapi.tiangolo.com/tutorial/static-files/)), but does not
spell out what `html=True` does for `/`.

## Decision
- All API routes are under `/api` (`include_router(prefix="/api")`).
- When `AGENT_REPLAY_VIEWER_DIST` is set and the directory exists, `create_app` mounts
  `StaticFiles(directory=<dist>, html=True)` at `/` after including the API router, so `/api/*`
  is matched first.
- The viewer uses hash routes (`#/runs/<id>`), so the server never needs a client-side routing
  fallback; only `/` serving `index.html` matters. An API test (`tests/integration/test_api.py`)
  checks that `GET /` returns the viewer's `index.html` and `GET /api/health` still returns JSON.
- One image: a `node:22.23.4-slim` stage runs `npm ci` and `npm run build`; the final Python
  stage copies `viewer/dist`. Compose has no separate viewer service.
- The viewer calls same-origin `/api`. `VITE_API_URL` exists only for a separately hosted viewer,
  where CORS (an allowlist, GET only) is needed.

## Alternatives considered
- `vite preview` or `vite dev` in a container: not meant for serving.
- An nginx container: one more service and config file for no gain at this size.

## Consequences
- One origin in compose, so no CORS locally.
- The image needs a Node build stage, which makes the build slower, but the runtime has no Node.
- The viewer's dev proxy forwards `/api` to the API without rewriting paths.
