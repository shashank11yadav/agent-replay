"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles

from agent_replay.api.routes import router
from agent_replay.config import Settings
from agent_replay.db.pool import get_pool


def content_security_policy(cors_origins: list[str]) -> str:
    """CSP for the API and the built viewer, which loads only its own bundle."""
    connect = " ".join(["'self'", *cors_origins])
    return f"default-src 'self'; connect-src {connect}; frame-ancestors 'none'"


def create_app(settings: Settings) -> FastAPI:
    """Build the read-only API; the DB pool lives for the app's lifespan."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        with get_pool(settings) as pool:
            app.state.pool = pool
            yield

    origins = list(settings.cors_origins)
    bad = [o for o in origins if o.strip() in ("", "*")]
    if bad:
        raise ValueError(
            "CORS origins must be explicit; '*' and empty entries are not allowed"
        )

    hosts = list(settings.allowed_hosts)
    if not hosts or any(not h.strip() for h in hosts):
        raise ValueError("AGENT_REPLAY_ALLOWED_HOSTS must list at least one host")

    app = FastAPI(
        title="agent-replay",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET"],
        allow_credentials=False,
    )
    headers = {
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Content-Security-Policy": content_security_policy(origins),
    }

    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        for name, value in headers.items():
            response.headers.setdefault(name, value)
        return response

    # Added last so it runs first: a request for another Host never reaches a route.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)
    app.include_router(router, prefix="/api")
    if settings.viewer_dist is not None:
        app.mount(
            "/",
            StaticFiles(directory=settings.viewer_dist, html=True),
            name="viewer",
        )
    return app
