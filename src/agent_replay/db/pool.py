"""Connection pool factory."""

from __future__ import annotations

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from agent_replay.config import Settings

# The API only reads; no request should hold a connection longer than this.
STATEMENT_TIMEOUT_MS = 5000


def get_pool(settings: Settings) -> ConnectionPool:
    """Build an unopened pool returning dict rows; the caller opens and closes it.

    Every connection gets a statement timeout, so a slow query cannot pin the database.
    """
    if not settings.database_url:
        raise ValueError("AGENT_REPLAY_DATABASE_URL is not set")
    return ConnectionPool(
        settings.database_url,
        kwargs={
            "row_factory": dict_row,
            "options": f"-c statement_timeout={STATEMENT_TIMEOUT_MS}",
        },
        open=False,
    )
