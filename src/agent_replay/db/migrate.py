"""Plain-SQL migration runner: one transaction per file, recorded once."""

from __future__ import annotations

from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


# Arbitrary constant key for the migration advisory lock ("agrp" in ASCII).
_LOCK_KEY = 0x61677270


class MigrationError(Exception):
    """A migration file failed to apply."""

    def __init__(self, name: str, cause: Exception) -> None:
        self.name = name
        super().__init__(f"migration {name} failed: {cause}")


def list_migrations(directory: Path = MIGRATIONS_DIR) -> list[Path]:
    """Return the ``*.sql`` files in filename order."""
    return sorted(directory.glob("*.sql"), key=lambda p: p.name)


def apply_migrations(conninfo: str, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply pending migrations in order; return the names applied.

    Raises MigrationError naming the first failing file; that file's changes are
    rolled back and it is not recorded.
    """
    applied: list[str] = []
    with psycopg.connect(conninfo) as conn:
        # Session-level lock: concurrent runners queue here and then see the
        # schema the first one applied. Released when the connection closes.
        conn.execute("SELECT pg_advisory_lock(%s)", (_LOCK_KEY,))
        conn.commit()
        conn.execute(_BOOTSTRAP)
        conn.commit()
        done = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
        conn.commit()
        for path in list_migrations(directory):
            if path.name in done:
                continue
            try:
                with conn.transaction():
                    conn.execute(path.read_text())
                    conn.execute(
                        "INSERT INTO schema_migrations (version) VALUES (%s)",
                        (path.name,),
                    )
            except psycopg.Error as exc:
                raise MigrationError(path.name, exc) from exc
            applied.append(path.name)
    return applied
