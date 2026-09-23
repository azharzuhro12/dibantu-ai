"""Scratch-database isolation for everything that resets business data.

The evaluation (and the business-data tests) reset and re-seed the
store before every case via ``reset_mock_data()`` — a TRUNCATE. Left
unchecked that would wipe whatever database ``DATABASE_URL`` happens to
point at, including the real one used by the running Docker stack
(``python -m evaluation.evaluator`` loads ``.env`` like any host-run
process). This module prevents exactly that: it points the application
at a dedicated scratch database (dropped and re-created up front) so
the main business database is never touched.

Both pytest (``conftest.py``) and the evaluator CLI use it, so the
isolation rules live in one place.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine

__all__ = [
    "DEFAULT_TEST_DATABASE_URL",
    "ScratchDatabaseUnavailable",
    "database_name",
    "ensure_scratch_database",
]

#: Default scratch database on the Compose Postgres (host port 5433).
#: Override for both pytest and the evaluator with TEST_DATABASE_URL.
DEFAULT_TEST_DATABASE_URL = (
    "postgresql+psycopg://dibantu:dibantu_password@localhost:5433/dibantu_ai_test"
)


class ScratchDatabaseUnavailable(RuntimeError):
    """Raised when the PostgreSQL scratch server is not reachable."""


def _admin_dsn(database_url: str) -> str:
    """Postgres-admin DSN (database 'postgres') for the same server."""
    plain = database_url.replace("postgresql+psycopg://", "postgresql://")
    return plain.rsplit("/", 1)[0] + "/postgres"


def database_name(database_url: str) -> str:
    """Return the database name of a URL (for safe logging: no password)."""
    return database_url.rsplit("/", 1)[1]


def ensure_scratch_database() -> str:
    """Point DATABASE_URL at a freshly created scratch database.

    Returns the scratch URL. When DATABASE_URL already points at the
    scratch database (pytest's ``test_database`` fixture ran first, or a
    caller set it up explicitly) the existing database is reused as-is;
    otherwise it is dropped and re-created, the schema is applied, and
    ``os.environ["DATABASE_URL"]`` is switched over so the application's
    lazily created engines connect to it from now on.

    Raises ScratchDatabaseUnavailable when no PostgreSQL server is
    reachable at the scratch URL.
    """
    url = os.environ.get("TEST_DATABASE_URL") or DEFAULT_TEST_DATABASE_URL
    if os.environ.get("DATABASE_URL", "") == url:
        return url

    name = database_name(url)
    try:
        import psycopg

        with psycopg.connect(_admin_dsn(url), autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            conn.execute(f'CREATE DATABASE "{name}"')
    except Exception as exc:  # noqa: BLE001 - any failure means "unreachable"
        raise ScratchDatabaseUnavailable(
            f"PostgreSQL scratch server unreachable ({exc.__class__.__name__})."
            " Start it: docker compose up -d postgres"
        ) from exc

    from app.db.models import Base

    engine = create_engine(url)
    try:
        Base.metadata.create_all(engine)
    finally:
        engine.dispose()

    # Drop any engine cached for the previous URL (tests, earlier runs)
    # so subsequent sessions bind to the scratch database.
    os.environ["DATABASE_URL"] = url
    return url
