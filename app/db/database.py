"""Engine and session management for the DibantuAI PostgreSQL layer.

The connection string comes from the ``DATABASE_URL`` environment
variable (see ``.env.example``); nothing is hardcoded. Engines are
created lazily and cached per URL so tests can point the app at a
scratch database by setting the variable before the first call.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

# Load .env (idempotent) so host-run processes find DATABASE_URL too.
load_dotenv()


class DatabaseNotConfiguredError(RuntimeError):
    """Raised when DATABASE_URL is missing from the environment."""


def get_database_url() -> str:
    """Return the configured PostgreSQL URL (postgresql+psycopg://...)."""
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise DatabaseNotConfiguredError(
            "DATABASE_URL is not configured (see .env.example)."
        )
    return url


#: Lazily created engines, cached per URL (tests may switch URLs).
_ENGINES: dict[str, Engine] = {}


def get_engine() -> Engine:
    """Return a cached engine for the currently configured database."""
    url = get_database_url()
    engine = _ENGINES.get(url)
    if engine is None:
        engine = create_engine(url, pool_pre_ping=True)
        _ENGINES[url] = engine
    return engine


def get_session_factory() -> sessionmaker:
    """Return a session factory bound to the current engine."""
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Yield a session that commits on success and rolls back on error.

    The business tools use this for every database interaction; raising
    inside the ``with`` block leaves no partial writes behind.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def dispose_engines() -> None:
    """Close every cached engine (used by test teardown)."""
    for engine in _ENGINES.values():
        engine.dispose()
    _ENGINES.clear()
