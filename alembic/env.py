"""Alembic environment for DibantuAI.

Reads DATABASE_URL from the environment (with .env loaded for host
runs) — the same source of truth the application uses — and targets
the ORM metadata in app.db.models.
"""

from __future__ import annotations

import os

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import create_engine

from app.db.models import Base

# Load .env for host-run migrations (idempotent; containers get the
# variable from Docker Compose instead).
load_dotenv()

config = context.config

target_metadata = Base.metadata


def database_url() -> str:
    """Resolve the migration URL: DATABASE_URL env wins over the ini."""
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        return url
    return config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live database."""
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against the live database."""
    engine = create_engine(database_url(), pool_pre_ping=True)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
