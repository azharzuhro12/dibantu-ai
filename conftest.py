"""Pytest configuration.

Ensures the project root is on ``sys.path`` so tests can import the
``app`` package regardless of how pytest is invoked, and provides the
``test_database`` fixture: an isolated PostgreSQL scratch database
(created on the local Docker Compose Postgres) used by every test and
evaluation run that touches business data — see
``evaluation/db_isolation.py`` for why resetting tools must never run
against the main database.

The business-data tests are skipped — with an explicit reason — when
no PostgreSQL test server is reachable, so the rest of the suite never
depends on an unavailable external service. Start one with:

    docker compose up -d postgres
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Project root on sys.path (repo root = parent of this file).
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def test_database():
    """Provide a freshly created scratch database for the session.

    Sets DATABASE_URL so the application's lazy engine connects to the
    scratch database, creates the schema, and yields its URL. Skips
    the requesting tests when no PostgreSQL server is reachable.
    """
    from evaluation.db_isolation import (
        ScratchDatabaseUnavailable,
        ensure_scratch_database,
    )

    try:
        url = ensure_scratch_database()
    except ScratchDatabaseUnavailable as exc:
        pytest.skip(str(exc), allow_module_level=False)

    yield url

    from app.db.database import dispose_engines

    dispose_engines()


@pytest.fixture()
def rag_service(tmp_path):
    """Isolated, pre-ingested RAG stack for knowledge-base tests (Step 12).

    Mirrors the scratch-database idea: the demo knowledge documents are
    ingested with deterministic ``hashing`` embeddings into a per-test
    temporary store — no model download, no network, and the real
    ``data/rag`` directory is never touched — and the shared service
    singleton is restored afterwards so later tests rebuild from
    settings.
    """
    from app.rag.service import RagService, reset_rag_service, set_rag_service

    service = RagService(
        knowledge_dir=str(ROOT / "data" / "knowledge"),
        store_dir=str(tmp_path / "chroma"),
        embeddings_provider="hashing",
    )
    service.ingest()
    set_rag_service(service)
    yield service
    reset_rag_service()
