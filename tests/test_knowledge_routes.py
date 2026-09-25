"""Tests for the knowledge-base API endpoints (Step 12).

GET /api/knowledge/search and POST /api/knowledge/ingest run against
the rag_service fixture (deterministic hashing embeddings, temporary
store), so no model download or network access is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.service import RagService, reset_rag_service, set_rag_service


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_search_returns_cited_results(client, rag_service) -> None:
    response = client.get(
        "/api/knowledge/search", params={"q": "aturan refund pesanan dibayar"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["results"]
    assert body["results"][0]["source"] == "refund_policy.md"
    assert "refund_policy.md" in body["sources"]


def test_search_validates_query(client, rag_service) -> None:
    assert client.get("/api/knowledge/search").status_code == 422  # missing
    assert (
        client.get("/api/knowledge/search", params={"q": ""}).status_code == 422
    )


def test_search_validates_top_k(client, rag_service) -> None:
    ok = client.get(
        "/api/knowledge/search", params={"q": "refund", "top_k": 1}
    )
    assert ok.status_code == 200
    assert len(ok.json()["results"]) == 1
    assert (
        client.get(
            "/api/knowledge/search", params={"q": "refund", "top_k": 11}
        ).status_code
        == 422
    )


def test_ingest_runs_and_reports(client, rag_service) -> None:
    response = client.post("/api/knowledge/ingest")
    assert response.status_code == 200
    body = response.json()
    # The fixture already ingested: everything is unchanged now.
    assert body["documents_found"] == 4
    assert body["documents_skipped"] == 4
    assert body["chunks_created"] == 0
    assert body["total_chunks"] > 0
    assert body["knowledge_dir"].endswith("data/knowledge")


def test_ingest_takes_no_path_parameter(client, rag_service) -> None:
    """Ingestion is restricted to the configured directory: callers
    cannot point it at arbitrary filesystem paths."""
    response = client.post("/api/knowledge/ingest", json={"path": "/etc"})
    assert response.status_code == 200  # body is ignored entirely
    body = response.json()
    assert body["knowledge_dir"].endswith("data/knowledge")


def test_unavailable_stack_returns_503(client, tmp_path) -> None:
    service = RagService(
        knowledge_dir=str(tmp_path / "nowhere"),  # missing directory
        store_dir=str(tmp_path / "chroma"),
        embeddings_provider="hashing",
    )
    set_rag_service(service)
    try:
        ingest = client.post("/api/knowledge/ingest")
        assert ingest.status_code == 503
        assert "does not exist" in ingest.json()["detail"]
    finally:
        reset_rag_service()
