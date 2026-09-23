"""Tests for the /health endpoint and Swagger docs."""

from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_ok() -> None:
    """GET /health must return the standard service status payload."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "DibantuAI",
        "version": "0.1.0",
    }


def test_swagger_docs_are_served() -> None:
    """GET /docs must serve the built-in Swagger UI."""
    with TestClient(app) as client:
        response = client.get("/docs")

    assert response.status_code == 200
