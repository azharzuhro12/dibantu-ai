"""Tests for the CORS middleware that lets the web frontend call the API.

The Next.js frontend talks to this backend directly from the browser.
`next dev` serves on 3000, or 3002 when another project already holds
3000, so the default allowlist covers both ports on localhost and
127.0.0.1. Cross-origin GETs and the POST /api/chat preflight must be
answered with the right CORS headers; everything else must not.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

#: Every local-dev origin the frontend can end up on.
FRONTEND_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3002",
    "http://127.0.0.1:3002",
]

FRONTEND_ORIGIN = "http://localhost:3000"

PREFLIGHT_HEADERS = {
    "Access-Control-Request-Method": "POST",
    "Access-Control-Request-Headers": "content-type",
}


@pytest.mark.parametrize("origin", FRONTEND_ORIGINS)
def test_cors_allows_frontend_origin_on_get(origin: str) -> None:
    """A cross-origin GET from any frontend dev origin must be allowed."""
    with TestClient(app) as client:
        response = client.get("/health", headers={"Origin": origin})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


@pytest.mark.parametrize("origin", FRONTEND_ORIGINS)
@pytest.mark.parametrize(
    "path", ["/health", "/api/chat", "/api/approvals"]
)
def test_cors_preflight_succeeds(origin: str, path: str) -> None:
    """Preflight OPTIONS must succeed for every endpoint the frontend calls."""
    with TestClient(app) as client:
        response = client.options(
            path, headers={"Origin": origin, **PREFLIGHT_HEADERS}
        )

    # Starlette answered preflights with 200 before 0.35 and 204 after;
    # any 2xx is a valid preflight response for the browser.
    assert response.status_code in {200, 204}
    assert response.headers["access-control-allow-origin"] == origin
    assert "POST" in response.headers["access-control-allow-methods"]
    allowed_headers = response.headers["access-control-allow-headers"].lower()
    assert "content-type" in allowed_headers


def test_cors_preflight_for_chat_post() -> None:
    """The browser preflight for POST /api/chat (JSON) must pass."""
    with TestClient(app) as client:
        response = client.options(
            "/api/chat",
            headers={"Origin": FRONTEND_ORIGIN, **PREFLIGHT_HEADERS},
        )

    assert response.status_code in {200, 204}
    assert response.headers["access-control-allow-origin"] == FRONTEND_ORIGIN
    assert "POST" in response.headers["access-control-allow-methods"]
    allowed_headers = response.headers["access-control-allow-headers"].lower()
    assert "content-type" in allowed_headers


def test_cors_ignores_unknown_origin() -> None:
    """Non-allowlisted origins get no allow-origin header."""
    with TestClient(app) as client:
        response = client.get(
            "/health", headers={"Origin": "https://evil.example"}
        )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_cors_rejects_preflight_from_unknown_origin() -> None:
    """Preflight from a non-allowlisted origin must not be allowed."""
    with TestClient(app) as client:
        response = client.options(
            "/api/chat",
            headers={"Origin": "https://evil.example", **PREFLIGHT_HEADERS},
        )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
