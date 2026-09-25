"""Offline tests for POST /api/chat with a mocked agent.

The real agent dependency is overridden with one whose GLM calls are answered
by an in-memory transport, so no network access or real API key is needed.
"""

import httpx
from fastapi.testclient import TestClient

from app.agent.agent import DibantuAgent
from app.api.routes import get_agent
from app.config import Settings
from app.main import app


def make_settings(api_key: str = "test-key") -> Settings:
    """Build isolated settings so the real .env is never touched."""
    return Settings(
        glm_api_key=api_key,
        glm_base_url="https://glm.test/api/anthropic",
        glm_model="glm-test",
    )


def mock_agent(api_key: str = "test-key", status: int = 200) -> DibantuAgent:
    """Build an agent whose GLM exchange never leaves the process."""

    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, text="boom")
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "Halo! Saya DibantuAI."}]},
        )

    return DibantuAgent(make_settings(api_key), transport=httpx.MockTransport(handler))


def post_chat(agent: DibantuAgent, message: str) -> httpx.Response:
    """POST to /api/chat with the agent dependency overridden."""
    app.dependency_overrides[get_agent] = lambda: agent
    try:
        with TestClient(app) as client:
            return client.post("/api/chat", json={"message": message})
    finally:
        app.dependency_overrides.pop(get_agent, None)


def test_chat_returns_mocked_reply() -> None:
    response = post_chat(mock_agent(), "Halo, siapa kamu?")
    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "Halo! Saya DibantuAI."
    assert body["run_id"].startswith("run-")  # Step 15 tracing id


def test_chat_unconfigured_returns_503() -> None:
    response = post_chat(mock_agent(api_key=""), "Halo")
    assert response.status_code == 503
    assert "GLM_API_KEY" in response.json()["detail"]


def test_chat_glm_http_error_returns_502() -> None:
    response = post_chat(mock_agent(status=500), "Halo")
    assert response.status_code == 502


def test_chat_rejects_empty_message() -> None:
    response = post_chat(mock_agent(), "")
    assert response.status_code == 422
