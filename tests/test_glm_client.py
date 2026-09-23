"""Offline tests for GLMClient using httpx.MockTransport.

No network access, no real API key: settings carry a dummy key and every
HTTP exchange is answered by an in-memory transport.
"""

import asyncio
import json

import httpx
import pytest

from app.agent.glm_client import (
    GLMAPIError,
    GLMAuthError,
    GLMClient,
    GLMConfigError,
    GLMTimeoutError,
)
from app.config import Settings


def make_settings(api_key: str = "test-key") -> Settings:
    """Build isolated settings so the real .env is never touched."""
    return Settings(
        glm_api_key=api_key,
        glm_base_url="https://glm.test/api/anthropic",
        glm_model="glm-test",
    )


def ok_transport() -> httpx.MockTransport:
    """Transport that asserts the outgoing request and returns one text block."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/anthropic/v1/messages"
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.headers["anthropic-version"] == "2023-06-01"
        payload = json.loads(request.content)
        assert payload["model"] == "glm-test"
        assert payload["messages"] == [{"role": "user", "content": "Halo"}]
        return httpx.Response(
            200, json={"content": [{"type": "text", "text": " Halo! "}]}
        )

    return httpx.MockTransport(handler)


def complete(client: GLMClient) -> str:
    """Run one completion synchronously (pytest-asyncio is not installed)."""
    return asyncio.run(client.complete([{"role": "user", "content": "Halo"}]))


def test_complete_returns_trimmed_text() -> None:
    client = GLMClient(make_settings(), transport=ok_transport())
    assert complete(client) == "Halo!"


def test_complete_requires_api_key() -> None:
    client = GLMClient(make_settings(api_key=""))
    with pytest.raises(GLMConfigError):
        complete(client)


def test_complete_maps_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connection timed out")

    client = GLMClient(make_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(GLMTimeoutError):
        complete(client)


def test_complete_maps_auth_error() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(401, json={}))
    client = GLMClient(make_settings(), transport=transport)
    with pytest.raises(GLMAuthError):
        complete(client)


def test_complete_maps_http_error() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(500, text="boom"))
    client = GLMClient(make_settings(), transport=transport)
    with pytest.raises(GLMAPIError):
        complete(client)


def test_complete_rejects_payload_without_text() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"content": []})
    )
    client = GLMClient(make_settings(), transport=transport)
    with pytest.raises(GLMAPIError):
        complete(client)
