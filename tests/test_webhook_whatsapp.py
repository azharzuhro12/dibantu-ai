"""Offline tests for the simulated POST /webhook/whatsapp (Step 7).

The agent dependency is overridden with a ``DibantuAgent`` whose GLM HTTP
exchanges are answered by a scripted ``httpx.MockTransport`` -- the same
seam as the /api/chat tests -- while the real registry tools run against
the seeded mock store. No Meta credentials, no network, no real API key.
"""

from __future__ import annotations

import json

import pytest
from typing import Any

import httpx
from fastapi.testclient import TestClient

from app.agent.agent import AGENT_SYSTEM_PROMPT, DibantuAgent
from app.api.routes import get_agent
from app.config import Settings
from app.main import app
from app.tools.business_tools import check_stock, reset_mock_data
from app.tools.registry import get_tool_schemas


@pytest.fixture(autouse=True)
def _business_database(test_database):
    """Business tools run against the PostgreSQL test database."""
    yield

SENDER = "628123456789"


def make_settings(api_key: str = "test-key") -> Settings:
    """Build isolated settings so the real .env is never touched."""
    return Settings(
        glm_api_key=api_key,
        glm_base_url="https://glm.test/api/anthropic",
        glm_model="glm-test",
    )


class ScriptedGLM:
    """MockTransport body returning queued responses, recording payloads."""

    def __init__(self, bodies: list[dict[str, Any]]) -> None:
        self._bodies = list(bodies)
        self.payloads: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.payloads.append(json.loads(request.content))
        return httpx.Response(200, json=self._bodies.pop(0))


def text_body(text: str) -> dict[str, Any]:
    """Build a plain-text Anthropic-style response body."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
    }


def tool_body(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Build a single-tool_use Anthropic-style response body."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": f"toolu_{name}",
                "name": name,
                "input": tool_input,
            }
        ],
        "stop_reason": "tool_use",
    }


def webhook_agent(
    bodies: list[dict[str, Any]],
    *,
    api_key: str = "test-key",
) -> tuple[DibantuAgent, ScriptedGLM]:
    """Build a DibantuAgent answered by a scripted GLM transport."""
    glm = ScriptedGLM(bodies)
    agent = DibantuAgent(
        make_settings(api_key), transport=httpx.MockTransport(glm.handler)
    )
    return agent, glm


def post_webhook(
    agent: DibantuAgent,
    payload: dict[str, Any],
) -> httpx.Response:
    """POST an arbitrary JSON payload to /webhook/whatsapp."""
    app.dependency_overrides[get_agent] = lambda: agent
    try:
        with TestClient(app) as client:
            return client.post("/webhook/whatsapp", json=payload)
    finally:
        app.dependency_overrides.pop(get_agent, None)


def post_message(agent: DibantuAgent, message: str, sender: str = SENDER):
    """POST a WhatsApp-style message from ``sender``."""
    return post_webhook(agent, {"from": sender, "message": message})


def test_webhook_replies_to_valid_message_preserving_sender() -> None:
    agent, glm = webhook_agent([text_body("Halo! Saya DibantuAI.")])

    response = post_message(agent, "Halo, siapa kamu?")

    assert response.status_code == 200
    assert response.json() == {"to": SENDER, "response": "Halo! Saya DibantuAI."}
    # The WhatsApp text was handed to the agent as the user message.
    assert len(glm.payloads) == 1
    assert glm.payloads[0]["messages"] == [
        {"role": "user", "content": "Halo, siapa kamu?"}
    ]


def test_webhook_uses_existing_agent_loop() -> None:
    """The webhook goes through the same tool-calling agent as /api/chat."""
    agent, glm = webhook_agent([text_body("Oke.")])

    response = post_message(agent, "Halo")

    assert response.status_code == 200
    payload = glm.payloads[0]
    # Same registry schemas and Step 4/5 system prompt as the agent loop.
    assert payload["tools"] == get_tool_schemas()
    assert payload["system"] == AGENT_SYSTEM_PROMPT
    assert payload["model"] == "glm-test"


def test_webhook_stock_check_end_to_end() -> None:
    reset_mock_data()
    agent, glm = webhook_agent(
        [
            tool_body("check_stock", {"product_name": "Kopi Susu"}),
            text_body("Stok Kopi Susu 24 unit."),
        ]
    )

    response = post_message(agent, "Cek stok kopi susu")

    assert response.status_code == 200
    assert response.json() == {"to": SENDER, "response": "Stok Kopi Susu 24 unit."}
    # The real business tool ran and its result reached GLM.
    tool_result = glm.payloads[1]["messages"][2]["content"][0]
    assert tool_result["tool_use_id"] == "toolu_check_stock"
    assert json.loads(tool_result["content"])["stock"] == 24


def test_webhook_multi_step_order_end_to_end() -> None:
    reset_mock_data()
    agent, glm = webhook_agent(
        [
            tool_body("search_customer", {"name": "Budi"}),
            tool_body(
                "create_order",
                {
                    "customer_name": "Budi Santoso",
                    "items": [{"product_name": "Kopi Susu", "quantity": 3}],
                },
            ),
            text_body("Pesanan ORD-0005 dibuat: 3 Kopi Susu untuk Budi, total 54000."),
        ]
    )

    response = post_message(agent, "Budi pesan 3 kopi susu ya")

    assert response.status_code == 200
    body = response.json()
    assert body["to"] == SENDER
    assert "ORD-0005" in body["response"]
    assert len(glm.payloads) == 3  # search -> order -> final reply
    order_result = json.loads(
        glm.payloads[2]["messages"][4]["content"][0]["content"]
    )
    assert order_result["success"] is True
    assert order_result["order"]["total_price"] == 54000
    assert check_stock("Kopi Susu")["stock"] == 21


def test_webhook_rejects_missing_or_empty_sender() -> None:
    agent, _ = webhook_agent([text_body("x")])

    assert post_webhook(agent, {"message": "Halo"}).status_code == 422
    assert (
        post_webhook(agent, {"from": "", "message": "Halo"}).status_code == 422
    )


def test_webhook_rejects_missing_or_empty_message() -> None:
    agent, _ = webhook_agent([text_body("x")])

    assert post_webhook(agent, {"from": SENDER}).status_code == 422
    assert (
        post_webhook(agent, {"from": SENDER, "message": ""}).status_code == 422
    )


def test_webhook_glm_http_error_returns_502() -> None:
    agent = DibantuAgent(
        make_settings(),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(500, text="boom")
        ),
    )

    response = post_message(agent, "Halo")

    assert response.status_code == 502
    assert "GLM API error" in response.json()["detail"]


def test_webhook_unconfigured_agent_returns_503() -> None:
    agent, glm = webhook_agent([text_body("x")], api_key="")

    response = post_message(agent, "Halo")

    assert response.status_code == 503
    assert "GLM_API_KEY" in response.json()["detail"]
    assert glm.payloads == []


def test_webhook_tool_error_does_not_crash() -> None:
    """An unknown tool becomes an error result; the webhook still replies."""
    reset_mock_data()
    agent, glm = webhook_agent(
        [
            tool_body("teleport", {"city": "Bandung"}),
            text_body("Maaf, saya tidak punya kemampuan itu."),
        ]
    )

    response = post_message(agent, "Teleport saya ke Bandung")

    assert response.status_code == 200
    assert response.json() == {
        "to": SENDER,
        "response": "Maaf, saya tidak punya kemampuan itu.",
    }
    tool_result = glm.payloads[1]["messages"][2]["content"][0]
    assert tool_result["content"] == '{"error": "Unknown tool: teleport"}'
