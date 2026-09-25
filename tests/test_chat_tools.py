"""Offline API-integration tests for POST /api/chat with tool calling (Step 4).

The agent dependency is overridden with a ``DibantuAgent`` whose GLM HTTP
exchanges are answered by a scripted ``httpx.MockTransport``. The real
registry tools run against the mock store, so the complete flow -- user
message -> /api/chat -> Agent -> GLM -> tool_use -> business tool ->
tool_result -> GLM -> final text -> ChatResponse -- is exercised with no
network access and no real API key.
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

REGISTERED_TOOL_NAMES = [
    "check_stock",
    "create_order",
    "update_stock",
    "search_customer",
    "get_sales_report",
    "get_low_stock",
    "search_knowledge_base",
    "save_memory",
    "search_memory",
    "delete_memory",
    # Advertised sensitive-request schemas (Step 16): the model can ask
    # for these, which creates a human approval — they are never
    # dispatchable (get_tool still returns None for them).
    "refund_order",
    "cancel_order",
    "bulk_stock_update",
]


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
    """Build a tool_use Anthropic-style response body."""
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


def tool_agent(
    bodies: list[dict[str, Any]],
    *,
    api_key: str = "test-key",
    tools: dict[str, Any] | None = None,
) -> tuple[DibantuAgent, ScriptedGLM]:
    """Build a DibantuAgent answered by a scripted GLM transport."""
    glm = ScriptedGLM(bodies)
    agent = DibantuAgent(
        make_settings(api_key),
        transport=httpx.MockTransport(glm.handler),
        tools=tools,
    )
    return agent, glm


def post_raw(agent: DibantuAgent, payload: dict[str, Any]) -> httpx.Response:
    """POST an arbitrary JSON payload to /api/chat with the agent injected."""
    app.dependency_overrides[get_agent] = lambda: agent
    try:
        with TestClient(app) as client:
            return client.post("/api/chat", json=payload)
    finally:
        app.dependency_overrides.pop(get_agent, None)


def post_chat(agent: DibantuAgent, message: str) -> httpx.Response:
    """POST a chat message to /api/chat with the agent dependency overridden."""
    return post_raw(agent, {"message": message})


def tool_result_of(glm: ScriptedGLM, call_index: int) -> dict[str, Any]:
    """Return the parsed tool_result content of GLM call ``call_index``."""
    messages = glm.payloads[call_index]["messages"]
    block = next(
        block
        for message in messages
        if isinstance(message["content"], list)
        for block in message["content"]
        if block.get("type") == "tool_result"
    )
    return json.loads(block["content"])


def test_chat_direct_response_without_tool_calls() -> None:
    """A plain-text GLM reply is returned as-is with no tool execution."""
    agent, glm = tool_agent([text_body("Halo! Saya DibantuAI.")])

    response = post_chat(agent, "Halo, siapa kamu?")

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "Halo! Saya DibantuAI."
    assert body["run_id"].startswith("run-")  # Step 15 tracing id
    assert len(glm.payloads) == 1
    # The agent advertises the registry schemas and the Step 4 prompt.
    assert glm.payloads[0]["tools"] == get_tool_schemas()
    assert [tool["name"] for tool in glm.payloads[0]["tools"]] == (
        REGISTERED_TOOL_NAMES
    )
    assert glm.payloads[0]["system"] == AGENT_SYSTEM_PROMPT
    assert glm.payloads[0]["messages"] == [
        {"role": "user", "content": "Halo, siapa kamu?"}
    ]


def test_chat_single_tool_call_end_to_end() -> None:
    """check_stock runs against the real mock store and feeds GLM's reply."""
    reset_mock_data()
    agent, glm = tool_agent(
        [
            tool_body("check_stock", {"product_name": "Croissant"}),
            text_body("Stok Croissant tersisa 8 unit dengan harga Rp25.000."),
        ]
    )

    response = post_chat(agent, "Berapa stok Croissant?")

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "Stok Croissant tersisa 8 unit dengan harga Rp25.000."
    assert body["run_id"].startswith("run-")  # Step 15 tracing id
    assert len(glm.payloads) == 2
    second_messages = glm.payloads[1]["messages"]
    assert second_messages[1] == {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                # Step 5: the API-provided tool_use id is echoed back.
                "id": "toolu_check_stock",
                "name": "check_stock",
                "input": {"product_name": "Croissant"},
            }
        ],
    }
    assert second_messages[2]["content"][0]["type"] == "tool_result"
    assert second_messages[2]["content"][0]["tool_use_id"] == "toolu_check_stock"
    result = tool_result_of(glm, 1)
    assert result == {
        "success": True,
        "product_name": "Croissant",
        "stock": 8,
        "price": 25000,
        "in_stock": True,
        "low_stock": True,
    }


def test_chat_multi_step_tool_calls() -> None:
    """Two chained tool calls run in order before the final reply."""
    reset_mock_data()
    agent, glm = tool_agent(
        [
            tool_body("check_stock", {"product_name": "Kopi Susu"}),
            tool_body("get_low_stock", {"threshold": 20}),
            text_body(
                "Kopi Susu masih 24 unit. Stok di bawah 20: Croissant (8) "
                "dan Matcha Latte (15)."
            ),
        ]
    )

    response = post_chat(
        agent, "Cek stok Kopi Susu, lalu sebutkan produk dengan stok di bawah 20."
    )

    assert response.status_code == 200
    assert "Croissant" in response.json()["response"]
    assert len(glm.payloads) == 3
    final_messages = glm.payloads[2]["messages"]
    assert [message["role"] for message in final_messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert json.loads(final_messages[2]["content"][0]["content"])["stock"] == 24
    low_stock = json.loads(final_messages[4]["content"][0]["content"])
    assert low_stock["success"] is True
    assert [item["product_name"] for item in low_stock["products"]] == [
        "Croissant",
        "Matcha Latte",
    ]


def test_chat_create_order_tool_has_real_side_effects() -> None:
    """An order placed through the API actually deducts mock-store stock."""
    reset_mock_data()
    agent, glm = tool_agent(
        [
            tool_body(
                "create_order",
                {
                    "customer_name": "Budi Santoso",
                    "items": [{"product_name": "Teh Manis", "quantity": 2}],
                },
            ),
            text_body("Pesanan dibuat: 2 Teh Manis untuk Budi Santoso, total Rp20.000."),
        ]
    )

    response = post_chat(agent, "Budi mau pesan 2 Teh Manis.")

    assert response.status_code == 200
    result = tool_result_of(glm, 1)
    assert result["success"] is True
    assert result["order"]["total_price"] == 20000
    assert result["new_customer"] is False
    assert check_stock("Teh Manis")["stock"] == 28  # 30 seeded - 2 ordered


def test_chat_unknown_tool_returns_error_result_to_model() -> None:
    """A tool GLM never had becomes an error result, not a server crash."""
    reset_mock_data()
    agent, glm = tool_agent(
        [
            tool_body("teleport", {"city": "Bandung"}),
            text_body("Maaf, saya tidak punya kemampuan untuk itu."),
        ]
    )

    response = post_chat(agent, "Teleport saya ke Bandung.")

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "Maaf, saya tidak punya kemampuan untuk itu."
    assert body["run_id"].startswith("run-")  # Step 15 tracing id
    messages = glm.payloads[1]["messages"]
    assert messages[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_teleport",
        "content": '{"error": "Unknown tool: teleport"}',
    }


def test_chat_tool_exception_is_reported_to_model() -> None:
    """A raising tool becomes an error result; the reply still completes."""
    reset_mock_data()

    def broken_stock(product_name: str) -> dict[str, Any]:
        raise RuntimeError("store crashed")

    agent, glm = tool_agent(
        [
            tool_body("check_stock", {"product_name": "Kopi Susu"}),
            text_body("Maaf, pemeriksaan stok sedang gagal."),
        ],
        tools={"check_stock": broken_stock},
    )

    response = post_chat(agent, "Berapa stok Kopi Susu?")

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "Maaf, pemeriksaan stok sedang gagal."
    assert body["run_id"].startswith("run-")  # Step 15 tracing id
    messages = glm.payloads[1]["messages"]
    assert messages[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_check_stock",
        "content": '{"error": "RuntimeError: store crashed"}',
    }


def test_chat_rejects_empty_or_missing_message() -> None:
    """Validation still applies: empty and absent messages are 422s."""
    agent, _ = tool_agent([text_body("x")])

    assert post_chat(agent, "").status_code == 422
    assert post_raw(agent, {}).status_code == 422


def test_chat_unconfigured_agent_returns_503() -> None:
    """Without an API key the endpoint answers 503 before any GLM call."""
    agent, glm = tool_agent([text_body("x")], api_key="")

    response = post_chat(agent, "Halo")

    assert response.status_code == 503
    assert "GLM_API_KEY" in response.json()["detail"]
    assert glm.payloads == []


def test_chat_glm_http_error_returns_502() -> None:
    """A GLM-side HTTP failure inside the loop still maps to 502."""
    agent = DibantuAgent(
        make_settings(),
        transport=httpx.MockTransport(lambda request: httpx.Response(500, text="boom")),
    )

    response = post_chat(agent, "Halo")

    assert response.status_code == 502


def test_agent_system_prompt_sets_tool_rules() -> None:
    """The Step 4 prompt covers tool use, honesty, and clarity rules."""
    prompt = AGENT_SYSTEM_PROMPT.lower()
    assert "tools" in prompt
    assert "never invent" in prompt
    assert "tool result" in prompt
    assert "clarification" in prompt
    assert "never claim an action succeeded" in prompt
    assert "concise" in prompt
