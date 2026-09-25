"""Agent-level RAG tests (Step 12): tool selection, grounding with
citations, the unknown-knowledge path, and mixed RAG + database flows.

Same offline pattern as test_chat_tools: the GLM HTTP exchanges are
answered by a scripted httpx.MockTransport while the real registry
tools run, so the complete flow — user message -> Agent -> GLM ->
search_knowledge_base -> ChromaDB -> tool_result -> GLM -> cited final
answer — needs no network, no API key, and no model download (the
rag_service fixture uses deterministic hashing embeddings).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agent.agent import AGENT_SYSTEM_PROMPT, DibantuAgent
from app.api.routes import get_agent
from app.config import Settings
from app.main import app
from app.tools.business_tools import reset_mock_data


class ScriptedGLM:
    """MockTransport body returning queued responses, recording payloads."""

    def __init__(self, bodies: list[dict[str, Any]]) -> None:
        self._bodies = list(bodies)
        self.payloads: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.payloads.append(json.loads(request.content))
        return httpx.Response(200, json=self._bodies.pop(0))


def text_body(text: str) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
    }


def tool_body(calls: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Build one response body requesting one or more tools."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": f"toolu_{index}_{name}",
                "name": name,
                "input": tool_input,
            }
            for index, (name, tool_input) in enumerate(calls)
        ],
        "stop_reason": "tool_use",
    }


def make_agent(bodies: list[dict[str, Any]]) -> tuple[DibantuAgent, ScriptedGLM]:
    glm = ScriptedGLM(bodies)
    agent = DibantuAgent(
        Settings(
            glm_api_key="test-key",
            glm_base_url="https://glm.test/api/anthropic",
            glm_model="glm-test",
        ),
        transport=httpx.MockTransport(glm.handler),
    )
    return agent, glm


def post_chat(agent: DibantuAgent, message: str) -> httpx.Response:
    app.dependency_overrides[get_agent] = lambda: agent
    try:
        with TestClient(app) as client:
            return client.post("/api/chat", json={"message": message})
    finally:
        app.dependency_overrides.pop(get_agent, None)


def tool_results_of(glm: ScriptedGLM) -> list[dict[str, Any]]:
    """Every parsed tool_result the agent fed back to the model."""
    results = []
    for payload in glm.payloads:
        for message in payload["messages"]:
            if not isinstance(message["content"], list):
                continue
            for block in message["content"]:
                if block.get("type") == "tool_result":
                    results.append(json.loads(block["content"]))
    return results


def executed_tool_names(glm: ScriptedGLM) -> list[str]:
    """Tool names the agent actually executed (echoed tool_use blocks)."""
    names = []
    for payload in glm.payloads:
        for message in payload["messages"]:
            if not isinstance(message["content"], list):
                continue
            for block in message["content"]:
                if block.get("type") == "tool_use":
                    names.append(block["name"])
    return names


def test_agent_selects_knowledge_tool_and_cites_source(rag_service) -> None:
    """Policy question -> search_knowledge_base -> grounded cited reply."""
    reply = (
        "Menurut refund_policy.md (bagian Syarat Kelayakan Refund), refund "
        "hanya bisa diajukan untuk pesanan yang sudah dibayar dan paling "
        "lambat 1x24 jam setelah pesanan dibuat.\n\n"
        "Sumber: refund_policy.md — Syarat Kelayakan Refund."
    )
    agent, glm = make_agent(
        [
            tool_body(
                [("search_knowledge_base", {"query": "aturan refund pesanan sudah dibayar"})]
            ),
            text_body(reply),
        ]
    )

    response = post_chat(agent, "Apa aturan refund untuk pesanan yang sudah dibayar?")

    assert response.status_code == 200
    assert "refund_policy.md" in response.json()["response"]
    assert executed_tool_names(glm) == ["search_knowledge_base"]
    results = tool_results_of(glm)
    assert results and results[0]["success"] is True
    # Grounding: everything cited reached the model via a real tool result.
    assert results[0]["results"][0]["source"] == "refund_policy.md"
    joined = json.dumps(results, ensure_ascii=False)
    assert "1x24 jam" in joined


def test_agent_says_unavailable_for_unknown_knowledge(rag_service) -> None:
    """After searching, the agent must admit missing info, not invent."""
    agent, glm = make_agent(
        [
            tool_body(
                [("search_knowledge_base", {"query": "kebijakan lembur karyawan"})]
            ),
            text_body(
                "Maaf, kebijakan mengenai lembur karyawan tidak tersedia "
                "di basis pengetahuan."
            ),
        ]
    )

    response = post_chat(agent, "Berapa tarif lembur karyawan?")

    assert response.status_code == 200
    assert "tidak tersedia" in response.json()["response"]
    assert executed_tool_names(glm) == ["search_knowledge_base"]


def test_agent_prompt_teaches_rag_grounding() -> None:
    prompt = AGENT_SYSTEM_PROMPT.lower()
    assert "search_knowledge_base" in prompt
    assert "only from the returned passages" in prompt
    assert "not available in the knowledge base" in prompt
    assert "never invent or guess policy" in prompt
    # The two data worlds stay separate.
    assert "strictly separate" in prompt


def test_mixed_policy_and_stock_request(rag_service, test_database) -> None:
    """One turn may combine RAG and database tools (Step 5 batching)."""
    reset_mock_data()
    reply = (
        "Menurut inventory_policy.md, ambang stok rendah standar adalah 10 "
        "unit per produk. Stok Kopi Susu saat ini 24 unit, jadi masih di "
        "atas ambang.\n\nSumber: inventory_policy.md — Kebijakan Stok Rendah."
    )
    agent, glm = make_agent(
        [
            tool_body(
                [
                    ("search_knowledge_base", {"query": "ambang stok rendah"}),
                    ("check_stock", {"product_name": "Kopi Susu"}),
                ]
            ),
            text_body(reply),
        ]
    )

    response = post_chat(
        agent, "Apa aturan stok rendah dan berapa stok Kopi Susu sekarang?"
    )

    assert response.status_code == 200
    assert "24" in response.json()["response"]
    assert set(executed_tool_names(glm)) == {"search_knowledge_base", "check_stock"}
    joined = json.dumps(tool_results_of(glm), ensure_ascii=False)
    assert "inventory_policy.md" in joined  # policy from the knowledge base
    assert '"stock": 24' in joined  # live number from PostgreSQL
    reset_mock_data()


def test_knowledge_tool_advertised_in_chat(rag_service) -> None:
    """The registry (hence the GLM payload) includes the RAG tool."""
    agent, glm = make_agent([text_body("oke")])
    response = post_chat(agent, "halo")

    assert response.status_code == 200
    advertised = [tool["name"] for tool in glm.payloads[0]["tools"]]
    assert "search_knowledge_base" in advertised
