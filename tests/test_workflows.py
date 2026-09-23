"""Step 5 workflow tests: multi-step business operations through /api/chat.

Every test scripts the GLM HTTP exchange with an in-memory transport
(no real GLM API, no network) while the real registry tools run against
the seeded mock store. This verifies the complete workflow chain --
user message -> /api/chat -> Agent -> GLM -> tool_use -> business tool
-> tool_result -> GLM -> final reply -> ChatResponse -- including
state changes (or their absence) in the store.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agent.agent import AGENT_SYSTEM_PROMPT, DibantuAgent
from app.api.routes import get_agent
from app.config import Settings
from app.main import app
from app.tools import (
    check_stock,
    get_sales_report,
    reset_mock_data,
)


@pytest.fixture(autouse=True)
def fresh_store(test_database):
    """Start every test from the seeded store (PostgreSQL test DB)."""
    reset_mock_data()
    yield


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
    return tools_body([(name, None, tool_input)])


def tools_body(
    calls: list[tuple[str, str | None, dict[str, Any]]],
) -> dict[str, Any]:
    """Build a response body with one tool_use block per call.

    Each call is a ``(name, id, input)`` triple; ``id`` may be None to
    omit the API-style tool_use id (the agent then synthesizes one).
    """
    blocks = []
    for name, tool_id, tool_input in calls:
        block: dict[str, Any] = {"type": "tool_use", "name": name, "input": tool_input}
        if tool_id is not None:
            block["id"] = tool_id
        blocks.append(block)
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": blocks,
        "stop_reason": "tool_use",
    }


def workflow_agent(
    bodies: list[dict[str, Any]],
) -> tuple[DibantuAgent, ScriptedGLM]:
    """Build a DibantuAgent answered by a scripted GLM transport."""
    glm = ScriptedGLM(bodies)
    agent = DibantuAgent(
        make_settings(), transport=httpx.MockTransport(glm.handler)
    )
    return agent, glm


def run_chat(agent: DibantuAgent, message: str) -> dict[str, Any]:
    """POST a chat message with the agent dependency overridden."""
    app.dependency_overrides[get_agent] = lambda: agent
    try:
        with TestClient(app) as client:
            response = client.post("/api/chat", json={"message": message})
        return {"status": response.status_code, "body": response.json()}
    finally:
        app.dependency_overrides.pop(get_agent, None)


def tool_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every parsed tool_result block of one GLM payload."""
    results = []
    for message in payload["messages"]:
        content = message["content"]
        if isinstance(content, list):
            results.extend(
                json.loads(block["content"])
                for block in content
                if block.get("type") == "tool_result"
            )
    return results


# ---------------------------------------------------------------------------
# 1. Stock check workflow
# ---------------------------------------------------------------------------


def test_stock_check_workflow_answers_from_real_data() -> None:
    """"Cukup untuk 20 pesanan?" is answered from check_stock output only."""
    agent, glm = workflow_agent(
        [
            tool_body("check_stock", {"product_name": "Kopi Susu"}),
            text_body(
                "Stok Kopi Susu 24 unit — cukup untuk 20 pesanan "
                "(sisa 4 unit, harga Rp18.000/pesanan)."
            ),
        ]
    )

    result = run_chat(agent, "Apakah stok kopi susu cukup untuk 20 pesanan?")

    assert result["status"] == 200
    assert "cukup" in result["body"]["response"]
    assert "24" in result["body"]["response"]
    # The reply is grounded: the tool result carried the real seeded stock.
    assert tool_results(glm.payloads[1])[0] == {
        "success": True,
        "product_name": "Kopi Susu",
        "stock": 24,
        "price": 18000,
        "in_stock": True,
        "low_stock": False,
    }
    assert check_stock("Kopi Susu")["stock"] == 24  # lookups never mutate


# ---------------------------------------------------------------------------
# 2./3. Order and multi-product workflows
# ---------------------------------------------------------------------------


def test_order_workflow_identifies_customer_then_creates_order() -> None:
    """search_customer -> create_order runs in sequence with real effects."""
    agent, glm = workflow_agent(
        [
            tool_body("search_customer", {"name": "Budi"}),
            tool_body(
                "create_order",
                {
                    "customer_name": "Budi Santoso",
                    "items": [{"product_name": "Kopi Susu", "quantity": 3}],
                },
            ),
            text_body(
                "Pesanan ORD-0005 dibuat untuk Budi Santoso: 3 Kopi Susu, "
                "total Rp54.000."
            ),
        ]
    )

    result = run_chat(agent, "Budi pesan 3 kopi susu.")

    assert result["status"] == 200
    assert "ORD-0005" in result["body"]["response"]
    assert len(glm.payloads) == 3
    # Step 1 found the existing customer and propagated the record.
    customer = tool_results(glm.payloads[1])[-1]
    assert customer["count"] == 1
    assert customer["customers"][0]["name"] == "Budi Santoso"
    # Step 2 created exactly one order; the tool result proves it.
    order = tool_results(glm.payloads[2])[-1]
    assert order["success"] is True
    assert order["new_customer"] is False
    assert order["order"]["order_id"] == "ORD-0005"
    assert order["order"]["total_price"] == 54000
    # create_order deducts stock itself.
    assert check_stock("Kopi Susu")["stock"] == 21


def test_multi_product_workflow_checks_each_product_then_orders_once() -> None:
    """Three stock checks, then ONE create_order carrying every line item."""
    agent, glm = workflow_agent(
        [
            tool_body("check_stock", {"product_name": "Kopi Susu"}),
            tool_body("check_stock", {"product_name": "Croissant"}),
            tool_body("check_stock", {"product_name": "Americano"}),
            tool_body(
                "create_order",
                {
                    "customer_name": "Budi Santoso",
                    "items": [
                        {"product_name": "Kopi Susu", "quantity": 3},
                        {"product_name": "Croissant", "quantity": 2},
                        {"product_name": "Americano", "quantity": 1},
                    ],
                },
            ),
            text_body(
                "Semua stok cukup. ORD-0005 dibuat: 3 Kopi Susu, 2 "
                "Croissant, 1 Americano — total Rp126.000."
            ),
        ]
    )

    result = run_chat(
        agent, "Budi pesan 3 kopi susu, 2 croissant, dan 1 americano. Cek stoknya dulu."
    )

    assert result["status"] == 200
    assert "Rp126.000" in result["body"]["response"]
    # 3 checks + 1 order = 4 tool executions inside the cap of 5.
    assert len(glm.payloads) == 5
    # Every check propagated its real stock into the next turn.
    stocks = [tool_results(p)[-1] for p in glm.payloads[1:4]]
    assert [(s["product_name"], s["stock"]) for s in stocks] == [
        ("Kopi Susu", 24),
        ("Croissant", 8),
        ("Americano", 40),
    ]
    # One order call carried all three items and the correct total.
    order = tool_results(glm.payloads[4])[-1]
    assert order["success"] is True
    assert len(order["order"]["items"]) == 3
    assert order["order"]["total_price"] == 126000
    # Stock fell exactly by the ordered quantities.
    assert check_stock("Kopi Susu")["stock"] == 21
    assert check_stock("Croissant")["stock"] == 6
    assert check_stock("Americano")["stock"] == 39


def test_multi_product_workflow_with_parallel_tool_calls() -> None:
    """GLM may batch the three stock checks in one turn; all must run."""
    agent, glm = workflow_agent(
        [
            tools_body(
                [
                    ("check_stock", "toolu_a1", {"product_name": "Kopi Susu"}),
                    ("check_stock", "toolu_a2", {"product_name": "Croissant"}),
                    ("check_stock", "toolu_a3", {"product_name": "Americano"}),
                ]
            ),
            tool_body(
                "create_order",
                {
                    "customer_name": "Budi Santoso",
                    "items": [
                        {"product_name": "Kopi Susu", "quantity": 3},
                        {"product_name": "Croissant", "quantity": 2},
                        {"product_name": "Americano", "quantity": 1},
                    ],
                },
            ),
            text_body("Stok semua produk cukup; ORD-0005 dibuat, total Rp126.000."),
        ]
    )

    result = run_chat(
        agent, "Budi pesan 3 kopi susu, 2 croissant, dan 1 americano. Cek stoknya dulu."
    )

    assert result["status"] == 200
    # 3 parallel checks + 1 order = 4 executions; 3 GLM turns total.
    assert len(glm.payloads) == 3
    second_messages = glm.payloads[1]["messages"]
    # The echoed assistant turn keeps the API's tool_use ids and order.
    echoed = [
        block for block in second_messages[1]["content"] if block["type"] == "tool_use"
    ]
    assert [block["id"] for block in echoed] == ["toolu_a1", "toolu_a2", "toolu_a3"]
    # One user message carries all three matching tool_results.
    results = second_messages[2]["content"]
    assert [block["tool_use_id"] for block in results] == [
        "toolu_a1",
        "toolu_a2",
        "toolu_a3",
    ]
    assert [json.loads(block["content"])["stock"] for block in results] == [24, 8, 40]
    # The order still went through once, with every item.
    order = tool_results(glm.payloads[2])[-1]
    assert order["success"] is True
    assert order["order"]["total_price"] == 126000


# ---------------------------------------------------------------------------
# 4. Insufficient stock workflow
# ---------------------------------------------------------------------------


def test_insufficient_stock_workflow_never_orders() -> None:
    """The agent checks stock, sees the shortage, and does not order."""
    agent, glm = workflow_agent(
        [
            tool_body("check_stock", {"product_name": "Kopi Susu"}),
            text_body(
                "Maaf, stok Kopi Susu hanya 24 unit — tidak cukup untuk 100 "
                "pesanan (kurang 76). Mau pesan 24 atau kurang?"
            ),
        ]
    )

    result = run_chat(agent, "Budi pesan 100 kopi susu.")

    assert result["status"] == 200
    assert "tidak cukup" in result["body"]["response"]
    assert "24" in result["body"]["response"]
    # No create_order was ever requested (messages only; every payload
    # advertises the create_order schema, which is fine).
    for payload in glm.payloads:
        assert "create_order" not in json.dumps(payload["messages"])
    assert len(glm.payloads) == 2
    # Store untouched: stock intact and still only the 4 seeded orders.
    assert check_stock("Kopi Susu")["stock"] == 24
    assert get_sales_report("monthly")["total_orders"] == 4


def test_rejected_order_leaves_stock_untouched() -> None:
    """create_order's atomic validation blocks partial mutation."""
    agent, glm = workflow_agent(
        [
            tool_body(
                "create_order",
                {
                    "customer_name": "Budi Santoso",
                    "items": [
                        {"product_name": "Croissant", "quantity": 2},
                        {"product_name": "Kopi Susu", "quantity": 100},
                    ],
                },
            ),
            text_body(
                "Pesanan gagal: stok Kopi Susu tidak cukup (diminta 100, "
                "tersedia 24). Stok tidak berubah."
            ),
        ]
    )

    result = run_chat(agent, "Budi pesan 2 croissant dan 100 kopi susu.")

    assert result["status"] == 200
    assert "gagal" in result["body"]["response"]
    # The tool's failure is visible to the agent, not swallowed.
    failure = tool_results(glm.payloads[1])[0]
    assert failure["success"] is False
    assert failure["error"] == "INSUFFICIENT_STOCK"
    assert failure["available"] == 24
    # Atomicity: neither the shortaged product nor the in-stock one moved.
    assert check_stock("Kopi Susu")["stock"] == 24
    assert check_stock("Croissant")["stock"] == 8
    assert get_sales_report("monthly")["total_orders"] == 4


# ---------------------------------------------------------------------------
# 5. Sales report workflow
# ---------------------------------------------------------------------------


def test_sales_report_workflow_summarizes_real_data() -> None:
    """"Penjualan hari ini?" is answered from get_sales_report output."""
    agent, glm = workflow_agent(
        [
            tool_body("get_sales_report", {"period": "daily"}),
            text_body(
                "Penjualan hari ini: 1 pesanan, 2 item terjual, pendapatan "
                "Rp36.000. Produk terlaris: Kopi Susu."
            ),
        ]
    )

    result = run_chat(agent, "Berapa total penjualan hari ini?")

    assert result["status"] == 200
    assert "Rp36.000" in result["body"]["response"]
    # The summary matches the actual seeded daily window.
    report = tool_results(glm.payloads[1])[0]
    assert report == {
        "success": True,
        "period": "daily",
        "window_days": 1,
        "total_orders": 1,
        "total_items_sold": 2,
        "total_revenue": 36000,
        "products_sold": {"Kopi Susu": 2},
        "top_product": "Kopi Susu",
    }


# ---------------------------------------------------------------------------
# 6. Low stock workflow
# ---------------------------------------------------------------------------


def test_low_stock_workflow_lists_returned_products() -> None:
    """"Stok menipis?" lists exactly the products get_low_stock returned."""
    agent, glm = workflow_agent(
        [
            tool_body("get_low_stock", {"threshold": 16}),
            text_body(
                "2 produk stoknya menipis: Croissant (8 unit) dan "
                "Matcha Latte (15 unit)."
            ),
        ]
    )

    result = run_chat(agent, "Produk apa saja yang stoknya mulai menipis?")

    assert result["status"] == 200
    assert "Croissant" in result["body"]["response"]
    low = tool_results(glm.payloads[1])[0]
    assert low["success"] is True
    assert [item["product_name"] for item in low["products"]] == [
        "Croissant",
        "Matcha Latte",
    ]
    assert [item["stock"] for item in low["products"]] == [8, 15]


# ---------------------------------------------------------------------------
# Cross-cutting: propagation, sequence shape, prompt playbooks
# ---------------------------------------------------------------------------


def test_tool_results_propagate_to_every_later_turn() -> None:
    """Every intermediate result is still present in the final GLM call."""
    agent, glm = workflow_agent(
        [
            tool_body("search_customer", {"name": "Budi"}),
            tool_body("check_stock", {"product_name": "Kopi Susu"}),
            tool_body(
                "create_order",
                {
                    "customer_name": "Budi Santoso",
                    "items": [{"product_name": "Kopi Susu", "quantity": 1}],
                },
            ),
            text_body("Selesai: ORD-0005 untuk Budi Santoso, 1 Kopi Susu."),
        ]
    )

    result = run_chat(agent, "Cari customer Budi, cek stok kopi susu, lalu buat pesanan 1 kopi susu.")

    assert result["status"] == 200
    final_messages = glm.payloads[3]["messages"]
    assert [message["role"] for message in final_messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    # All three tool results travelled with the conversation.
    propagated = tool_results(glm.payloads[3])
    assert propagated[0]["customers"][0]["name"] == "Budi Santoso"
    assert propagated[1]["stock"] == 24
    assert propagated[2]["order"]["order_id"] == "ORD-0005"
    assert check_stock("Kopi Susu")["stock"] == 23


def test_workflow_prompt_contains_playbooks() -> None:
    """The Step 5 prompt encodes the order/shortage/report playbooks."""
    prompt = AGENT_SYSTEM_PROMPT.lower()
    assert "check_stock" in prompt
    assert "search_customer" in prompt
    assert "one create_order call" in prompt
    assert "insufficient stock" in prompt
    assert "get_sales_report" in prompt
    assert "get_low_stock" in prompt
