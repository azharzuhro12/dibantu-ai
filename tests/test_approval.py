"""Tests for the human-in-the-loop approval system (Step 8 → Step 13).

Three groups: the approval manager (now backed by the ``approvals``
PostgreSQL table via the scratch test database), the central
interception inside the Agent tool-dispatch path (driven by a fake GLM
client, no network), and the approval API endpoints. The mock store
and the approval store are reset around every test.

Step 13 note: the ``Approval`` objects returned by the manager are now
immutable snapshots of database rows, so tests assert field equality
(``==``) instead of object identity (``is``), and re-fetch a record to
observe a decision made after the snapshot was taken.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.agent.agent import AGENT_SYSTEM_PROMPT, Agent
from app.approval import (
    ApprovalError,
    ApprovalNotFoundError,
    SENSITIVE_ACTIONS,
    approve,
    create_pending_approval,
    get_approval,
    get_pending_approval,
    is_sensitive_action,
    list_pending,
    reject,
    reset_approvals,
)
from app.main import app
from app.tools.business_tools import get_sales_report, reset_mock_data
from app.tools.registry import get_tool


@pytest.fixture(autouse=True)
def clean_stores(test_database):
    """Start every test with an empty approval store and seeded store."""
    reset_approvals()
    reset_mock_data()
    yield
    reset_approvals()
    reset_mock_data()


def make_approval(**overrides: Any):
    """Create a pending refund approval with sensible defaults."""
    params: dict[str, Any] = {
        "action": "refund_order",
        "requested_by": "agent",
        "payload": {"order_id": "ORD-0001", "amount": 36000},
    }
    params.update(overrides)
    return create_pending_approval(**params)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


def test_sensitive_actions_are_defined() -> None:
    assert SENSITIVE_ACTIONS == (
        "refund_order",
        "cancel_order",
        "bulk_stock_update",
    )
    assert is_sensitive_action("refund_order")
    assert not is_sensitive_action("create_order")
    assert not is_sensitive_action("teleport")


def test_create_and_get_approval() -> None:
    approval = make_approval()

    assert approval.approval_id.startswith("apr-")
    assert approval.action == "refund_order"
    assert approval.requested_by == "agent"
    assert approval.payload == {"order_id": "ORD-0001", "amount": 36000}
    assert approval.status == "pending"
    assert approval.created_at
    assert approval.decided_at is None
    # Snapshots now come from the database: same record, field-equal.
    assert get_approval(approval.approval_id) == approval
    assert get_pending_approval(approval.approval_id) == approval


def test_create_rejects_non_sensitive_actions() -> None:
    with pytest.raises(ApprovalError):
        create_pending_approval(
            action="create_order", requested_by="agent", payload={}
        )


def test_list_pending_and_get_after_decision() -> None:
    first = make_approval()
    second = make_approval(action="cancel_order", payload={"order_id": "ORD-0002"})

    pending = list_pending()
    assert [a.approval_id for a in pending] == [
        first.approval_id,
        second.approval_id,
    ]

    approve(first.approval_id)
    # Still retrievable by id, but no longer pending.
    assert get_approval(first.approval_id).status == "approved"
    assert get_pending_approval(first.approval_id) is None
    assert [a.approval_id for a in list_pending()] == [second.approval_id]


def test_approve_pending_approval() -> None:
    approval = make_approval()

    decided = approve(approval.approval_id)

    assert decided.status == "approved"
    assert decided.decided_at
    assert decided.payload == approval.payload  # approved payload returned


def test_reject_pending_approval() -> None:
    approval = make_approval()

    decided = reject(approval.approval_id)

    assert decided.status == "rejected"
    assert decided.decided_at


def test_cannot_approve_rejected_approval() -> None:
    approval = make_approval()
    reject(approval.approval_id)

    with pytest.raises(ApprovalError):
        approve(approval.approval_id)
    # Decision unchanged in the store (re-fetch: snapshots are immutable).
    assert get_approval(approval.approval_id).status == "rejected"


def test_cannot_reject_approved_approval() -> None:
    approval = make_approval()
    approve(approval.approval_id)

    with pytest.raises(ApprovalError):
        reject(approval.approval_id)
    assert get_approval(approval.approval_id).status == "approved"


def test_unknown_approval_not_found() -> None:
    assert get_approval("apr-does-not-exist") is None
    assert get_pending_approval("apr-does-not-exist") is None
    with pytest.raises(ApprovalNotFoundError):
        approve("apr-does-not-exist")
    with pytest.raises(ApprovalNotFoundError):
        reject("apr-does-not-exist")


# ---------------------------------------------------------------------------
# Agent integration (fake GLM client, offline)
# ---------------------------------------------------------------------------


@dataclass
class FakeGLMResponse:
    """Minimal stand-in for ``GLMResponse`` as consumed by ``Agent``."""

    text: str = ""
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None


class FakeGLMClient:
    """Test double yielding queued responses and recording every call."""

    def __init__(self, responses: list[FakeGLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str,
        tools: list[dict[str, Any]],
    ) -> FakeGLMResponse:
        self.calls.append(
            {"messages": [dict(message) for message in messages], "system": system}
        )
        return self._responses.pop(0)


def test_sensitive_action_creates_approval_not_execution() -> None:
    fake = FakeGLMClient(
        [
            FakeGLMResponse(
                tool_name="refund_order",
                tool_input={"order_id": "ORD-0001", "amount": 36000},
            ),
            FakeGLMResponse(
                text="Refund ORD-0001 butuh persetujuan manusia dulu."
            ),
        ]
    )
    agent = Agent(fake, tools={}, schemas=[])

    result = asyncio.run(agent.run("Refund pesanan ORD-0001 dong"))

    assert result == "Refund ORD-0001 butuh persetujuan manusia dulu."
    # Exactly one approval exists, pending, with the requested payload.
    pending = list_pending()
    assert len(pending) == 1
    approval = pending[0]
    assert approval.action == "refund_order"
    assert approval.requested_by == "agent"
    assert approval.payload == {"order_id": "ORD-0001", "amount": 36000}
    assert approval.status == "pending"
    # Nothing was executed: not a real tool, and the store is untouched.
    assert get_tool("refund_order") is None
    assert get_sales_report("monthly")["total_orders"] == 4


def test_approval_result_is_visible_to_the_agent() -> None:
    fake = FakeGLMClient(
        [
            FakeGLMResponse(
                tool_name="cancel_order", tool_input={"order_id": "ORD-0003"}
            ),
            FakeGLMResponse(text="Pembatalan menunggu persetujuan."),
        ]
    )
    agent = Agent(fake, tools={}, schemas=[])

    asyncio.run(agent.run("Batalkan pesanan ORD-0003"))

    approval = list_pending()[0]
    tool_result = fake.calls[1]["messages"][2]["content"][0]
    assert tool_result["type"] == "tool_result"
    body = json.loads(tool_result["content"])
    assert body == {
        "approval_required": True,
        "approval_id": approval.approval_id,
        "action": "cancel_order",
        "status": "pending",
        "message": body["message"],  # guidance text, checked below
    }
    assert "NOT been executed" in body["message"]
    assert approval.approval_id in body["message"]


def test_every_sensitive_action_is_intercepted() -> None:
    for action, payload in [
        ("refund_order", {"order_id": "ORD-0001"}),
        ("cancel_order", {"order_id": "ORD-0002"}),
        ("bulk_stock_update", {"updates": [{"product_name": "Kopi Susu", "quantity_change": 50}]}),
    ]:
        reset_approvals()
        fake = FakeGLMClient(
            [
                FakeGLMResponse(tool_name=action, tool_input=payload),
                FakeGLMResponse(text="Menunggu persetujuan manusia."),
            ]
        )
        agent = Agent(fake, tools={}, schemas=[])

        result = asyncio.run(agent.run(f"Lakukan {action}"))

        assert result == "Menunggu persetujuan manusia."
        assert [a.action for a in list_pending()] == [action]
        body = json.loads(fake.calls[1]["messages"][2]["content"][0]["content"])
        assert body["approval_required"] is True
        assert body["action"] == action


def test_normal_tool_still_executes_normally() -> None:
    fake = FakeGLMClient(
        [
            FakeGLMResponse(
                tool_name="check_stock", tool_input={"product_name": "Kopi Susu"}
            ),
            FakeGLMResponse(text="Stok Kopi Susu 24 unit."),
        ]
    )
    agent = Agent(fake, schemas=[])  # real registry tools (tools not overridden)

    result = asyncio.run(agent.run("Berapa stok kopi susu?"))

    assert result == "Stok Kopi Susu 24 unit."
    assert list_pending() == []  # no approval created for normal tools
    tool_result = json.loads(fake.calls[1]["messages"][2]["content"][0]["content"])
    assert tool_result == {
        "success": True,
        "product_name": "Kopi Susu",
        "stock": 24,
        "price": 18000,
        "in_stock": True,
        "low_stock": False,
    }


def test_system_prompt_explains_approval_results() -> None:
    prompt = AGENT_SYSTEM_PROMPT.lower()
    assert "approval_required" in prompt
    assert "refund_order" in prompt
    assert "cancel_order" in prompt
    assert "bulk_stock_update" in prompt


# ---------------------------------------------------------------------------
# Approval API
# ---------------------------------------------------------------------------


def api_get_approvals() -> Any:
    with TestClient(app) as client:
        return client.get("/api/approvals")


def api_decide(approval_id: str, decision: str) -> Any:
    with TestClient(app) as client:
        return client.post(f"/api/approvals/{approval_id}/{decision}")


def test_api_lists_pending_approvals() -> None:
    assert api_get_approvals().json() == {"approvals": [], "count": 0}

    first = make_approval()
    second = make_approval(action="cancel_order", payload={"order_id": "ORD-0002"})

    body = api_get_approvals().json()
    assert body["count"] == 2
    assert [a["approval_id"] for a in body["approvals"]] == [
        first.approval_id,
        second.approval_id,
    ]
    entry = body["approvals"][0]
    assert entry["action"] == "refund_order"
    assert entry["status"] == "pending"
    assert entry["requested_by"] == "agent"
    assert entry["payload"] == {"order_id": "ORD-0001", "amount": 36000}
    assert entry["created_at"]
    assert entry["decided_at"] is None


def test_api_approve_pending_approval() -> None:
    approval = make_approval()

    response = api_decide(approval.approval_id, "approve")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "approved"
    assert body["approval_id"] == approval.approval_id
    assert body["payload"] == {"order_id": "ORD-0001", "amount": 36000}
    assert body["decided_at"]
    # Approved approvals leave the pending list.
    assert api_get_approvals().json()["count"] == 0


def test_api_reject_pending_approval() -> None:
    approval = make_approval()

    response = api_decide(approval.approval_id, "reject")

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert api_get_approvals().json()["count"] == 0


def test_api_unknown_approval_returns_404() -> None:
    assert api_decide("apr-nope", "approve").status_code == 404
    assert api_decide("apr-nope", "reject").status_code == 404


def test_api_cannot_decide_twice() -> None:
    approval = make_approval()

    assert api_decide(approval.approval_id, "approve").status_code == 200
    duplicate = api_decide(approval.approval_id, "approve")
    assert duplicate.status_code == 409
    assert "already approved" in duplicate.json()["detail"]
    assert api_decide(approval.approval_id, "reject").status_code == 409


def test_api_reject_then_approve_conflicts() -> None:
    approval = make_approval()

    assert api_decide(approval.approval_id, "reject").status_code == 200
    assert api_decide(approval.approval_id, "approve").status_code == 409
    assert get_approval(approval.approval_id).status == "rejected"


def test_agent_created_approval_flows_through_api() -> None:
    """Approval created inside the Agent loop is visible to the API."""
    fake = FakeGLMClient(
        [
            FakeGLMResponse(
                tool_name="refund_order", tool_input={"order_id": "ORD-0004"}
            ),
            FakeGLMResponse(text="Refund menunggu persetujuan."),
        ]
    )
    asyncio.run(Agent(fake, tools={}, schemas=[]).run("Refund ORD-0004"))

    listed = api_get_approvals().json()
    assert listed["count"] == 1
    approval_id = listed["approvals"][0]["approval_id"]

    approved = api_decide(approval_id, "approve")

    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"
    assert approved.json()["payload"] == {"order_id": "ORD-0004"}
