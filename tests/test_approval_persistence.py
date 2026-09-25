"""Persistent human-in-the-loop approval tests (Step 13).

Covers the PostgreSQL-backed approval store beyond the behavioral
suite in ``test_approval.py``: repository CRUD, state transitions and
their guards, durability across engine/session recreation (the testable
equivalent of a backend restart), API exposure of the new persisted
fields, the authorization boundary (an approval record — pending or
approved — never executes anything), and the guarded atomic decision
that makes two competing decide attempts impossible.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.approval import (
    ApprovalError,
    ApprovalNotPendingError,
    approve,
    create_pending_approval,
    get_approval,
    get_pending_approval,
    list_pending,
    reject,
    reset_approvals,
)
from app.approval import repository
from app.db.database import dispose_engines, session_scope
from app.main import app
from app.tools.business_tools import get_sales_report, reset_mock_data


@pytest.fixture(autouse=True)
def clean_stores(test_database):
    """Start every test with empty approval + seeded business stores."""
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
# A. Repository CRUD
# ---------------------------------------------------------------------------


def test_repository_insert_find_and_pending_list() -> None:
    with session_scope() as session:
        repository.insert_approval(
            session,
            approval_id="apr-crud0001",
            action="refund_order",
            requested_by="agent",
            payload={"order_id": "ORD-0001"},
            created_at=datetime(2026, 9, 25, 10, 0, 0),
        )
        found = repository.find_approval(session, "apr-crud0001")
        assert found is not None
        assert found.action == "refund_order"
        assert found.payload == {"order_id": "ORD-0001"}
        assert found.status == "pending"
        assert found.decided_at is None
        assert found.decision_reason is None
        assert repository.find_approval(session, "apr-missing") is None
        assert [r.approval_id for r in repository.pending_approvals(session)] == [
            "apr-crud0001"
        ]


def test_repository_rows_leave_pending_list_once_decided() -> None:
    first = make_approval()
    second = make_approval(action="cancel_order", payload={"order_id": "ORD-0002"})

    assert [a.approval_id for a in list_pending()] == [
        first.approval_id,
        second.approval_id,
    ]
    approve(first.approval_id)
    assert [a.approval_id for a in list_pending()] == [second.approval_id]


# ---------------------------------------------------------------------------
# B. Valid state transitions (pending -> approved / pending -> rejected)
# ---------------------------------------------------------------------------


def test_transition_pending_to_approved_persists_decision() -> None:
    approval = make_approval()

    decided = approve(approval.approval_id)

    assert decided.status == "approved"
    assert decided.decided_at
    assert decided.decision_reason is None
    stored = get_approval(approval.approval_id)
    assert stored.status == "approved"
    assert stored.decided_at == decided.decided_at
    assert get_pending_approval(approval.approval_id) is None


def test_transition_pending_to_rejected_persists_reason() -> None:
    approval = make_approval()

    decided = reject(approval.approval_id, reason="Amount exceeds policy limit.")

    assert decided.status == "rejected"
    assert decided.decision_reason == "Amount exceeds policy limit."
    assert get_approval(approval.approval_id).decision_reason == (
        "Amount exceeds policy limit."
    )


# ---------------------------------------------------------------------------
# C. Invalid transitions are refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("second,expected", [("approve", "rejected"), ("reject", "rejected")])
def test_decided_approval_cannot_be_decided_again(second: str, expected: str) -> None:
    approval = make_approval()
    reject(approval.approval_id)

    with pytest.raises(ApprovalNotPendingError):
        (approve if second == "approve" else reject)(approval.approval_id)
    assert get_approval(approval.approval_id).status == expected


@pytest.mark.parametrize("second", ["approve", "reject"])
def test_approved_approval_cannot_be_decided_again(second: str) -> None:
    approval = make_approval()
    approve(approval.approval_id)

    with pytest.raises(ApprovalNotPendingError):
        (approve if second == "approve" else reject)(approval.approval_id)
    assert get_approval(approval.approval_id).status == "approved"


def test_overlong_reason_is_rejected_before_touching_the_store() -> None:
    approval = make_approval()
    with pytest.raises(ApprovalError):
        reject(approval.approval_id, reason="x" * 501)
    assert get_approval(approval.approval_id).status == "pending"


# ---------------------------------------------------------------------------
# D. Persistence across engine/session recreation
# ---------------------------------------------------------------------------


def test_approval_survives_engine_dispose_and_recreate() -> None:
    created = make_approval()
    approved = approve(created.approval_id)  # snapshot AFTER the decision
    pending = make_approval(action="cancel_order", payload={"order_id": "ORD-0002"})

    dispose_engines()  # close every pooled connection, drop cached engines

    # Fresh engines/sessions re-read the rows from PostgreSQL.
    assert get_approval(created.approval_id) == approved
    assert get_approval(created.approval_id).status == "approved"
    assert get_approval(pending.approval_id) == pending
    assert [a.approval_id for a in list_pending()] == [pending.approval_id]


# ---------------------------------------------------------------------------
# E. Restart-like behavior: state lives in the DB, not the process
# ---------------------------------------------------------------------------


def test_pending_approvals_survive_backend_restart_simulation() -> None:
    """The manager holds no module-level state; the DB is the source.

    The in-memory store of Step 8 would be empty here. Disposing the
    engines and serving a request from a freshly created app instance
    is the testable slice of "container restarted".
    """
    before = make_approval()

    dispose_engines()
    with TestClient(app) as client:  # fresh ASGI lifespan, fresh sessions
        body = client.get("/api/approvals").json()

    assert body["count"] == 1
    assert body["approvals"][0]["approval_id"] == before.approval_id
    assert body["approvals"][0]["status"] == "pending"


# ---------------------------------------------------------------------------
# F. API: persisted fields and errors
# ---------------------------------------------------------------------------


def test_api_reject_records_reason_in_response() -> None:
    approval = make_approval()

    with TestClient(app) as client:
        response = client.post(
            f"/api/approvals/{approval.approval_id}/reject",
            params={"reason": "Duplicate request."},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected"
    assert body["decision_reason"] == "Duplicate request."
    assert body["decided_at"]


def test_api_approve_accepts_optional_reason() -> None:
    approval = make_approval()

    with TestClient(app) as client:
        response = client.post(
            f"/api/approvals/{approval.approval_id}/approve",
            params={"reason": "Verified with the customer by phone."},
        )

    assert response.status_code == 200
    assert response.json()["decision_reason"] == "Verified with the customer by phone."


def test_api_pending_list_exposes_decision_reason_field() -> None:
    make_approval()

    with TestClient(app) as client:
        body = client.get("/api/approvals").json()

    assert body["count"] == 1
    assert body["approvals"][0]["decision_reason"] is None


def test_api_overlong_reason_returns_422() -> None:
    approval = make_approval()

    with TestClient(app) as client:
        response = client.post(
            f"/api/approvals/{approval.approval_id}/reject",
            params={"reason": "x" * 501},
        )

    assert response.status_code == 422
    assert get_approval(approval.approval_id).status == "pending"


# ---------------------------------------------------------------------------
# G. Security / authorization boundary
# ---------------------------------------------------------------------------


def test_approved_approval_executes_nothing() -> None:
    """Approving records the decision only — no business side effects."""
    report_before = get_sales_report("monthly")
    approval = make_approval(payload={"order_id": "ORD-0001", "amount": 36000})

    decided = approve(approval.approval_id)

    assert decided.status == "approved"
    # No refund happened: order count, and therefore revenue, unchanged.
    report_after = get_sales_report("monthly")
    assert report_after["total_orders"] == report_before["total_orders"]
    assert report_after["total_revenue"] == report_before["total_revenue"]
    # And the sensitive "tool" still does not exist to dispatch to.
    from app.tools.registry import get_tool

    assert get_tool("refund_order") is None


def test_pending_approval_does_not_execute_anything_either() -> None:
    make_approval(payload={"order_id": "ORD-0001", "amount": 36000})

    report = get_sales_report("monthly")

    assert report["total_orders"] == 4  # seeded value, untouched


def test_approval_ids_cannot_be_reused_after_decision() -> None:
    approval = make_approval()
    approve(approval.approval_id)

    # Not pending anymore, so nothing can treat it as an open request…
    assert get_pending_approval(approval.approval_id) is None
    # …and every further decision attempt fails without changing state.
    with pytest.raises(ApprovalNotPendingError):
        reject(approval.approval_id)
    assert get_approval(approval.approval_id).status == "approved"


def test_agent_created_approval_is_persisted_not_executed() -> None:
    from dataclasses import dataclass

    @dataclass
    class FakeGLMResponse:
        text: str = ""
        tool_name: str | None = None
        tool_input: dict[str, Any] | None = None

    class FakeGLMClient:
        def __init__(self, responses):
            self._responses = list(responses)

        async def complete_with_tools(self, messages, *, system, tools):
            return self._responses.pop(0)

    from app.agent.agent import Agent

    fake = FakeGLMClient(
        [
            FakeGLMResponse(
                tool_name="refund_order",
                tool_input={"order_id": "ORD-0001", "amount": 36000},
            ),
            FakeGLMResponse(text="Refund menunggu persetujuan manusia."),
        ]
    )

    dispose_engines()  # any cached state is gone; only PostgreSQL knows
    asyncio.run(Agent(fake, tools={}, schemas=[]).run("Refund ORD-0001"))

    pending = list_pending()
    assert len(pending) == 1
    assert pending[0].action == "refund_order"
    assert pending[0].payload == {"order_id": "ORD-0001", "amount": 36000}
    assert get_sales_report("monthly")["total_orders"] == 4


# ---------------------------------------------------------------------------
# H. Concurrency: competing decisions
# ---------------------------------------------------------------------------


def test_guarded_update_lets_exactly_one_competing_decision_win() -> None:
    """Two guarded UPDATEs race; the second must change zero rows."""
    approval = make_approval()
    now = datetime.now()

    with session_scope() as first:
        changed_first = repository.decide_approval(
            first,
            approval.approval_id,
            expected_status="pending",
            new_status="approved",
            decided_at=now,
        )
    with session_scope() as second:  # a competing request, same moment
        changed_second = repository.decide_approval(
            second,
            approval.approval_id,
            expected_status="pending",
            new_status="rejected",
            decided_at=now,
        )

    assert (changed_first, changed_second) == (1, 0)
    stored = get_approval(approval.approval_id)
    assert stored.status == "approved"  # the first decision stands
    assert stored.decided_at == now.isoformat(timespec="seconds")
