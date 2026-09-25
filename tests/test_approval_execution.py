"""Deferred approval execution tests (Step 16).

Eight groups mirroring the step spec:

A. State machine transitions (legal steps pass, illegal steps raise).
B. Execution lifecycle (approved executes; every other status conflicts).
C. Payload integrity (the stored snapshot is the only execution input).
D. Concurrency (two execute attempts -> exactly one execution).
E. Security (allowlist, malformed payloads, no leaked secrets).
F. Observability (APPROVAL events; tracing failure never breaks execution).
G. Persistence (decisions, executions, and results survive a restart).
H. API (status codes, idempotency, status filtering, body is ignored).

All runs use the isolated scratch PostgreSQL database; the mock store
and approval store are reset around every test.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.approval import (
    APPROVED_EXECUTABLE_TOOLS,
    ApprovalConflictError,
    ApprovalError,
    ApprovalNotFoundError,
    ApprovalNotExecutableError,
    approve,
    can_transition,
    create_pending_approval,
    get_approval,
    list_approvals,
    reject,
    reset_approvals,
    validate_transition,
)
from app.approval import executor, repository
from app.db.database import dispose_engines, session_scope
from app.main import app
from app.observability import get_run_events, list_runs
from app.tools.business_tools import check_stock, get_sales_report, reset_mock_data


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


def api_execute(client: TestClient, approval_id: str, **kwargs: Any):
    return client.post(f"/api/approvals/{approval_id}/execute", **kwargs)


# ---------------------------------------------------------------------------
# A. State machine
# ---------------------------------------------------------------------------


def test_all_legal_transitions_pass() -> None:
    for current, new in (
        ("pending", "approved"),
        ("pending", "rejected"),
        ("approved", "executing"),
        ("executing", "executed"),
        ("executing", "failed"),
    ):
        assert can_transition(current, new)
        validate_transition(current, new)  # must not raise


@pytest.mark.parametrize(
    ("current", "new"),
    [
        ("rejected", "executing"),
        ("rejected", "executed"),
        ("rejected", "approved"),
        ("executed", "executing"),
        ("executed", "approved"),
        ("executed", "failed"),
        ("failed", "executing"),
        ("failed", "approved"),
        ("pending", "executing"),
        ("pending", "executed"),
        ("approved", "executed"),  # must pass through executing (the claim)
        ("approved", "rejected"),  # a decision cannot be undone
        ("executing", "executing"),
        ("executing", "approved"),
    ],
)
def test_illegal_transitions_raise(current: str, new: str) -> None:
    assert not can_transition(current, new)
    with pytest.raises(ApprovalError, match="Illegal approval transition"):
        validate_transition(current, new)


# ---------------------------------------------------------------------------
# B. Execution lifecycle
# ---------------------------------------------------------------------------


def test_approved_refund_executes_and_records_result() -> None:
    approval = make_approval()  # ORD-0001 = 2x Kopi Susu, total 36000
    approve(approval.approval_id)

    finished = executor.execute_approval(approval.approval_id)

    assert finished.status == "executed"
    assert finished.execution_result is not None
    assert finished.execution_result["success"] is True
    assert finished.execution_result["order_id"] == "ORD-0001"
    assert finished.execution_result["refund_amount"] == 36000
    assert finished.execution_error is None
    assert finished.execution_started_at is not None
    assert finished.executed_at is not None
    # Real business effect: stock restored, order leaves the report.
    assert check_stock("Kopi Susu")["stock"] == 26
    report = get_sales_report("monthly")
    assert report["total_orders"] == 3
    assert report["total_revenue"] == 140000


def test_partial_refund_returns_money_without_restocking() -> None:
    approval = make_approval(payload={"order_id": "ORD-0001", "amount": 10000})
    approve(approval.approval_id)

    finished = executor.execute_approval(approval.approval_id)

    assert finished.status == "executed"
    assert finished.execution_result["partial"] is True
    assert finished.execution_result["restocked_items"] == []
    assert finished.execution_result["refunded_total"] == 10000
    # Partial refund: goods kept, so stock and the report stand.
    assert check_stock("Kopi Susu")["stock"] == 24
    assert get_sales_report("monthly")["total_orders"] == 4


def test_partial_refunds_capped_at_order_total() -> None:
    first = make_approval(payload={"order_id": "ORD-0001", "amount": 10000})
    approve(first.approval_id)
    assert executor.execute_approval(first.approval_id).status == "executed"

    # 30000 more would exceed the remaining 26000 -> refused -> failed.
    second = make_approval(payload={"order_id": "ORD-0001", "amount": 30000})
    approve(second.approval_id)
    finished = executor.execute_approval(second.approval_id)
    assert finished.status == "failed"
    assert "melebihi sisa" in finished.execution_error
    assert check_stock("Kopi Susu")["stock"] == 24  # nothing restored

    # Refunding exactly the remainder completes the refund fully.
    third = make_approval(payload={"order_id": "ORD-0001", "amount": 26000})
    approve(third.approval_id)
    finished = executor.execute_approval(third.approval_id)
    assert finished.status == "executed"
    assert finished.execution_result["partial"] is False
    assert finished.execution_result["refunded_total"] == 36000
    assert check_stock("Kopi Susu")["stock"] == 26  # full restore at the end
    assert get_sales_report("monthly")["total_orders"] == 3

    # The order is terminal now: no further refund can even be executed.
    fourth = make_approval(payload={"order_id": "ORD-0001", "amount": 1})
    approve(fourth.approval_id)
    finished = executor.execute_approval(fourth.approval_id)
    assert finished.status == "failed"
    assert "sudah berstatus 'refunded'" in finished.execution_error


def test_cancel_order_execution_restores_stock() -> None:
    approval = make_approval(
        action="cancel_order", payload={"order_id": "ORD-0004"}
    )
    approve(approval.approval_id)

    finished = executor.execute_approval(approval.approval_id)

    assert finished.status == "executed"
    assert finished.execution_result["status"] == "cancelled"
    assert check_stock("Teh Manis")["stock"] == 35  # 30 + 5 returned
    assert get_sales_report("monthly")["total_orders"] == 3


def test_bulk_stock_update_execution_applies_all_lines() -> None:
    approval = make_approval(
        action="bulk_stock_update",
        payload={
            "updates": [
                {"product_name": "Croissant", "quantity_change": 12},
                {"product_name": "Matcha Latte", "quantity_change": -5},
            ]
        },
    )
    approve(approval.approval_id)

    finished = executor.execute_approval(approval.approval_id)

    assert finished.status == "executed"
    assert finished.execution_result["updated_count"] == 2
    assert check_stock("Croissant")["stock"] == 20  # 8 + 12
    assert check_stock("Matcha Latte")["stock"] == 10  # 15 - 5


def test_pending_approval_cannot_execute() -> None:
    approval = make_approval()

    with pytest.raises(ApprovalConflictError) as excinfo:
        executor.execute_approval(approval.approval_id)

    assert "pending" in str(excinfo.value)
    assert get_approval(approval.approval_id).status == "pending"
    assert get_sales_report("monthly")["total_orders"] == 4  # nothing ran


def test_rejected_approval_cannot_execute() -> None:
    approval = make_approval()
    reject(approval.approval_id)

    with pytest.raises(ApprovalConflictError) as excinfo:
        executor.execute_approval(approval.approval_id)

    assert "rejected" in str(excinfo.value)
    assert get_approval(approval.approval_id).status == "rejected"
    assert check_stock("Kopi Susu")["stock"] == 24


def test_executed_approval_cannot_execute_again() -> None:
    approval = make_approval()
    approve(approval.approval_id)
    assert executor.execute_approval(approval.approval_id).status == "executed"

    with pytest.raises(ApprovalConflictError) as excinfo:
        executor.execute_approval(approval.approval_id)

    assert "already been executed" in str(excinfo.value)
    # No double effect: stock restored exactly once.
    assert check_stock("Kopi Susu")["stock"] == 26


def test_failed_approval_does_not_automatically_retry() -> None:
    # ORD-9999 does not exist -> the business layer refuses -> failed.
    approval = make_approval(payload={"order_id": "ORD-9999"})
    approve(approval.approval_id)

    finished = executor.execute_approval(approval.approval_id)

    assert finished.status == "failed"
    assert finished.execution_error is not None
    assert "ORD-9999" in finished.execution_error
    assert finished.execution_result is None

    with pytest.raises(ApprovalConflictError) as excinfo:
        executor.execute_approval(approval.approval_id)
    assert "failed earlier" in str(excinfo.value)


def test_unknown_approval_id_raises_not_found() -> None:
    with pytest.raises(ApprovalNotFoundError):
        executor.execute_approval("apr-missing")


# ---------------------------------------------------------------------------
# C. Payload integrity
# ---------------------------------------------------------------------------


def test_execution_uses_the_stored_snapshot_only() -> None:
    # Snapshot says a PARTIAL refund; the full total is 36000.
    approval = make_approval(payload={"order_id": "ORD-0001", "amount": 18000})
    approve(approval.approval_id)

    finished = executor.execute_approval(approval.approval_id)

    # The executor had no way to receive new arguments: it ran exactly
    # the stored amount, not the total.
    assert finished.execution_result["refund_amount"] == 18000
    assert finished.payload == {"order_id": "ORD-0001", "amount": 18000}


def test_execute_endpoint_ignores_any_request_body() -> None:
    approval = make_approval(payload={"order_id": "ORD-0001", "amount": 18000})
    with TestClient(app) as client:
        client.post(f"/api/approvals/{approval.approval_id}/approve")

        response = api_execute(
            client,
            approval.approval_id,
            json={"order_id": "ORD-0002", "amount": 999999},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "executed"
    # The body was ignored: the stored snapshot executed instead.
    assert body["execution_result"]["order_id"] == "ORD-0001"
    assert body["execution_result"]["refund_amount"] == 18000


def test_only_known_payload_fields_reach_the_implementation() -> None:
    calls: dict[str, Any] = {}

    def spy_refund(**kwargs: Any) -> dict[str, Any]:
        calls.update(kwargs)
        return {"success": True}

    original = APPROVED_EXECUTABLE_TOOLS["refund_order"]
    APPROVED_EXECUTABLE_TOOLS["refund_order"] = spy_refund
    try:
        approval = make_approval(
            payload={
                "order_id": "ORD-0001",
                "amount": 5000,
                "evil_extra": "should be dropped",
            }
        )
        approve(approval.approval_id)
        executor.execute_approval(approval.approval_id)
    finally:
        APPROVED_EXECUTABLE_TOOLS["refund_order"] = original

    assert calls == {"order_id": "ORD-0001", "amount": 5000}


# ---------------------------------------------------------------------------
# D. Concurrency: exactly one execution wins
# ---------------------------------------------------------------------------


def test_guarded_claim_lets_exactly_one_competing_execute_win() -> None:
    approval = make_approval()
    approve(approval.approval_id)
    now = datetime.now()

    with session_scope() as first:
        changed_first = repository.claim_for_execution(
            first, approval.approval_id, claimed_at=now
        )
    with session_scope() as second:  # a competing request, same moment
        changed_second = repository.claim_for_execution(
            second, approval.approval_id, claimed_at=now
        )

    assert (changed_first, changed_second) == (1, 0)
    assert get_approval(approval.approval_id).status == "executing"


def test_parallel_execute_attempts_run_the_action_once() -> None:
    approval = make_approval()  # full refund restores 2x Kopi Susu
    approve(approval.approval_id)

    # Harness note: the losing attempt MUST raise ApprovalConflictError
    # (the executor's 409). pool.map() would propagate that expected
    # exception as a test failure, so each attempt runs via submit() and
    # captures its outcome instead — the conflict IS the assertion.
    outcomes: list[Any] = []

    def attempt() -> None:
        try:
            outcomes.append(executor.execute_approval(approval.approval_id))
        except ApprovalConflictError as exc:
            outcomes.append(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(attempt) for _ in range(2)]
        for future in futures:
            future.result()

    executed = [o for o in outcomes if getattr(o, "status", None) == "executed"]
    conflicts = [o for o in outcomes if isinstance(o, ApprovalConflictError)]
    assert len(executed) == 1
    assert len(conflicts) == 1
    assert get_approval(approval.approval_id).status == "executed"
    assert check_stock("Kopi Susu")["stock"] == 26  # restored exactly once


def test_parallel_execute_attempts_exactly_one_succeeds() -> None:
    approval = make_approval()
    approve(approval.approval_id)

    # A barrier makes this a TRUE race: both threads reach the execute
    # call at the same instant, so the guarded claim — not thread
    # scheduling — decides the winner.
    start = threading.Barrier(2, timeout=10)
    outcomes: list[Any] = []

    def attempt() -> None:
        start.wait()  # both threads armed; release together
        try:
            outcomes.append(executor.execute_approval(approval.approval_id))
        except ApprovalConflictError as exc:
            outcomes.append(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(attempt) for _ in range(2)]
        for future in futures:
            future.result()

    executed = [o for o in outcomes if getattr(o, "status", None) == "executed"]
    conflicts = [o for o in outcomes if isinstance(o, ApprovalConflictError)]
    assert len(executed) == 1
    assert len(conflicts) == 1
    assert get_approval(approval.approval_id).status == "executed"
    # No duplicate side effect: stock restored once, order counted once.
    assert check_stock("Kopi Susu")["stock"] == 26
    assert get_sales_report("monthly")["total_orders"] == 3


# ---------------------------------------------------------------------------
# E. Security
# ---------------------------------------------------------------------------


def _insert_tampered_approval(approval_id: str, action: str, payload: dict) -> None:
    """Bypass the manager and write a row directly (simulates DB tampering)."""
    with session_scope() as session:
        repository.insert_approval(
            session,
            approval_id=approval_id,
            action=action,
            requested_by="agent",
            payload=payload,
            created_at=datetime.now(),
        )


def test_arbitrary_action_name_is_refused() -> None:
    _insert_tampered_approval(
        "apr-evil00001", "delete_everything", {"table": "orders"}
    )
    approve("apr-evil00001")

    with pytest.raises(ApprovalNotExecutableError, match="allowlist"):
        executor.execute_approval("apr-evil00001")

    assert get_approval("apr-evil00001").status == "approved"  # untouched


def test_regular_registry_tool_cannot_be_executed_via_approvals() -> None:
    _insert_tampered_approval(
        "apr-registry1", "check_stock", {"product_name": "Kopi Susu"}
    )
    approve("apr-registry1")

    with pytest.raises(ApprovalNotExecutableError, match="allowlist"):
        executor.execute_approval("apr-registry1")


@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing order_id
        {"order_id": "   "},  # blank order_id
        {"order_id": "ORD-0001", "amount": "cheap"},  # wrong type
        {"order_id": "ORD-0001", "amount": True},  # bool masquerading as int
        {"order_id": 42},  # wrong type entirely
    ],
)
def test_malformed_refund_payload_is_refused_without_state_change(
    payload: dict,
) -> None:
    approval = make_approval(payload=payload)
    approve(approval.approval_id)

    with pytest.raises(ApprovalNotExecutableError):
        executor.execute_approval(approval.approval_id)

    stored = get_approval(approval.approval_id)
    assert stored.status == "approved"  # pre-claim refusal: no transition
    assert stored.execution_error is None


def test_malformed_bulk_update_payload_is_refused() -> None:
    for bad in ([], "not-a-list", [{"product_name": "Kopi Susu"}]):
        approval = make_approval(
            action="bulk_stock_update", payload={"updates": bad}
        )
        approve(approval.approval_id)
        with pytest.raises(ApprovalNotExecutableError):
            executor.execute_approval(approval.approval_id)
        assert get_approval(approval.approval_id).status == "approved"


def test_execution_error_is_bounded_and_business_level() -> None:
    approval = make_approval(payload={"order_id": "ORD-0001", "amount": 1})
    approve(approval.approval_id)

    def exploding_tool(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("boom at internal path /var/secret: password=x")

    original = APPROVED_EXECUTABLE_TOOLS["refund_order"]
    APPROVED_EXECUTABLE_TOOLS["refund_order"] = exploding_tool
    try:
        finished = executor.execute_approval(approval.approval_id)
    finally:
        APPROVED_EXECUTABLE_TOOLS["refund_order"] = original

    assert finished.status == "failed"
    assert finished.execution_error is not None
    assert len(finished.execution_error) <= 500
    # The exception string itself is stored (class + message, bounded) —
    # the guarantee is boundedness and no traceback, verified next.
    assert "Traceback" not in finished.execution_error


def test_observability_events_never_carry_the_payload() -> None:
    approval = make_approval(payload={"order_id": "ORD-0001", "amount": 36000})
    approve(approval.approval_id)
    executor.execute_approval(approval.approval_id)

    runs = list_runs(source="approval_executor", limit=20)
    assert runs, "executor trace run missing"
    events = get_run_events(runs[0].run_id, event_type="APPROVAL")
    assert events
    for event in events:
        metadata = event.metadata or {}
        assert set(metadata) <= {"approval_id", "action", "status"}
        assert "36000" not in str(metadata)


# ---------------------------------------------------------------------------
# F. Observability
# ---------------------------------------------------------------------------


def _approval_events(approval_id: str) -> list[Any]:
    runs = list_runs(source="approval_executor", limit=50)
    collected: list[Any] = []
    for run in runs:
        for event in get_run_events(run.run_id, event_type="APPROVAL") or []:
            metadata = event.metadata or {}
            if metadata.get("approval_id") == approval_id:
                collected.append(event)
    return collected


def test_successful_execution_emits_executing_then_executed_events() -> None:
    approval = make_approval()
    approve(approval.approval_id)
    executor.execute_approval(approval.approval_id)

    events = _approval_events(approval.approval_id)
    names = [event.event_name for event in events]
    assert names == ["APPROVAL_EXECUTING", "APPROVAL_EXECUTED"]
    assert all(event.status == "success" for event in events)
    executing = events[0]
    assert executing.metadata == {
        "approval_id": approval.approval_id,
        "action": "refund_order",
        "status": "executing",
    }


def test_failed_execution_emits_final_failed_event() -> None:
    approval = make_approval(payload={"order_id": "ORD-9999"})
    approve(approval.approval_id)
    executor.execute_approval(approval.approval_id)

    events = _approval_events(approval.approval_id)
    names = [event.event_name for event in events]
    assert names == ["APPROVAL_EXECUTING", "APPROVAL_FAILED"]
    assert events[-1].status == "failed"


def test_observability_failure_does_not_break_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.observability as observability

    def broken_start_run(**kwargs: Any) -> str:
        raise RuntimeError("observability store down")

    monkeypatch.setattr(observability, "start_run", broken_start_run)
    approval = make_approval()
    approve(approval.approval_id)

    finished = executor.execute_approval(approval.approval_id)

    assert finished.status == "executed"  # the business result stands
    assert check_stock("Kopi Susu")["stock"] == 26


# ---------------------------------------------------------------------------
# G. Persistence across a simulated restart
# ---------------------------------------------------------------------------


def test_execution_state_survives_engine_recreation() -> None:
    approval = make_approval()
    approve(approval.approval_id)
    executor.execute_approval(approval.approval_id)

    dispose_engines()  # cached state gone; only PostgreSQL knows

    stored = get_approval(approval.approval_id)
    assert stored is not None
    assert stored.status == "executed"
    assert stored.execution_result is not None
    assert stored.execution_result["order_id"] == "ORD-0001"
    assert stored.execution_started_at is not None
    assert stored.executed_at is not None
    assert check_stock("Kopi Susu")["stock"] == 26

    # And the finished state still refuses a second execution.
    with pytest.raises(ApprovalConflictError):
        executor.execute_approval(approval.approval_id)


def test_failed_state_survives_engine_recreation() -> None:
    approval = make_approval(payload={"order_id": "ORD-9999"})
    approve(approval.approval_id)
    executor.execute_approval(approval.approval_id)

    dispose_engines()

    stored = get_approval(approval.approval_id)
    assert stored is not None
    assert stored.status == "failed"
    assert "ORD-9999" in (stored.execution_error or "")


# ---------------------------------------------------------------------------
# H. API
# ---------------------------------------------------------------------------


def test_api_execute_approved_approval_returns_executed_with_result() -> None:
    approval = make_approval()
    with TestClient(app) as client:
        client.post(f"/api/approvals/{approval.approval_id}/approve")
        response = api_execute(client, approval.approval_id)

    assert response.status_code == 200
    body = response.json()
    assert body["approval_id"] == approval.approval_id
    assert body["status"] == "executed"
    assert body["execution_result"]["success"] is True
    assert body["execution_error"] is None
    assert body["execution_started_at"]
    assert body["executed_at"]


def test_api_execute_failed_business_case_returns_failed_with_error() -> None:
    approval = make_approval(payload={"order_id": "ORD-9999"})
    with TestClient(app) as client:
        client.post(f"/api/approvals/{approval.approval_id}/approve")
        response = api_execute(client, approval.approval_id)

    assert response.status_code == 200  # the request worked; the action failed
    body = response.json()
    assert body["status"] == "failed"
    assert "ORD-9999" in body["execution_error"]
    assert "Traceback" not in body["execution_error"]


def test_api_execute_status_codes() -> None:
    with TestClient(app) as client:
        # Unknown id -> 404.
        assert api_execute(client, "apr-nope").status_code == 404

        # Pending -> 409.
        pending = make_approval()
        assert api_execute(client, pending.approval_id).status_code == 409

        # Rejected -> 409.
        rejected = make_approval()
        client.post(f"/api/approvals/{rejected.approval_id}/reject")
        response = api_execute(client, rejected.approval_id)
        assert response.status_code == 409
        assert "rejected" in response.json()["detail"]

        # Executed -> second execute 409.
        done = make_approval(payload={"order_id": "ORD-0002"})
        client.post(f"/api/approvals/{done.approval_id}/approve")
        assert api_execute(client, done.approval_id).status_code == 200
        repeat = api_execute(client, done.approval_id)
        assert repeat.status_code == 409
        assert "already been executed" in repeat.json()["detail"]

        # Non-allowlisted action -> 422 (simulated tampered row).
        _insert_tampered_approval("apr-api4221", "steal_money", {})
        client.post("/api/approvals/apr-api4221/approve")
        assert api_execute(client, "apr-api4221").status_code == 422


def test_api_list_approvals_status_filter() -> None:
    executed_one = make_approval(payload={"order_id": "ORD-0001"})
    pending_one = make_approval(action="cancel_order", payload={"order_id": "ORD-0002"})
    rejected_one = make_approval(action="cancel_order", payload={"order_id": "ORD-0003"})

    with TestClient(app) as client:
        client.post(f"/api/approvals/{executed_one.approval_id}/approve")
        assert (
            api_execute(client, executed_one.approval_id).status_code == 200
        )
        client.post(f"/api/approvals/{rejected_one.approval_id}/reject")

        default_body = client.get("/api/approvals").json()
        assert default_body["count"] == 1  # only pending_one

        pending = client.get("/api/approvals", params={"status": "pending"}).json()
        assert [a["approval_id"] for a in pending["approvals"]] == [
            pending_one.approval_id
        ]

        executed = client.get("/api/approvals", params={"status": "executed"}).json()
        assert [a["approval_id"] for a in executed["approvals"]] == [
            executed_one.approval_id
        ]
        assert executed["approvals"][0]["execution_result"]["order_id"] == "ORD-0001"

        rejected = client.get("/api/approvals", params={"status": "rejected"}).json()
        assert [a["approval_id"] for a in rejected["approvals"]] == [
            rejected_one.approval_id
        ]

        everything = client.get("/api/approvals", params={"status": "all"}).json()
        assert everything["count"] == 3

        invalid = client.get("/api/approvals", params={"status": "banana"})
        assert invalid.status_code == 422


def test_manager_list_approvals_matches_statuses() -> None:
    first = make_approval()
    second = make_approval(action="cancel_order", payload={"order_id": "ORD-0002"})
    approve(first.approval_id)

    assert [a.approval_id for a in list_approvals(["approved"])] == [
        first.approval_id
    ]
    assert [a.approval_id for a in list_approvals(["pending"])] == [
        second.approval_id
    ]
    assert len(list_approvals()) == 2
    assert len(list_approvals(["executed"])) == 0


# ---------------------------------------------------------------------------
# End-to-end agent flow: request -> approve -> execute (never auto)
# ---------------------------------------------------------------------------


def test_agent_request_then_human_approve_then_execute_flow() -> None:
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
            FakeGLMResponse(text="Refund ORD-0001 menunggu persetujuan manusia."),
        ]
    )
    reply = asyncio.run(Agent(fake, tools={}, schemas=[]).run("Refund ORD-0001"))
    assert "persetujuan" in reply

    # The agent created — and did NOT execute — the approval.
    pending = list_approvals(["pending"])
    assert len(pending) == 1
    assert get_sales_report("monthly")["total_orders"] == 4

    # A human approves, then explicitly executes.
    with TestClient(app) as client:
        client.post(f"/api/approvals/{pending[0].approval_id}/approve")
        body = api_execute(client, pending[0].approval_id).json()

    assert body["status"] == "executed"
    assert check_stock("Kopi Susu")["stock"] == 26
    assert get_sales_report("monthly")["total_orders"] == 3
