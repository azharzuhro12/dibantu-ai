"""Human-in-the-loop approval manager (Step 8 → Step 13 → Step 16).

Sensitive business actions are never executed directly by the agent.
Requesting one creates a pending ``Approval`` here, and the agent
receives a structured ``approval_required`` result instead of an
execution result. A human later decides via the approval API
(approve/reject).

Since Step 13 the store is the ``approvals`` PostgreSQL table (via
``app/approval/repository.py``), so pending approvals and decisions
survive backend and container restarts. The module keeps the exact
public surface of the old in-memory implementation; the ``Approval``
dataclass is now an immutable snapshot of one database row rather than
a live reference to shared state. Decisions run through a guarded
``UPDATE ... WHERE status = 'pending'`` so concurrent decide calls can
never both succeed.

Since Step 16 the lifecycle continues after approval — approved ->
executing -> executed/failed — but that machinery lives in
``app/approval/executor.py``; this module owns the store, the status
vocabulary, and the decision transitions only. Approving still records
the human decision and nothing else: execution only ever happens
through the explicit execute path, driven by the immutable payload
snapshot stored at creation time.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Iterable

from app.approval import repository
from app.db.database import session_scope

__all__ = [
    "ALLOWED_TRANSITIONS",
    "Approval",
    "ApprovalError",
    "ApprovalNotFoundError",
    "ApprovalNotPendingError",
    "SENSITIVE_ACTIONS",
    "STATUS_APPROVED",
    "STATUS_EXECUTED",
    "STATUS_EXECUTING",
    "STATUS_FAILED",
    "STATUS_PENDING",
    "STATUS_REJECTED",
    "approval_required_result",
    "approve",
    "can_transition",
    "create_pending_approval",
    "get_approval",
    "get_pending_approval",
    "is_sensitive_action",
    "list_approvals",
    "list_pending",
    "reject",
    "reset_approvals",
    "validate_transition",
]

#: Actions that require human approval before (ever) being executed.
SENSITIVE_ACTIONS: tuple[str, ...] = (
    "refund_order",
    "cancel_order",
    "bulk_stock_update",
)

STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_EXECUTING = "executing"
STATUS_EXECUTED = "executed"
STATUS_FAILED = "failed"

#: Every status an approval row can hold (matches the DB CHECK, 0005).
ALL_STATUSES: tuple[str, ...] = (
    STATUS_PENDING,
    STATUS_APPROVED,
    STATUS_REJECTED,
    STATUS_EXECUTING,
    STATUS_EXECUTED,
    STATUS_FAILED,
)

#: The complete legal transition table (Step 16). Everything not listed
#: is rejected by ``validate_transition`` — most importantly
#: rejected/failed can never re-enter the pipeline, and a decided or
#: finished approval can never be decided again.
ALLOWED_TRANSITIONS: frozenset[tuple[str, str]] = frozenset(
    {
        (STATUS_PENDING, STATUS_APPROVED),
        (STATUS_PENDING, STATUS_REJECTED),
        (STATUS_APPROVED, STATUS_EXECUTING),
        (STATUS_EXECUTING, STATUS_EXECUTED),
        (STATUS_EXECUTING, STATUS_FAILED),
    }
)

#: Longest decision reason stored (matches the DB column width).
MAX_DECISION_REASON_LENGTH = 500


class ApprovalError(Exception):
    """Base class for approval errors."""


class ApprovalNotFoundError(ApprovalError):
    """Raised when an approval id does not exist."""


class ApprovalNotPendingError(ApprovalError):
    """Raised when deciding an approval that was already decided."""

    def __init__(self, approval: "Approval") -> None:
        self.approval = approval
        super().__init__(
            f"Approval '{approval.approval_id}' is already "
            f"{approval.status}, not pending."
        )


@dataclass(frozen=True)
class Approval:
    """One sensitive-action approval and (since Step 16) its execution.

    An immutable snapshot of one ``approvals`` row: timestamps are the
    same ISO strings the API has always returned (local time, second
    precision), and ``payload`` is the exact JSON the agent wanted to
    execute — the immutable snapshot the executor runs after approval;
    clients can never re-supply or mutate it. Execution fields stay
    ``None`` until an execution is attempted.
    """

    approval_id: str
    action: str
    requested_by: str
    payload: dict[str, Any]
    status: str
    created_at: str
    decided_at: str | None = None
    decision_reason: str | None = None
    execution_started_at: str | None = None
    executed_at: str | None = None
    execution_error: str | None = None
    execution_result: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly copy of the whole record."""
        return asdict(self)


def validate_transition(current: str, new: str) -> None:
    """Raise ``ApprovalError`` unless current -> new is a legal step.

    The single source of truth for the state machine; both the decision
    path and the executor's claim/record statements may only perform
    transitions listed in ``ALLOWED_TRANSITIONS``.
    """
    if (current, new) not in ALLOWED_TRANSITIONS:
        raise ApprovalError(
            f"Illegal approval transition: '{current}' -> '{new}'."
        )


def can_transition(current: str, new: str) -> bool:
    """True iff current -> new is one of ``ALLOWED_TRANSITIONS``."""
    return (current, new) in ALLOWED_TRANSITIONS


def reset_approvals() -> None:
    """Delete every approval row (test helper)."""
    with session_scope() as session:
        repository.delete_all_approvals(session)


def is_sensitive_action(name: str) -> bool:
    """Return True if ``name`` is an action that requires approval."""
    return name in SENSITIVE_ACTIONS


def create_pending_approval(
    *,
    action: str,
    requested_by: str,
    payload: dict[str, Any],
) -> Approval:
    """Register a request for a sensitive action as pending.

    Only actions listed in ``SENSITIVE_ACTIONS`` may be registered, so
    the approval store cannot be used to gate arbitrary calls.
    """
    if not is_sensitive_action(action):
        raise ApprovalError(
            f"Action '{action}' is not a sensitive action "
            f"(expected one of: {', '.join(SENSITIVE_ACTIONS)})."
        )
    if not isinstance(requested_by, str) or not requested_by.strip():
        raise ApprovalError("requested_by must be a non-empty string.")
    if not isinstance(payload, dict):
        raise ApprovalError("payload must be a JSON object (dict).")
    with session_scope() as session:
        record = repository.insert_approval(
            session,
            approval_id=f"apr-{uuid.uuid4().hex[:12]}",
            action=action,
            requested_by=requested_by.strip(),
            payload=dict(payload),
            created_at=datetime.now(),
        )
        return _as_approval(record)


def get_approval(approval_id: str) -> Approval | None:
    """Return the approval with this id (any status), or None."""
    with session_scope() as session:
        record = repository.find_approval(session, approval_id)
        return _as_approval(record) if record is not None else None


def get_pending_approval(approval_id: str) -> Approval | None:
    """Return the approval only while it is still pending."""
    approval = get_approval(approval_id)
    if approval is not None and approval.status == STATUS_PENDING:
        return approval
    return None


def approve(approval_id: str, *, reason: str | None = None) -> Approval:
    """Mark a pending approval as approved and return it.

    Approving records the human decision ONLY — the action itself still
    runs nothing. Execution is a separate, explicit step (Step 16:
    ``app/approval/executor.py`` via ``POST /api/approvals/{id}/execute``)
    so a browser refresh or duplicate approve can never trigger it.
    """
    return _decide(approval_id, STATUS_APPROVED, reason)


def reject(approval_id: str, *, reason: str | None = None) -> Approval:
    """Mark a pending approval as rejected and return it."""
    return _decide(approval_id, STATUS_REJECTED, reason)


def list_pending() -> list[Approval]:
    """Return every pending approval, oldest first."""
    with session_scope() as session:
        return [
            _as_approval(record)
            for record in repository.pending_approvals(session)
        ]


def list_approvals(
    statuses: Iterable[str] | None = None,
) -> list[Approval]:
    """Approvals filtered by status (oldest first); all when ``None``.

    Unknown status names simply match nothing (the API layer validates
    its query parameter before calling).
    """
    with session_scope() as session:
        return [
            _as_approval(record)
            for record in repository.approvals_by_status(session, statuses)
        ]


def approval_required_result(approval: Approval) -> dict[str, Any]:
    """Structured result returned to the agent instead of execution."""
    return {
        "approval_required": True,
        "approval_id": approval.approval_id,
        "action": approval.action,
        "status": approval.status,
        "message": (
            f"'{approval.action}' requires human approval and has NOT "
            f"been executed. It is pending as {approval.approval_id}; "
            "tell the user that a human must approve this action."
        ),
    }


def _decide(
    approval_id: str, new_status: str, reason: str | None = None
) -> Approval:
    """Apply a decision to a pending approval, with strict guards.

    The decision is one guarded UPDATE (``... WHERE status = 'pending'``):
    a rowcount of 1 means this call decided it, 0 means another request
    already did (or the id is unknown) — the row is then re-read to
    raise the right error. This is safe under concurrency: two
    simultaneous decisions cannot both change the row. The transition is
    also validated against ``ALLOWED_TRANSITIONS`` first (defense in
    depth: the guard already enforces pending-only decisions).
    """
    validate_transition(STATUS_PENDING, new_status)
    if reason is not None and len(reason) > MAX_DECISION_REASON_LENGTH:
        raise ApprovalError(
            "reason must be at most "
            f"{MAX_DECISION_REASON_LENGTH} characters."
        )
    decided_at = datetime.now()
    with session_scope() as session:
        changed = repository.decide_approval(
            session,
            approval_id,
            expected_status=STATUS_PENDING,
            new_status=new_status,
            decided_at=decided_at,
            decision_reason=reason,
        )
        record = repository.find_approval(session, approval_id)
        if changed == 0:
            if record is None:
                raise ApprovalNotFoundError(
                    f"Approval '{approval_id}' not found."
                )
            raise ApprovalNotPendingError(_as_approval(record))
        return _as_approval(record)


def _as_approval(record: Any) -> Approval:
    """Serialize one ``ApprovalRecord`` row into the public dataclass."""
    return Approval(
        approval_id=record.approval_id,
        action=record.action,
        requested_by=record.requested_by,
        payload=dict(record.payload),
        status=record.status,
        created_at=record.created_at.isoformat(timespec="seconds"),
        decided_at=(
            record.decided_at.isoformat(timespec="seconds")
            if record.decided_at is not None
            else None
        ),
        decision_reason=record.decision_reason,
        execution_started_at=(
            record.execution_started_at.isoformat(timespec="seconds")
            if record.execution_started_at is not None
            else None
        ),
        executed_at=(
            record.executed_at.isoformat(timespec="seconds")
            if record.executed_at is not None
            else None
        ),
        execution_error=record.execution_error,
        execution_result=(
            dict(record.execution_result)
            if record.execution_result is not None
            else None
        ),
    )
