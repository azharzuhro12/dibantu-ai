"""In-memory human-in-the-loop approval manager (Step 8).

Sensitive business actions are never executed directly by the agent.
Requesting one creates a pending ``Approval`` here, and the agent
receives a structured ``approval_required`` result instead of an
execution result. A human later decides via the approval API
(approve/reject). This is an MVP: the store is a plain module-level
dict (no database, no persistence, no authentication), fine for a
single-process local service and its tests.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

__all__ = [
    "Approval",
    "ApprovalError",
    "ApprovalNotFoundError",
    "ApprovalNotPendingError",
    "SENSITIVE_ACTIONS",
    "STATUS_APPROVED",
    "STATUS_PENDING",
    "STATUS_REJECTED",
    "approval_required_result",
    "approve",
    "create_pending_approval",
    "get_approval",
    "get_pending_approval",
    "is_sensitive_action",
    "list_pending",
    "reject",
    "reset_approvals",
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


@dataclass
class Approval:
    """One pending-or-decided sensitive action request."""

    approval_id: str
    action: str
    requested_by: str
    payload: dict[str, Any]
    status: str
    created_at: str
    decided_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly copy of the whole record."""
        return asdict(self)


#: approval_id -> Approval (insertion order = creation order).
_STORE: dict[str, Approval] = {}


def reset_approvals() -> None:
    """Clear every approval (test helper)."""
    _STORE.clear()


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
    approval = Approval(
        approval_id=f"apr-{uuid.uuid4().hex[:12]}",
        action=action,
        requested_by=requested_by.strip(),
        payload=dict(payload),
        status=STATUS_PENDING,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    _STORE[approval.approval_id] = approval
    return approval


def get_approval(approval_id: str) -> Approval | None:
    """Return the approval with this id (any status), or None."""
    return _STORE.get(approval_id)


def get_pending_approval(approval_id: str) -> Approval | None:
    """Return the approval only while it is still pending."""
    approval = _STORE.get(approval_id)
    if approval is not None and approval.status == STATUS_PENDING:
        return approval
    return None


def approve(approval_id: str) -> Approval:
    """Mark a pending approval as approved and return it.

    MVP: approving only records the decision and returns the approved
    payload -- no real refund/payment execution happens anywhere.
    """
    return _decide(approval_id, STATUS_APPROVED)


def reject(approval_id: str) -> Approval:
    """Mark a pending approval as rejected and return it."""
    return _decide(approval_id, STATUS_REJECTED)


def list_pending() -> list[Approval]:
    """Return every pending approval, oldest first."""
    return [
        approval
        for approval in _STORE.values()
        if approval.status == STATUS_PENDING
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


def _decide(approval_id: str, new_status: str) -> Approval:
    """Apply a decision to a pending approval, with strict guards."""
    approval = _STORE.get(approval_id)
    if approval is None:
        raise ApprovalNotFoundError(f"Approval '{approval_id}' not found.")
    if approval.status != STATUS_PENDING:
        raise ApprovalNotPendingError(approval)
    approval.status = new_status
    approval.decided_at = datetime.now().isoformat(timespec="seconds")
    return approval
