"""Human-in-the-loop approval package for DibantuAI (Step 8 → Step 16).

Sensitive business actions (refund_order, cancel_order,
bulk_stock_update) are intercepted centrally in the agent's tool
dispatch: instead of executing, a pending approval is created here and
the agent gets a structured approval-required result. Humans decide
through the approval API endpoints; since Step 16, an APPROVED action
can additionally be executed — exactly once, from the immutable stored
payload — through the explicit executor and its API endpoint.
"""

from .executor import (
    APPROVED_EXECUTABLE_TOOLS,
    ApprovalConflictError,
    ApprovalNotExecutableError,
    execute_approval,
)
from .manager import (
    ALL_STATUSES,
    ALLOWED_TRANSITIONS,
    Approval,
    ApprovalError,
    ApprovalNotFoundError,
    ApprovalNotPendingError,
    SENSITIVE_ACTIONS,
    STATUS_APPROVED,
    STATUS_EXECUTED,
    STATUS_EXECUTING,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_REJECTED,
    approval_required_result,
    approve,
    can_transition,
    create_pending_approval,
    get_approval,
    get_pending_approval,
    is_sensitive_action,
    list_approvals,
    list_pending,
    reject,
    reset_approvals,
    validate_transition,
)

__all__ = [
    "ALL_STATUSES",
    "ALLOWED_TRANSITIONS",
    "APPROVED_EXECUTABLE_TOOLS",
    "Approval",
    "ApprovalConflictError",
    "ApprovalError",
    "ApprovalNotFoundError",
    "ApprovalNotExecutableError",
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
    "execute_approval",
    "get_approval",
    "get_pending_approval",
    "is_sensitive_action",
    "list_approvals",
    "list_pending",
    "reject",
    "reset_approvals",
    "validate_transition",
]
