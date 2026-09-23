"""Human-in-the-loop approval package for DibantuAI (Step 8).

Sensitive business actions (refund_order, cancel_order,
bulk_stock_update) are intercepted centrally in the agent's tool
dispatch: instead of executing, a pending approval is created here and
the agent gets a structured approval-required result. Humans decide
through the approval API endpoints.
"""

from .manager import (
    Approval,
    ApprovalError,
    ApprovalNotFoundError,
    ApprovalNotPendingError,
    SENSITIVE_ACTIONS,
    approval_required_result,
    approve,
    create_pending_approval,
    get_approval,
    get_pending_approval,
    is_sensitive_action,
    list_pending,
    reject,
    reset_approvals,
)

__all__ = [
    "Approval",
    "ApprovalError",
    "ApprovalNotFoundError",
    "ApprovalNotPendingError",
    "SENSITIVE_ACTIONS",
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
