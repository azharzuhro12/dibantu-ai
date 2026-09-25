"""Deferred sensitive-action executor (Step 16).

Executes an APPROVED sensitive action using — exclusively — the
immutable payload snapshot stored when the approval was created. The
client that triggers execution supplies nothing but the approval id:
no arguments, no tool name, so "approve request A but execute request
B" is impossible by construction.

Safety properties, in the order they are enforced:

1. **Allowlist.** Only the three sensitive actions with real, safe
   business implementations can ever execute
   (``APPROVED_EXECUTABLE_TOOLS`` maps name -> implementation). The
   executor is not a generic function runner: anything not in the map
   is refused before any state changes.
2. **Structural payload validation** (pre-claim). A payload missing
   required keys or holding values of the wrong shape is refused with
   422 and NO state change — the approval stays approved and the
   broken snapshot remains inspectable.
3. **Atomic claim.** ``approved -> executing`` is one guarded
   ``UPDATE ... WHERE status = 'approved'``: of two concurrent execute
   requests exactly one claims the row (rowcount 1); the loser gets a
   conflict. The database is the concurrency authority — there is no
   in-process lock.
4. **Terminal record.** The claimed row ends ``executed`` (with the
   tool's result dict) or ``failed`` (with a bounded, business-level
   error message — never a traceback, never credentials). Both states
   are terminal: re-executing, or executing a rejected/pending
   approval, returns a conflict without touching anything.
5. **Observability.** Each execution emits APPROVAL events (executing,
   then executed/failed) with safe metadata only; tracing failures are
   swallowed and can never fail the execution or the API.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable

from app.approval import repository
from app.approval.manager import (
    Approval,
    ApprovalError,
    ApprovalNotFoundError,
    STATUS_APPROVED,
    STATUS_EXECUTED,
    STATUS_EXECUTING,
    STATUS_FAILED,
    get_approval,
)
from app.db.database import session_scope
from app.tools.sensitive_actions import (
    MAX_BULK_UPDATES,
    bulk_stock_update,
    cancel_order,
    refund_order,
)

__all__ = [
    "APPROVED_EXECUTABLE_TOOLS",
    "ApprovalConflictError",
    "ApprovalNotExecutableError",
    "execute_approval",
]

logger = logging.getLogger(__name__)

#: The explicit execution allowlist: sensitive action -> real business
#: implementation. Anything absent from this mapping is never executed,
#: no matter what a database row claims. This is deliberately NOT the
#: agent tool registry — the agent must not be able to reach these
#: functions directly.
APPROVED_EXECUTABLE_TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "refund_order": refund_order,
    "cancel_order": cancel_order,
    "bulk_stock_update": bulk_stock_update,
}

#: Longest execution error stored (matches the DB column width).
_MAX_EXECUTION_ERROR_LENGTH = 500


class ApprovalNotExecutableError(ApprovalError):
    """Raised when an approval can never be executed as stored.

    Covers a non-allowlisted action and structurally malformed
    payloads — problems a retry cannot fix. No state is changed.
    """


class ApprovalConflictError(ApprovalError):
    """Raised when the approval's status forbids execution now.

    Carries the current approval so the API can answer with the exact
    situation (pending decision, already executing, already executed,
    failed earlier, or rejected).
    """

    def __init__(self, approval: Approval) -> None:
        self.approval = approval
        if approval.status == "executing":
            detail = "is already executing"
        elif approval.status == STATUS_EXECUTED:
            detail = "has already been executed"
        elif approval.status == STATUS_FAILED:
            detail = "failed earlier and is not retried automatically"
        elif approval.status == STATUS_APPROVED:
            detail = "was claimed by another execution just now"
        else:
            detail = f"is {approval.status}, not approved"
        super().__init__(
            f"Approval '{approval.approval_id}' {detail}; "
            "no execution was performed."
        )


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ApprovalNotExecutableError(
            f"Stored payload field '{key}' must be a non-empty string."
        )
    return value


def _validate_payload_structure(
    action: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """Return the exact kwargs for the action's implementation.

    Raises ``ApprovalNotExecutableError`` when the stored snapshot is
    structurally unusable (a retry can never fix that). Only known
    fields pass through — unknown extra keys are dropped, never
    forwarded, so the snapshot cannot smuggle arbitrary parameters.
    Item-level bulk validation runs here too (pre-claim), so a broken
    snapshot never claims the row; the business layer re-validates
    everything as defense in depth.
    """
    if action == "refund_order":
        kwargs: dict[str, Any] = {"order_id": _require_str(payload, "order_id")}
        amount = payload.get("amount")
        if amount is not None:
            if isinstance(amount, bool) or not isinstance(amount, (int, float)):
                raise ApprovalNotExecutableError(
                    "Stored payload field 'amount' must be a number."
                )
            kwargs["amount"] = amount
        return kwargs
    if action == "cancel_order":
        return {"order_id": _require_str(payload, "order_id")}
    if action == "bulk_stock_update":
        updates = payload.get("updates")
        if not isinstance(updates, list) or not updates:
            raise ApprovalNotExecutableError(
                "Stored payload field 'updates' must be a non-empty list."
            )
        if len(updates) > MAX_BULK_UPDATES:
            raise ApprovalNotExecutableError(
                f"Stored payload field 'updates' exceeds {MAX_BULK_UPDATES} items."
            )
        for update in updates:
            if not isinstance(update, dict):
                raise ApprovalNotExecutableError(
                    "Every stored 'updates' item must be an object."
                )
            if not isinstance(update.get("product_name"), str) or not (
                update.get("product_name") or ""
            ).strip():
                raise ApprovalNotExecutableError(
                    "Every stored 'updates' item needs a non-empty 'product_name'."
                )
            change = update.get("quantity_change")
            if isinstance(change, bool) or not isinstance(change, int) or change == 0:
                raise ApprovalNotExecutableError(
                    "Every stored 'updates' item needs a non-zero integer "
                    "'quantity_change'."
                )
        return {"updates": updates}
    raise ApprovalNotExecutableError(  # pragma: no cover - guarded by allowlist
        f"Action '{action}' is not executable."
    )


def _trace_start(approval_id: str, action: str) -> str | None:
    """Open one executor trace run and emit APPROVAL_EXECUTING.

    Returns the run id, or ``None`` when observability is unavailable
    (the failure is swallowed — tracing must never break an execution).
    The same run later carries the terminal event, so one execution
    reads as one timeline: APPROVAL_EXECUTING then APPROVAL_EXECUTED
    or APPROVAL_FAILED.
    """
    try:
        from app.observability import record_event, start_run

        run_id = start_run(source="approval_executor")
        record_event(
            run_id,
            "APPROVAL",
            "APPROVAL_EXECUTING",
            status="success",
            duration_ms=0,
            metadata={
                "approval_id": approval_id,
                "action": action,
                "status": "executing",
            },
        )
        return run_id
    except Exception as exc:  # noqa: BLE001 - tracing must never break execution
        logger.warning(
            "observability: failed to trace approval execution (%s)",
            type(exc).__name__,
        )
        return None


def _trace_finish(
    trace_run_id: str | None,
    approval_id: str,
    action: str,
    phase: str,
    *,
    failed: bool = False,
) -> None:
    """Emit the terminal APPROVAL event and close the trace run.

    ``failed`` executions mark the trace run itself failed (RUN_FAILED)
    so both readings of the timeline agree. Best-effort, swallowed.
    """
    if trace_run_id is None:
        return
    try:
        from app.observability import complete_run, fail_run, record_event

        record_event(
            trace_run_id,
            "APPROVAL",
            f"APPROVAL_{phase.upper()}",
            status="failed" if failed else "success",
            duration_ms=0,
            metadata={
                "approval_id": approval_id,
                "action": action,
                "status": phase,
            },
        )
        if failed:
            fail_run(trace_run_id, error_type=None, duration_ms=0)
        else:
            complete_run(trace_run_id, duration_ms=0)
    except Exception as exc:  # noqa: BLE001 - tracing must never break execution
        logger.warning(
            "observability: failed to trace approval execution (%s)",
            type(exc).__name__,
        )


def execute_approval(approval_id: str) -> Approval:
    """Execute one approved approval; returns the finished approval.

    See the module docstring for the enforced safety properties. The
    returned ``Approval`` is the terminal snapshot: ``executed`` with
    ``execution_result`` set, or ``failed`` with ``execution_error``.
    """
    approval = get_approval(approval_id)
    if approval is None:
        raise ApprovalNotFoundError(f"Approval '{approval_id}' not found.")

    tool = APPROVED_EXECUTABLE_TOOLS.get(approval.action)
    if tool is None:
        raise ApprovalNotExecutableError(
            f"Action '{approval.action}' is not in the executable "
            "allowlist; refusing to execute."
        )
    if not isinstance(approval.payload, dict):
        raise ApprovalNotExecutableError(
            "Stored payload is not a JSON object; refusing to execute."
        )
    kwargs = _validate_payload_structure(approval.action, approval.payload)

    # Atomic claim: exactly one concurrent caller gets past this UPDATE.
    started_at = datetime.now()
    with session_scope() as session:
        changed = repository.claim_for_execution(
            session, approval_id, claimed_at=started_at
        )
    if changed == 0:
        current = get_approval(approval_id)
        raise ApprovalConflictError(
            current if current is not None else approval
        )

    trace_run_id = _trace_start(approval_id, approval.action)

    try:
        result = tool(**kwargs)
    except Exception as exc:  # noqa: BLE001 - any crash fails the execution
        error_message = f"{type(exc).__name__}: {exc}"[:_MAX_EXECUTION_ERROR_LENGTH]
        _record(approval_id, STATUS_FAILED, error=error_message)
        _trace_finish(
            trace_run_id, approval_id, approval.action, "failed", failed=True
        )
        final = get_approval(approval_id)
        if final is not None:
            return final
        raise  # the row vanished mid-execution; surface the store error

    if isinstance(result, dict) and result.get("success") is True:
        _record(approval_id, STATUS_EXECUTED, result=result)
        _trace_finish(trace_run_id, approval_id, approval.action, "executed")
    else:
        # The business implementation refused (validation, unknown
        # order, insufficient stock, database unavailable, ...). Record
        # the refusal as the execution failure — bounded, business-level.
        if isinstance(result, dict):
            error_message = str(
                result.get("message") or result.get("error") or "execution refused"
            )[:_MAX_EXECUTION_ERROR_LENGTH]
        else:  # pragma: no cover - implementations always return dicts
            error_message = "execution returned no result"
        _record(approval_id, STATUS_FAILED, error=error_message)
        _trace_finish(
            trace_run_id, approval_id, approval.action, "failed", failed=True
        )

    final = get_approval(approval_id)
    if final is None:  # pragma: no cover - row existed a moment ago
        raise ApprovalNotFoundError(
            f"Approval '{approval_id}' disappeared during execution."
        )
    return final


def _record(
    approval_id: str,
    status: str,
    *,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """Write the terminal outcome (guarded executing -> finished)."""
    with session_scope() as session:
        repository.record_execution_outcome(
            session,
            approval_id,
            status=status,
            finished_at=datetime.now(),
            result=result,
            error=error,
        )
