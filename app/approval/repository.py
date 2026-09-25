"""Data access layer for the persistent approval store (Step 13/16).

Every query the approval manager and executor need lives here,
mirroring the conventions of ``app/db/repository.py``: plain functions
that take an open :class:`~sqlalchemy.orm.Session`, no raw SQL in
routes or the manager, and transaction boundaries owned by the caller
(the manager uses ``session_scope``).

Both state-changing paths are guarded ``UPDATE ... WHERE status = ...``
statements so two concurrent writers can never both succeed: the
second statement simply matches zero rows (see ``decide_approval`` and
``claim_for_execution`` — the latter is the atomic execution claim that
makes the database the concurrency authority for Step 16).
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.db.models import ApprovalRecord

__all__ = [
    "approvals_by_status",
    "claim_for_execution",
    "decide_approval",
    "delete_all_approvals",
    "find_approval",
    "insert_approval",
    "pending_approvals",
    "record_execution_outcome",
]


def insert_approval(
    session: Session,
    *,
    approval_id: str,
    action: str,
    requested_by: str,
    payload: dict,
    created_at: datetime,
) -> ApprovalRecord:
    """Insert one pending approval row and return it (flushed)."""
    record = ApprovalRecord(
        approval_id=approval_id,
        action=action,
        requested_by=requested_by,
        payload=payload,
        status="pending",
        created_at=created_at,
        decided_at=None,
        decision_reason=None,
    )
    session.add(record)
    session.flush()  # assign the surrogate id before the caller reads it
    session.refresh(record)  # load the updated_at server default
    return record


def find_approval(session: Session, approval_id: str) -> ApprovalRecord | None:
    """Return the approval row with this public id (any status), or None."""
    return session.scalar(
        select(ApprovalRecord).where(ApprovalRecord.approval_id == approval_id)
    )


def pending_approvals(session: Session) -> list[ApprovalRecord]:
    """Every still-pending approval, oldest (insertion) first."""
    return list(
        session.scalars(
            select(ApprovalRecord)
            .where(ApprovalRecord.status == "pending")
            .order_by(ApprovalRecord.id)
        )
    )


def decide_approval(
    session: Session,
    approval_id: str,
    *,
    expected_status: str,
    new_status: str,
    decided_at: datetime,
    decision_reason: str | None = None,
) -> int:
    """Apply a decision via a guarded UPDATE; return the rowcount.

    The ``expected_status`` predicate lives inside the WHERE clause, so
    the check and the write are one atomic statement: under READ
    COMMITTED a competing transaction that commits first makes this
    UPDATE match (and change) exactly zero rows instead of deciding an
    already-decided approval. Callers must treat ``0`` as "someone else
    decided it (or it never existed)" and re-read the row to tell the
    two cases apart.
    """
    result = session.execute(
        update(ApprovalRecord)
        .where(
            ApprovalRecord.approval_id == approval_id,
            ApprovalRecord.status == expected_status,
        )
        .values(
            status=new_status,
            decided_at=decided_at,
            decision_reason=decision_reason,
        )
        .execution_options(synchronize_session=False)
    )
    return result.rowcount


def delete_all_approvals(session: Session) -> int:
    """Remove every approval row (test helper); returns rows deleted."""
    result = session.execute(delete(ApprovalRecord))
    return result.rowcount


def approvals_by_status(
    session: Session, statuses: Iterable[str] | None = None
) -> list[ApprovalRecord]:
    """Approvals filtered by status (oldest first); all when ``None``."""
    query = select(ApprovalRecord).order_by(ApprovalRecord.id)
    statuses = list(statuses) if statuses is not None else None
    if statuses:
        query = query.where(ApprovalRecord.status.in_(statuses))
    return list(session.scalars(query))


def claim_for_execution(
    session: Session,
    approval_id: str,
    *,
    claimed_at: datetime,
) -> int:
    """Atomically transition approved -> executing; return the rowcount.

    The guard lives inside the WHERE clause (``status = 'approved'``),
    so the check and the write are one statement: under READ COMMITTED,
    of two concurrent execute requests exactly one matches and changes
    the row — the other gets rowcount 0 and must treat the approval as
    claimed by someone else (or no longer approved). This makes the
    database, not any in-process lock, the concurrency authority.
    """
    result = session.execute(
        update(ApprovalRecord)
        .where(
            ApprovalRecord.approval_id == approval_id,
            ApprovalRecord.status == "approved",
        )
        .values(status="executing", execution_started_at=claimed_at)
        .execution_options(synchronize_session=False)
    )
    return result.rowcount


def record_execution_outcome(
    session: Session,
    approval_id: str,
    *,
    status: str,
    finished_at: datetime,
    result: dict | None = None,
    error: str | None = None,
) -> int:
    """Guarded UPDATE executing -> executed/failed; return the rowcount.

    Only a row still in ``executing`` can be finished, so a stale or
    concurrent writer cannot overwrite a finished outcome (rowcount 0
    is "someone else finished it first" — the caller re-reads the row).
    """
    values: dict = {"status": status, "executed_at": finished_at}
    if status == "executed":
        values["execution_result"] = result
    else:
        values["execution_error"] = error
    result_stmt = (
        update(ApprovalRecord)
        .where(
            ApprovalRecord.approval_id == approval_id,
            ApprovalRecord.status == "executing",
        )
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    return session.execute(result_stmt).rowcount
