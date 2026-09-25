"""Data access layer for agent observability (Step 15).

Same conventions as the other repositories: plain functions taking an
open :class:`~sqlalchemy.orm.Session`, transaction boundaries owned by
the caller (the manager uses ``session_scope``). Writes are append-
oriented single inserts/updates so each tracing write stays cheap and
independent; run completion/failure is a guarded UPDATE on ``run_id``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.db.models import AgentEventRecord, AgentRunRecord

__all__ = [
    "complete_run",
    "fail_run",
    "find_event",
    "find_run",
    "insert_event",
    "insert_run",
    "list_runs",
    "run_events",
    "update_event",
]


def insert_run(
    session: Session,
    *,
    run_id: str,
    source: str,
    owner_key: str | None,
    request_preview: str | None,
    started_at: datetime,
) -> AgentRunRecord:
    """Insert one running agent run row and return it."""
    record = AgentRunRecord(
        run_id=run_id,
        source=source,
        owner_key=owner_key,
        status="running",
        request_preview=request_preview,
        started_at=started_at,
        completed_at=None,
        duration_ms=None,
        error_type=None,
    )
    session.add(record)
    session.flush()
    return record


def complete_run(
    session: Session,
    run_id: str,
    *,
    completed_at: datetime,
    duration_ms: int,
) -> int:
    """Guarded UPDATE running -> completed; returns the rowcount."""
    result = session.execute(
        update(AgentRunRecord)
        .where(AgentRunRecord.run_id == run_id, AgentRunRecord.status == "running")
        .values(status="completed", completed_at=completed_at, duration_ms=duration_ms)
        .execution_options(synchronize_session=False)
    )
    return result.rowcount


def fail_run(
    session: Session,
    run_id: str,
    *,
    completed_at: datetime,
    duration_ms: int,
    error_type: str | None,
) -> int:
    """Guarded UPDATE running -> failed; returns the rowcount."""
    result = session.execute(
        update(AgentRunRecord)
        .where(AgentRunRecord.run_id == run_id, AgentRunRecord.status == "running")
        .values(
            status="failed",
            completed_at=completed_at,
            duration_ms=duration_ms,
            error_type=error_type,
        )
        .execution_options(synchronize_session=False)
    )
    return result.rowcount


def find_run(session: Session, run_id: str) -> AgentRunRecord | None:
    """Return the run row with this public run_id, or None."""
    return session.scalar(
        select(AgentRunRecord).where(AgentRunRecord.run_id == run_id)
    )


def list_runs(
    session: Session,
    *,
    status: str | None = None,
    source: str | None = None,
    owner_key: str | None = None,
    limit: int = 50,
) -> list[tuple[AgentRunRecord, int]]:
    """Recent runs (newest first) with their event counts."""
    event_count = (
        select(
            AgentEventRecord.run_id,
            func.count(AgentEventRecord.id).label("count"),
        )
        .group_by(AgentEventRecord.run_id)
        .subquery()
    )
    statement = select(AgentRunRecord, func.coalesce(event_count.c.count, 0)).outerjoin(
        event_count, event_count.c.run_id == AgentRunRecord.run_id
    )
    if status is not None:
        statement = statement.where(AgentRunRecord.status == status)
    if source is not None:
        statement = statement.where(AgentRunRecord.source == source)
    if owner_key is not None:
        statement = statement.where(AgentRunRecord.owner_key == owner_key)
    statement = statement.order_by(AgentRunRecord.id.desc()).limit(limit)
    return [
        (record, int(count))
        for record, count in session.execute(statement).all()
    ]


def insert_event(
    session: Session,
    *,
    run_id: str,
    event_type: str,
    event_name: str,
    status: str,
    iteration: int | None,
    started_at: datetime,
    completed_at: datetime | None,
    duration_ms: int | None,
    metadata_json: dict[str, Any] | None,
    error_type: str | None,
) -> AgentEventRecord:
    """Insert one event row (usually already completed) and return it."""
    record = AgentEventRecord(
        run_id=run_id,
        event_type=event_type,
        event_name=event_name,
        status=status,
        iteration=iteration,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=duration_ms,
        event_metadata=metadata_json,
        error_type=error_type,
    )
    session.add(record)
    session.flush()
    return record


def find_event(session: Session, event_id: int) -> AgentEventRecord | None:
    """Return one event row by id, or None."""
    return session.get(AgentEventRecord, event_id)


def update_event(
    session: Session,
    event_id: int,
    *,
    status: str,
    completed_at: datetime,
    duration_ms: int,
    metadata_json: dict[str, Any] | None = None,
    error_type: str | None = None,
) -> int:
    """Close an event (success/failed); returns the rowcount."""
    values: dict[str, Any] = {
        "status": status,
        "completed_at": completed_at,
        "duration_ms": duration_ms,
        "error_type": error_type,
    }
    if metadata_json is not None:
        values["event_metadata"] = metadata_json
    result = session.execute(
        update(AgentEventRecord)
        .where(AgentEventRecord.id == event_id)
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    return result.rowcount


def run_events(
    session: Session,
    run_id: str,
    *,
    event_type: str | None = None,
    limit: int = 500,
) -> list[AgentEventRecord]:
    """One run's events in occurrence order (insertion order)."""
    statement = select(AgentEventRecord).where(AgentEventRecord.run_id == run_id)
    if event_type is not None:
        statement = statement.where(AgentEventRecord.event_type == event_type)
    return list(
        session.scalars(statement.order_by(AgentEventRecord.id).limit(limit))
    )
