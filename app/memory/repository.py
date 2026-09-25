"""Data access layer for the persistent agent memory store (Step 14).

Mirrors the conventions of ``app/db/repository.py`` and
``app/approval/repository.py``: plain functions that take an open
:class:`~sqlalchemy.orm.Session`, transaction boundaries owned by the
caller (the manager uses ``session_scope``), and no raw SQL in routes
or tools. Update/delete are guarded ``UPDATE/DELETE ... WHERE id = ?
AND owner_key = ?`` so an owner can only ever touch their own rows.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.db.models import AgentMemoryRecord

__all__ = [
    "delete_all_memories",
    "delete_memory",
    "find_memory",
    "insert_memory",
    "list_memories",
    "owner_memories",
    "update_memory",
]


def insert_memory(
    session: Session,
    *,
    owner_key: str,
    memory_type: str,
    content: str,
    created_at: datetime,
) -> AgentMemoryRecord:
    """Insert one memory row and return it (flushed + refreshed)."""
    record = AgentMemoryRecord(
        owner_key=owner_key,
        memory_type=memory_type,
        content=content,
        created_at=created_at,
    )
    session.add(record)
    session.flush()  # assign the id before the caller reads it
    session.refresh(record)  # load the updated_at server default
    return record


def find_memory(session: Session, memory_id: int) -> AgentMemoryRecord | None:
    """Return the memory row with this id (any owner), or None."""
    return session.get(AgentMemoryRecord, memory_id)


def list_memories(
    session: Session,
    *,
    owner_key: str,
    memory_type: str | None = None,
    limit: int | None = None,
) -> list[AgentMemoryRecord]:
    """One owner's memories, newest (highest id) first."""
    statement = select(AgentMemoryRecord).where(
        AgentMemoryRecord.owner_key == owner_key
    )
    if memory_type is not None:
        statement = statement.where(AgentMemoryRecord.memory_type == memory_type)
    statement = statement.order_by(AgentMemoryRecord.id.desc())
    if limit is not None:
        statement = statement.limit(limit)
    return list(session.scalars(statement))


def owner_memories(
    session: Session, owner_key: str, memory_type: str | None = None
) -> list[AgentMemoryRecord]:
    """Every memory of one owner, oldest first — the scoring candidate set.

    Kept unpaginated on purpose: memory is per-owner, small, and the
    deterministic keyword scoring in the manager runs in Python.
    """
    statement = select(AgentMemoryRecord).where(
        AgentMemoryRecord.owner_key == owner_key
    )
    if memory_type is not None:
        statement = statement.where(AgentMemoryRecord.memory_type == memory_type)
    return list(
        session.scalars(statement.order_by(AgentMemoryRecord.id))
    )


def update_memory(
    session: Session,
    memory_id: int,
    *,
    owner_key: str,
    values: dict[str, Any],
) -> int:
    """Owner-guarded UPDATE; returns the rowcount (0 = not found/not yours)."""
    result = session.execute(
        update(AgentMemoryRecord)
        .where(
            AgentMemoryRecord.id == memory_id,
            AgentMemoryRecord.owner_key == owner_key,
        )
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    return result.rowcount


def delete_memory(session: Session, memory_id: int, *, owner_key: str) -> int:
    """Owner-guarded DELETE; returns the rowcount (0 = not found/not yours)."""
    result = session.execute(
        delete(AgentMemoryRecord).where(
            AgentMemoryRecord.id == memory_id,
            AgentMemoryRecord.owner_key == owner_key,
        )
    )
    return result.rowcount


def delete_all_memories(session: Session) -> int:
    """Remove every memory row (test helper); returns rows deleted."""
    result = session.execute(delete(AgentMemoryRecord))
    return result.rowcount
