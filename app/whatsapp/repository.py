"""PostgreSQL-backed idempotency for WhatsApp webhook messages.

Meta can deliver the same webhook event more than once (retries,
replays). Every incoming message id (the wamid) is *claimed* here with
one atomic ``INSERT ... ON CONFLICT DO NOTHING`` before the agent is
allowed to run: the insert succeeding means this worker is the first
and only one to process the message; a conflict means a duplicate that
must be ignored. Because the claim lives in PostgreSQL (not memory),
deduplication survives restarts and works across workers.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.db.models import WhatsAppEventRecord

__all__ = ["claim_message", "mark_failed", "mark_processed"]


def claim_message(session: Session, *, message_id: str, sender_id: str) -> bool:
    """Atomically claim ``message_id``; False when it was seen before.

    True means the caller owns the message and must process it; False
    means a duplicate delivery (or a worker already mid-flight) — the
    caller must NOT run the agent again. Ownership is decided by
    ``RETURNING``: the conflict-dodged insert returns a row only when
    this call was the one that inserted it (``rowcount`` is unreliable
    here — psycopg reports -1 for ON CONFLICT statements).
    """
    statement = (
        pg_insert(WhatsAppEventRecord)
        .values(
            message_id=message_id,
            sender_id=sender_id,
            received_at=datetime.now(),
            status="received",
        )
        .on_conflict_do_nothing(index_elements=[WhatsAppEventRecord.message_id])
        .returning(WhatsAppEventRecord.message_id)
    )
    inserted = session.execute(statement).scalar_one_or_none()
    return inserted is not None


def mark_processed(session: Session, message_id: str) -> None:
    """Record that the claimed message was fully handled."""
    _mark_status(session, message_id=message_id, status="processed")


def mark_failed(session: Session, message_id: str) -> None:
    """Record that processing failed (agent error or send error)."""
    _mark_status(session, message_id=message_id, status="failed")


def _mark_status(session: Session, *, message_id: str, status: str) -> None:
    session.execute(
        update(WhatsAppEventRecord)
        .where(WhatsAppEventRecord.message_id == message_id)
        .values(status=status, processed_at=datetime.now())
    )
