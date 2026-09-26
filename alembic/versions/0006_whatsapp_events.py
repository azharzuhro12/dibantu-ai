"""WhatsApp webhook idempotency ledger (Step 17).

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-26

One table backs duplicate-suppression for the real WhatsApp Cloud API
webhook: Meta can deliver the same event more than once, and every
incoming message is claimed here (atomic INSERT ... ON CONFLICT DO
NOTHING on the unique ``message_id``) BEFORE the agent runs, so a
redelivery can never execute the agent twice. The table stores ids and
timestamps only — never message content, never tokens.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "whatsapp_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("message_id", sa.String(length=128), nullable=False),
        sa.Column("sender_id", sa.String(length=40), nullable=False),
        sa.Column("received_at", sa.DateTime(), nullable=False),
        sa.Column("processed_at", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("message_id"),
    )
    op.create_index(
        op.f("ix_whatsapp_events_status"), "whatsapp_events", ["status"]
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_whatsapp_events_status"), table_name="whatsapp_events")
    op.drop_table("whatsapp_events")
