"""Persistent agent memories table (Step 14).

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-25

Gives the agent a durable, owner-scoped store for explicit business/
user facts (preferences, customer context, business context,
instructions). Memory is context only: nothing in this table ever
executes a business action, and ``owner_key`` is application-level
ownership, not authentication.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_memories",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("owner_key", sa.String(length=120), nullable=False),
        sa.Column("memory_type", sa.String(length=30), nullable=False),
        sa.Column("content", sa.String(length=2000), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "memory_type IN ('preference', 'customer_context', "
            "'business_context', 'instruction')",
            name="ck_agent_memories_type",
        ),
    )
    op.create_index(
        "ix_agent_memories_owner_key", "agent_memories", ["owner_key"]
    )


def downgrade() -> None:
    op.drop_index("ix_agent_memories_owner_key", table_name="agent_memories")
    op.drop_table("agent_memories")
