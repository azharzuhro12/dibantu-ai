"""Persistent human-in-the-loop approvals table (Step 13).

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-25

Gives the approval store (previously an in-memory dict in
app/approval/manager.py) a durable PostgreSQL home: pending approvals
now survive backend and container restarts. The surrogate ``id`` keeps
insertion order (created_at has only second precision); the public
``apr-...`` code lives in ``approval_id``. The payload column stores
the exact JSON the agent requested so a human can review it — approving
records the decision only, nothing is ever auto-executed.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "approvals",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("approval_id", sa.String(length=20), nullable=False),
        sa.Column("action", sa.String(length=60), nullable=False),
        sa.Column("requested_by", sa.String(length=120), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
        sa.Column("decision_reason", sa.String(length=500), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("approval_id", name="uq_approvals_approval_id"),
        sa.CheckConstraint(
            "status IN ('pending', 'approved', 'rejected')",
            name="ck_approvals_status",
        ),
    )
    op.create_index("ix_approvals_status", "approvals", ["status"])


def downgrade() -> None:
    op.drop_index("ix_approvals_status", table_name="approvals")
    op.drop_table("approvals")
