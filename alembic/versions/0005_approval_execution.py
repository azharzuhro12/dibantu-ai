"""Approval execution lifecycle columns (Step 16).

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-25

Extends the approvals state machine past the human decision:
``approved -> executing -> executed/failed``. The status CHECK is
replaced with a superset — every row written by earlier migrations
(``pending``/``approved``/``rejected``) remains valid, no data is
rewritten. New nullable columns record the execution timeline and its
outcome: ``execution_started_at``, ``executed_at``, a bounded
``execution_error`` string (business message only — never a traceback,
never credentials), and ``execution_result`` (the sensitive tool's
result dict). All are NULL until an execution is attempted, so pending
and decided-only rows are untouched.

Also adds ``orders.refunded_amount`` (default 0): the cumulative money
already refunded on an order, so partial refunds can never exceed the
order total across multiple approvals. Existing orders default to 0
(never refunded). Reversible in full.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_NEW_STATUS_CHECK = (
    "status IN ('pending', 'approved', 'rejected', "
    "'executing', 'executed', 'failed')"
)

_OLD_STATUS_CHECK = "status IN ('pending', 'approved', 'rejected')"


def upgrade() -> None:
    op.drop_constraint("ck_approvals_status", "approvals", type_="check")
    op.create_check_constraint(
        "ck_approvals_status", "approvals", _NEW_STATUS_CHECK
    )
    op.add_column(
        "approvals",
        sa.Column("execution_started_at", sa.DateTime(), nullable=True),
    )
    op.add_column("approvals", sa.Column("executed_at", sa.DateTime(), nullable=True))
    op.add_column(
        "approvals",
        sa.Column("execution_error", sa.String(length=500), nullable=True),
    )
    op.add_column("approvals", sa.Column("execution_result", sa.JSON(), nullable=True))
    op.add_column(
        "orders",
        sa.Column(
            "refunded_amount",
            sa.Numeric(12, 2),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )


def downgrade() -> None:
    op.drop_column("orders", "refunded_amount")
    op.drop_column("approvals", "execution_result")
    op.drop_column("approvals", "execution_error")
    op.drop_column("approvals", "executed_at")
    op.drop_column("approvals", "execution_started_at")
    # Rows using the new statuses must not survive a downgrade silently:
    # an execution in progress or finished collapses back to "approved"
    # (the decision is still known), never to "pending".
    op.execute(
        "UPDATE approvals SET status = 'approved' "
        "WHERE status IN ('executing', 'executed', 'failed')"
    )
    op.drop_constraint("ck_approvals_status", "approvals", type_="check")
    op.create_check_constraint(
        "ck_approvals_status", "approvals", _OLD_STATUS_CHECK
    )
