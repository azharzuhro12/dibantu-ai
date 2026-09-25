"""Agent observability tables (Step 15).

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-25

Two append-oriented tables back the lightweight, PostgreSQL-only agent
tracing: one row per agent execution (``agent_runs``) and one row per
traced operation (``agent_events`` — LLM calls, tool calls, memory/RAG/
approval operations), tied together by the stable ``run_id``. The event
table deliberately carries no foreign key so every write is independent
(a partial tracing failure can never cascade). Records hold safe,
compact metadata only: no prompts, raw responses, chain-of-thought,
secrets, or stack traces.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=40), nullable=False),
        sa.Column("owner_key", sa.String(length=120), nullable=True),
        sa.Column("source", sa.String(length=30), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("request_preview", sa.String(length=160), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("error_type", sa.String(length=100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", name="uq_agent_runs_run_id"),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name="ck_agent_runs_status",
        ),
    )
    op.create_index("ix_agent_runs_created_at", "agent_runs", ["created_at"])
    op.create_index("ix_agent_runs_status", "agent_runs", ["status"])

    op.create_table(
        "agent_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=40), nullable=False),
        sa.Column("event_type", sa.String(length=30), nullable=False),
        sa.Column("event_name", sa.String(length=40), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("iteration", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("error_type", sa.String(length=100), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "status IN ('started', 'success', 'failed')",
            name="ck_agent_events_status",
        ),
        sa.CheckConstraint(
            "event_type IN ('RUN', 'LLM', 'TOOL', 'MEMORY', 'RAG', 'APPROVAL')",
            name="ck_agent_events_type",
        ),
    )
    op.create_index("ix_agent_events_run_id", "agent_events", ["run_id"])
    op.create_index("ix_agent_events_event_type", "agent_events", ["event_type"])
    op.create_index("ix_agent_events_status", "agent_events", ["status"])


def downgrade() -> None:
    op.drop_index("ix_agent_events_status", table_name="agent_events")
    op.drop_index("ix_agent_events_event_type", table_name="agent_events")
    op.drop_index("ix_agent_events_run_id", table_name="agent_events")
    op.drop_table("agent_events")
    op.drop_index("ix_agent_runs_status", table_name="agent_runs")
    op.drop_index("ix_agent_runs_created_at", table_name="agent_runs")
    op.drop_table("agent_runs")
