"""SQLAlchemy 2.x ORM tables for DibantuAI business data.

Money is stored as ``Numeric(12, 2)`` (never floating point); the tools
convert to plain int/float when building their JSON responses so the
API payloads stay identical to the previous in-memory implementation.

Timestamps are naive local datetimes, matching the original
``datetime.now().isoformat(timespec="seconds")`` output.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    column,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all DibantuAI tables."""


MONEY = Numeric(12, 2)


class Product(Base):
    """A catalog product with price, stock, and a low-stock threshold."""

    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    stock_quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    low_stock_threshold: Mapped[int] = mapped_column(
        Integer, nullable=False, default=10
    )
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )


class Customer(Base):
    """A customer record with the aggregates the tools report."""

    __tablename__ = "customers"
    __table_args__ = (
        # Registration matches names case-insensitively (Budi == budi).
        Index(
            "uq_customers_name_lower",
            func.lower(column("name")),
            unique=True,
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(40))
    email: Mapped[str | None] = mapped_column(String(120))
    joined_at: Mapped[datetime] = mapped_column(nullable=False)
    total_orders: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_spent: Mapped[Decimal] = mapped_column(
        MONEY, nullable=False, default=0
    )
    orders: Mapped[list["Order"]] = relationship(back_populates="customer")


class Order(Base):
    """One order. The id is the public ``ORD-####`` code (string PK)."""

    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="completed"
    )
    total_price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    refunded_amount: Mapped[Decimal] = mapped_column(
        MONEY,
        nullable=False,
        default=Decimal("0"),
        server_default=text("0"),
    )
    created_at: Mapped[datetime] = mapped_column(nullable=False, index=True)

    customer: Mapped[Customer] = relationship(back_populates="orders")
    items: Mapped[list["OrderItem"]] = relationship(
        back_populates="order", cascade="all, delete-orphan"
    )


class OrderItem(Base):
    """One order line: product, quantity, and the priced snapshot."""

    __tablename__ = "order_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[str] = mapped_column(
        ForeignKey("orders.id"), nullable=False, index=True
    )
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id"), nullable=False, index=True
    )
    product_name: Mapped[str] = mapped_column(String(120), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    line_total: Mapped[Decimal] = mapped_column(MONEY, nullable=False)

    order: Mapped[Order] = relationship(back_populates="items")


class ApprovalRecord(Base):
    """A human-in-the-loop approval request (Step 8 → Step 13 → Step 16).

    The surrogate ``id`` preserves true insertion order (``created_at``
    only has second precision); ``approval_id`` is the public ``apr-...``
    code used by the API, the agent messages, and the frontend. Payload
    is the exact JSON the agent wanted to execute — the immutable
    snapshot a human reviews AND the only input the executor ever runs
    (never re-supplied by a client). Since Step 16 the lifecycle
    continues past the decision: approved → executing → executed/failed,
    with the execution timestamps, a bounded error string (no
    tracebacks, no secrets), and the tool's result dict kept on the row.
    The agent itself still never executes sensitive actions directly.
    """

    __tablename__ = "approvals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    approval_id: Mapped[str] = mapped_column(
        String(20), nullable=False, unique=True
    )
    action: Mapped[str] = mapped_column(String(60), nullable=False)
    requested_by: Mapped[str] = mapped_column(String(120), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending", index=True
    )
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )
    decided_at: Mapped[datetime | None] = mapped_column()
    decision_reason: Mapped[str | None] = mapped_column(String(500))
    execution_started_at: Mapped[datetime | None] = mapped_column()
    executed_at: Mapped[datetime | None] = mapped_column()
    execution_error: Mapped[str | None] = mapped_column(String(500))
    execution_result: Mapped[dict | None] = mapped_column(JSON)


class AgentMemoryRecord(Base):
    """One persistent agent memory (Step 14): a structured, owner-scoped fact.

    ``owner_key`` is an application-level identity (WhatsApp sender,
    chat owner key, or "default") — deliberately NOT authentication.
    ``memory_type`` is constrained to a small practical set (see
    app/memory/manager.py and the 0003 migration CHECK). The surrogate
    ``id`` doubles as a recency ordering (insertion order).
    """

    __tablename__ = "agent_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_key: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    memory_type: Mapped[str] = mapped_column(String(30), nullable=False)
    content: Mapped[str] = mapped_column(String(2000), nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )


class AgentRunRecord(Base):
    """One traced agent execution (Step 15).

    ``run_id`` is the stable public id threaded through every event of
    the run. ``request_preview`` is a deliberately short prefix of the
    user message (never full prompts); ``error_type`` is an exception
    class name only — stack traces are never persisted.
    """

    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(40), nullable=False, unique=True)
    owner_key: Mapped[str | None] = mapped_column(String(120))
    source: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    request_preview: Mapped[str | None] = mapped_column(String(160))
    started_at: Mapped[datetime] = mapped_column(nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column()
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    error_type: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), index=True
    )


class AgentEventRecord(Base):
    """One traced operation inside an agent run (Step 15).

    ``run_id`` deliberately has no foreign key: the event log is
    append-only and each write stays independent of every other write,
    so a partial tracing failure can never cascade. The ``metadata``
    column (attribute ``event_metadata``) holds small, safe JSON only —
    never prompts, raw responses, chain-of-thought, or secrets.
    """

    __tablename__ = "agent_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    event_name: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    iteration: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column()
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    event_metadata: Mapped[dict | None] = mapped_column("metadata", JSON)
    error_type: Mapped[str | None] = mapped_column(String(100))


def money(value: Decimal) -> int | float:
    """Convert a Numeric money value to a JSON-friendly number.

    IDR amounts are whole numbers, so almost everything becomes a plain
    ``int`` — keeping API payloads byte-identical to the in-memory era.
    """
    if value == value.to_integral_value():
        return int(value)
    return float(value)
