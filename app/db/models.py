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
    Numeric,
    String,
    column,
    func,
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


def money(value: Decimal) -> int | float:
    """Convert a Numeric money value to a JSON-friendly number.

    IDR amounts are whole numbers, so almost everything becomes a plain
    ``int`` — keeping API payloads byte-identical to the in-memory era.
    """
    if value == value.to_integral_value():
        return int(value)
    return float(value)
