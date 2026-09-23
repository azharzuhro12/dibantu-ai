"""Data access layer for the DibantuAI business tools.

Every query the tools need lives here, so the tools contain no raw
SQL and the ORM details stay in one place. All functions take an open
:class:`~sqlalchemy.orm.Session`; transaction boundaries are owned by
the caller (the tools use ``session_scope``).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.db.models import Customer, Order, OrderItem, Product, money


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------


def list_product_names(session: Session) -> list[str]:
    """All product names in catalog (seed) order."""
    return list(session.scalars(select(Product.name).order_by(Product.id)))


def find_product(session: Session, name: str) -> Product | None:
    """Return the product matching ``name`` case-insensitively, or None."""
    wanted = name.strip().casefold()
    return session.scalar(
        select(Product).where(func.lower(Product.name) == wanted)
    )


def lock_products_by_ids(session: Session, ids: list[int]) -> list[Product]:
    """Lock product rows FOR UPDATE (ordered by id to avoid deadlocks)."""
    return list(
        session.scalars(
            select(Product)
            .where(Product.id.in_(ids))
            .order_by(Product.id)
            .with_for_update()
        )
    )


def low_stock_products(session: Session, threshold: int) -> list[Product]:
    """Products with stock strictly below ``threshold``, lowest first.

    Ties keep catalog order (``id``), matching the previous in-memory
    sorted()-is-stable behavior.
    """
    return list(
        session.scalars(
            select(Product)
            .where(Product.stock_quantity < threshold)
            .order_by(Product.stock_quantity, Product.id)
        )
    )


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------


def find_customer(session: Session, name: str) -> Customer | None:
    """Return the customer matching ``name`` case-insensitively, or None."""
    wanted = name.strip().casefold()
    return session.scalar(
        select(Customer).where(func.lower(Customer.name) == wanted)
    )


def register_customer(
    session: Session, name: str
) -> tuple[Customer, bool]:
    """Find a customer by case-insensitive name or register a new one."""
    existing = find_customer(session, name)
    if existing is not None:
        return existing, False
    customer = Customer(
        name=name,
        phone=None,
        email=None,
        joined_at=datetime.now(),
        total_orders=0,
        total_spent=0,
    )
    session.add(customer)
    session.flush()  # assign id before the caller reads it
    return customer, True


def search_customers(session: Session, query: str) -> list[Customer]:
    """Customers whose name contains ``query`` (case-insensitive), in
    registration order."""
    return list(
        session.scalars(
            select(Customer)
            .where(func.lower(Customer.name).contains(query.casefold()))
            .order_by(Customer.id)
        )
    )


def customer_as_dict(customer: Customer) -> dict[str, Any]:
    """Serialize a customer exactly like the old in-memory record."""
    return {
        "name": customer.name,
        "phone": customer.phone,
        "email": customer.email,
        "joined_at": customer.joined_at.isoformat(timespec="seconds"),
        "total_orders": customer.total_orders,
        "total_spent": money(customer.total_spent),
    }


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


def next_order_id(session: Session) -> str:
    """The next ``ORD-####`` id (one above the highest existing number)."""
    numbers = session.scalars(
        select(func.substring(Order.id, 5)).order_by(Order.id)
    )
    highest = max((int(n) for n in numbers), default=0)
    return f"ORD-{highest + 1:04d}"


def insert_order(
    session: Session,
    *,
    order_id: str,
    customer: Customer,
    lines: list[tuple[Product, int]],
) -> Order:
    """Insert an order with its items and deduct stock for every line.

    The caller must have locked the product rows and verified stock.
    Aggregates on the customer are updated in the same transaction.
    """
    total = 0
    order = Order(
        id=order_id,
        customer_id=customer.id,
        status="completed",
        total_price=0,
        created_at=datetime.now(),
    )
    session.add(order)
    session.flush()  # order must exist before the items reference it

    for product, quantity in lines:
        product.stock_quantity -= quantity
        line_total = product.price * quantity
        total += line_total
        session.add(
            OrderItem(
                order_id=order.id,
                product_id=product.id,
                product_name=product.name,
                quantity=quantity,
                unit_price=product.price,
                line_total=line_total,
            )
        )

    order.total_price = total
    customer.total_orders += 1
    customer.total_spent += total
    return order


def order_as_dict(order: Order) -> dict[str, Any]:
    """Serialize an order exactly like the old in-memory record."""
    return {
        "order_id": order.id,
        "customer_name": order.customer.name,
        "items": [
            {
                "product_name": item.product_name,
                "quantity": item.quantity,
                "unit_price": money(item.unit_price),
                "line_total": money(item.line_total),
            }
            for item in order.items
        ],
        "total_price": money(order.total_price),
        "created_at": order.created_at.isoformat(timespec="seconds"),
    }


def orders_since(session: Session, cutoff: datetime) -> list[Order]:
    """Orders created at/after ``cutoff``, oldest first (items loaded)."""
    return list(
        session.scalars(
            select(Order)
            .where(Order.created_at >= cutoff)
            .options(selectinload(Order.items))
            .order_by(Order.created_at, Order.id)
        )
    )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def sales_window(
    session: Session, *, now: datetime, window_days: int
) -> dict[str, Any]:
    """Aggregate the rolling sales window from orders/order_items.

    Mirrors the previous in-memory computation exactly, including the
    insertion-ordered ``products_sold`` dict and first-max top product.
    """
    orders = orders_since(session, now - timedelta(days=window_days))

    products_sold: dict[str, int] = {}
    total_revenue = 0
    total_items_sold = 0
    for order in orders:
        total_revenue += order.total_price
        for item in order.items:
            products_sold[item.product_name] = (
                products_sold.get(item.product_name, 0) + item.quantity
            )
            total_items_sold += item.quantity

    return {
        "total_orders": len(orders),
        "total_items_sold": total_items_sold,
        "total_revenue": money(total_revenue) if orders else 0,
        "products_sold": products_sold,
        "top_product": (
            max(products_sold, key=products_sold.get)
            if products_sold
            else None
        ),
    }
