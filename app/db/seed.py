"""Seed the PostgreSQL database with the original mock business data.

The dataset below is the exact data that used to live in
``app/tools/business_tools.py`` — same products, prices, stock,
customers (with their aggregate stats), and the four historical orders
— moved verbatim, nothing invented. Seeding is idempotent: every row
is created only if its natural key is missing, so running it twice (or
on every container start) never duplicates data.

``reset_database`` additionally truncates first; it replaces the old
``reset_mock_data()`` semantics used by tests and the evaluation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db.models import Customer, Order, OrderItem, Product

__all__ = ["reset_database", "seed", "seed_if_empty"]

#: Original catalog seed (business_tools._SEED_PRODUCTS), verbatim:
#: canonical name -> price (IDR), stock, low-stock threshold.
SEED_PRODUCTS: list[tuple[str, int, int, int]] = [
    # name, price, stock, low_stock_threshold
    ("Kopi Susu", 18000, 24, 10),
    ("Americano", 22000, 40, 10),
    ("Croissant", 25000, 8, 10),
    ("Matcha Latte", 24000, 15, 10),
    ("Teh Manis", 10000, 30, 10),
]

#: Original customer registry (business_tools._seed_customers), verbatim:
#: name, phone, email, joined_at, total_orders, total_spent.
SEED_CUSTOMERS: list[tuple[str, str, str, str, int, int]] = [
    # name, phone, email, joined_at, total_orders, total_spent (IDR)
    (
        "Budi Santoso",
        "0812-1111-2222",
        "budi@example.com",
        "2026-01-15T09:00:00",
        12,
        412000,
    ),
    (
        "Siti Rahma",
        "0813-3333-4444",
        "siti@example.com",
        "2026-01-15T09:00:00",
        7,
        268000,
    ),
    (
        "Dewi Lestari",
        "0821-5555-6666",
        "dewi@example.com",
        "2026-01-15T09:00:00",
        3,
        97000,
    ),
    (
        "Andi Wijaya",
        "0857-7777-8888",
        "andi@example.com",
        "2026-01-15T09:00:00",
        5,
        154000,
    ),
]

#: Original order history (business_tools._seed_orders), verbatim:
#: order number, customer, product, quantity, hours before "now".
SEED_ORDERS: list[tuple[int, str, str, int, float]] = [
    (1, "Budi Santoso", "Kopi Susu", 2, 2),  # today
    (2, "Siti Rahma", "Matcha Latte", 1, 24 * 3),  # 3 days ago
    (3, "Andi Wijaya", "Americano", 3, 24 * 10),  # 10 days ago
    (4, "Dewi Lestari", "Teh Manis", 5, 24 * 20),  # 20 days ago
]


def seed(session: Session, now: datetime | None = None) -> dict[str, int]:
    """Insert any missing seed rows (idempotent); return insert counts.

    Existing rows are never modified, so real orders created after the
    first seed survive application restarts and re-seeds.
    """
    now = now or datetime.now()
    created = {"products": 0, "customers": 0, "orders": 0, "order_items": 0}

    for name, price, stock, threshold in SEED_PRODUCTS:
        exists = session.scalar(select(Product.id).where(Product.name == name))
        if exists is None:
            session.add(
                Product(
                    name=name,
                    price=Decimal(price),
                    stock_quantity=stock,
                    low_stock_threshold=threshold,
                )
            )
            created["products"] += 1
    session.flush()

    for name, phone, email, joined_at, orders, spent in SEED_CUSTOMERS:
        exists = session.scalar(select(Customer.id).where(Customer.name == name))
        if exists is None:
            session.add(
                Customer(
                    name=name,
                    phone=phone,
                    email=email,
                    joined_at=datetime.fromisoformat(joined_at),
                    total_orders=orders,
                    total_spent=Decimal(spent),
                )
            )
            created["customers"] += 1
    session.flush()

    prices = {name: Decimal(price) for name, price, _, _ in SEED_PRODUCTS}
    for number, customer, product, quantity, hours_ago in SEED_ORDERS:
        order_id = f"ORD-{number:04d}"
        if session.get(Order, order_id) is not None:
            continue
        unit_price = prices[product]
        customer_id = session.scalar(
            select(Customer.id).where(Customer.name == customer)
        )
        product_id = session.scalar(
            select(Product.id).where(Product.name == product)
        )
        session.add(
            Order(
                id=order_id,
                customer_id=customer_id,
                status="completed",
                total_price=unit_price * quantity,
                created_at=now - timedelta(hours=hours_ago),
            )
        )
        session.add(
            OrderItem(
                order_id=order_id,
                product_id=product_id,
                product_name=product,
                quantity=quantity,
                unit_price=unit_price,
                line_total=unit_price * quantity,
            )
        )
        created["orders"] += 1
        created["order_items"] += 1
    session.flush()
    return created


def seed_if_empty(
    session: Session, now: datetime | None = None
) -> dict[str, int]:
    """Seed only when the database has no products at all."""
    if session.scalar(select(Product.id).limit(1)) is not None:
        return {"products": 0, "customers": 0, "orders": 0, "order_items": 0}
    return seed(session, now=now)


def reset_database(session: Session, now: datetime | None = None) -> None:
    """Truncate everything and re-seed (tests / evaluation reset).

    Rows come back in seed order with fresh sequential ids (RESTART
    IDENTITY), so ordering-sensitive outputs — product lists, low-stock
    ties, customer search order — match the original in-memory store.
    """
    session.execute(
        text("TRUNCATE TABLE order_items, orders, customers, products "
             "RESTART IDENTITY CASCADE")
    )
    session.flush()
    seed(session, now=now)


def main() -> None:
    """CLI entrypoint: seed the configured database if it is empty."""
    from app.db.database import session_scope

    with session_scope() as session:
        created = seed_if_empty(session)
    if any(created.values()):
        print(f"Seeded database: {created}")
    else:
        print("Database already seeded — nothing to do.")


if __name__ == "__main__":
    main()
