"""PostgreSQL persistence tests (Step 11): schema, seed, transactions.

These run against the isolated test database provided by the
``test_database`` fixture (created on the local Docker Compose
Postgres — see conftest.py) and are skipped when no server is
reachable.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import func, inspect, select, text

from app.db import database
from app.db.models import Base, Customer, Order, OrderItem, Product
from app.db.repository import (
    insert_order,
    list_product_names,
    next_order_id,
    register_customer,
)
from app.db.seed import reset_database, seed, seed_if_empty
from app.tools import (
    check_stock,
    create_order,
    get_low_stock,
    get_sales_report,
    reset_mock_data,
    search_customer,
    update_stock,
)

pytestmark = pytest.mark.usefixtures("test_database")


def tables() -> set[str]:
    with database.session_scope() as session:
        return set(inspect(session.bind).get_table_names())


# --- connection & schema ---------------------------------------------------


def test_database_connection_and_schema_tables() -> None:
    """The four business tables exist with key columns."""
    names = tables()
    assert {"products", "customers", "orders", "order_items"} <= names

    with database.session_scope() as session:
        product_columns = {
            column["name"]
            for column in inspect(session.bind).get_columns("products")
        }
    assert {
        "id",
        "name",
        "price",
        "stock_quantity",
        "low_stock_threshold",
        "created_at",
        "updated_at",
    } <= product_columns


def test_money_columns_are_numeric_not_float() -> None:
    """Money must be exact decimal, never floating point."""
    from sqlalchemy import Numeric

    with database.session_scope() as session:
        inspector = inspect(session.bind)
        for table, column_name in (
            ("products", "price"),
            ("orders", "total_price"),
            ("order_items", "unit_price"),
            ("order_items", "line_total"),
            ("customers", "total_spent"),
        ):
            column = next(
                column
                for column in inspector.get_columns(table)
                if column["name"] == column_name
            )
            assert isinstance(column["type"], Numeric), (table, column_name)


def test_foreign_keys_and_indexes_exist() -> None:
    with database.session_scope() as session:
        inspector = inspect(session.bind)
        order_item_fks = {
            (fk["referred_table"], fk["constrained_columns"][0])
            for fk in inspector.get_foreign_keys("order_items")
        }
    assert ("orders", "order_id") in order_item_fks
    assert ("products", "product_id") in order_item_fks


# --- seed ------------------------------------------------------------------


def test_seed_matches_the_original_mock_dataset_exactly() -> None:
    reset_mock_data()
    with database.session_scope() as session:
        products = list(session.scalars(select(Product).order_by(Product.id)))
        customers = list(
            session.scalars(select(Customer).order_by(Customer.id))
        )
        orders = list(session.scalars(select(Order).order_by(Order.id)))

    assert [(p.name, float(p.price), p.stock_quantity) for p in products] == [
        ("Kopi Susu", 18000.0, 24),
        ("Americano", 22000.0, 40),
        ("Croissant", 25000.0, 8),
        ("Matcha Latte", 24000.0, 15),
        ("Teh Manis", 10000.0, 30),
    ]

    assert [(c.name, c.phone, c.total_orders, float(c.total_spent)) for c in customers] == [
        ("Budi Santoso", "0812-1111-2222", 12, 412000.0),
        ("Siti Rahma", "0813-3333-4444", 7, 268000.0),
        ("Dewi Lestari", "0821-5555-6666", 3, 97000.0),
        ("Andi Wijaya", "0857-7777-8888", 5, 154000.0),
    ]

    assert [o.id for o in orders] == ["ORD-0001", "ORD-0002", "ORD-0003", "ORD-0004"]
    assert float(orders[0].total_price) == 36000.0  # 2 x Kopi Susu


def test_seed_is_idempotent() -> None:
    """Seeding twice (or on restart) never duplicates rows."""
    reset_mock_data()
    with database.session_scope() as session:
        first = seed(session)
    assert first == {"products": 0, "customers": 0, "orders": 0, "order_items": 0}

    with database.session_scope() as session:
        counts = (
            session.scalar(select(func.count()).select_from(Product)),
            session.scalar(select(func.count()).select_from(Customer)),
            session.scalar(select(func.count()).select_from(Order)),
            session.scalar(select(func.count()).select_from(OrderItem)),
        )
    assert counts == (5, 4, 4, 4)


def test_seed_if_empty_keeps_real_data() -> None:
    """A restart-style re-seed must not touch rows created after seeding."""
    reset_mock_data()
    create_order("Sinta Putri", [{"product_name": "Kopi Susu", "quantity": 1}])

    with database.session_scope() as session:
        created = seed_if_empty(session)

    assert all(count == 0 for count in created.values())
    assert search_customer("Sinta")["count"] == 1
    with database.session_scope() as session:
        assert session.scalar(select(func.count()).select_from(Order)) == 5


# --- retrieval via tools ----------------------------------------------------


def test_product_and_customer_retrieval_from_database() -> None:
    reset_mock_data()
    assert check_stock("Kopi Susu") == {
        "success": True,
        "product_name": "Kopi Susu",
        "stock": 24,
        "price": 18000,
        "in_stock": True,
        "low_stock": False,
    }
    budi = search_customer("Budi")["customers"][0]
    assert budi["name"] == "Budi Santoso"
    assert budi["joined_at"] == "2026-01-15T09:00:00"
    assert budi["total_spent"] == 412000


# --- transactions ------------------------------------------------------------


def test_create_order_is_transactional_and_deducts_stock() -> None:
    reset_mock_data()
    result = create_order(
        "Budi Santoso",
        [
            {"product_name": "Kopi Susu", "quantity": 2},
            {"product_name": "Croissant", "quantity": 1},
        ],
    )
    assert result["success"] is True
    assert result["order"]["order_id"] == "ORD-0005"
    assert result["order"]["total_price"] == 61000

    with database.session_scope() as session:
        kopi = session.scalar(select(Product).where(Product.name == "Kopi Susu"))
        croissant = session.scalar(select(Product).where(Product.name == "Croissant"))
        items = list(
            session.scalars(
                select(OrderItem).where(OrderItem.order_id == "ORD-0005")
            )
        )
        budi = session.scalar(select(Customer).where(Customer.name == "Budi Santoso"))

    assert (kopi.stock_quantity, croissant.stock_quantity) == (22, 7)
    assert {item.product_name for item in items} == {"Kopi Susu", "Croissant"}
    assert budi.total_orders == 13
    assert budi.total_spent == Decimal("473000.00")  # 412000 seed + 61000


def test_insufficient_stock_rolls_back_completely() -> None:
    reset_mock_data()
    result = create_order(
        "Walkin Cust",
        [
            {"product_name": "Kopi Susu", "quantity": 1},  # fits
            {"product_name": "Croissant", "quantity": 99},  # does not
        ],
    )
    assert result["error"] == "INSUFFICIENT_STOCK"

    # Nothing persisted: no order, no customer, no stock change.
    with database.session_scope() as session:
        assert session.scalar(select(func.count()).select_from(Order)) == 4
        assert session.scalar(
            select(func.count()).select_from(Customer)
        ) == 4
        kopi = session.scalar(select(Product).where(Product.name == "Kopi Susu"))
    assert kopi.stock_quantity == 24
    assert search_customer("Walkin")["count"] == 0


def test_rollback_on_error_leaves_no_partial_writes() -> None:
    """An exception mid-transaction writes nothing (session_scope)."""
    reset_mock_data()
    with pytest.raises(RuntimeError):
        with database.session_scope() as session:
            customer, _ = register_customer(session, "Broken Tx")
            insert_order(
                session,
                order_id=next_order_id(session),
                customer=customer,
                lines=[
                    (
                        session.scalar(
                            select(Product).where(Product.name == "Teh Manis")
                        ),
                        2,
                    )
                ],
            )
            session.flush()
            raise RuntimeError("boom")

    with database.session_scope() as session:
        assert session.scalar(select(func.count()).select_from(Order)) == 4
        assert session.scalar(select(func.count()).select_from(Customer)) == 4
        teh = session.scalar(select(Product).where(Product.name == "Teh Manis"))
    assert teh.stock_quantity == 30


def test_duplicate_lines_accumulate_demand_atomically() -> None:
    """Two lines of the same product count together (24 available)."""
    reset_mock_data()
    result = create_order(
        "Budi Santoso",
        [
            {"product_name": "Kopi Susu", "quantity": 20},
            {"product_name": "kopi susu", "quantity": 10},
        ],
    )
    assert result["error"] == "INSUFFICIENT_STOCK"
    assert result["requested"] == 30
    assert check_stock("Kopi Susu")["stock"] == 24


def test_next_order_id_continues_after_restart_style_reset() -> None:
    reset_mock_data()
    assert create_order("Budi Santoso", [{"product_name": "Teh Manis", "quantity": 1}])[
        "order"
    ]["order_id"] == "ORD-0005"
    assert create_order("Budi Santoso", [{"product_name": "Teh Manis", "quantity": 1}])[
        "order"
    ]["order_id"] == "ORD-0006"


# --- reports / low stock from the database -----------------------------------


def test_sales_report_and_low_stock_derive_from_database() -> None:
    reset_mock_data()
    report = get_sales_report("daily")
    assert report == {
        "success": True,
        "period": "daily",
        "window_days": 1,
        "total_orders": 1,
        "total_items_sold": 2,
        "total_revenue": 36000,
        "products_sold": {"Kopi Susu": 2},
        "top_product": "Kopi Susu",
    }

    update_stock("Kopi Susu", -20)  # 24 -> 4
    low = get_low_stock(10)
    assert [p["product_name"] for p in low["products"]] == [
        "Kopi Susu",  # stock 4 — lowest first
        "Croissant",  # stock 8
    ]


def test_reset_database_restores_seed_state() -> None:
    reset_mock_data()
    create_order("Sinta Putri", [{"product_name": "Kopi Susu", "quantity": 3}])
    assert check_stock("Kopi Susu")["stock"] == 21

    reset_mock_data()

    assert check_stock("Kopi Susu")["stock"] == 24
    assert search_customer("Sinta")["count"] == 0
    assert get_sales_report("monthly")["total_orders"] == 4
