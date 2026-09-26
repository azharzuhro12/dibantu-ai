"""API tests for GET /api/orders (read-only order history listing).

The endpoint must serve the exact same PostgreSQL rows the
``create_order`` tool writes — same repository, same money conversion —
so the orders table and the chat answer can never disagree. These
tests run against the isolated scratch database (see conftest.py);
the main business database is never touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.db.database import session_scope
from app.db.models import Order, OrderItem
from app.main import app
from app.tools.business_tools import check_stock, reset_mock_data


@pytest.fixture(autouse=True)
def _business_database(test_database):
    """Order reads run against the PostgreSQL test database."""
    yield


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _seeded_store():
    """Start every test from the seeded catalog (restored after)."""
    reset_mock_data()
    yield
    reset_mock_data()


def test_orders_lists_seeded_orders_newest_first(client) -> None:
    response = client.get("/api/orders")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 4
    # Seed ages: ORD-0001 2h ago ... ORD-0004 20 days ago -> newest first.
    assert [o["order_id"] for o in body["orders"]] == [
        "ORD-0001",
        "ORD-0002",
        "ORD-0003",
        "ORD-0004",
    ]
    first = body["orders"][0]
    assert first["customer"]["name"] == "Budi Santoso"
    assert first["total_price"] == 36000  # 2 x Kopi Susu @18000
    assert first["status"] == "completed"
    assert first["created_at"]  # ISO timestamp is always present


def test_orders_contain_customer_and_item_information(client) -> None:
    body = client.get("/api/orders").json()

    order = next(o for o in body["orders"] if o["order_id"] == "ORD-0003")
    # Customer block: the info stored on the customer row.
    assert order["customer"]["name"] == "Andi Wijaya"
    assert order["customer"]["phone"] == "0857-7777-8888"
    assert order["customer"]["email"] == "andi@example.com"
    # Items: the priced snapshot — name, quantity, unit price, subtotal.
    assert order["items"] == [
        {
            "product_name": "Americano",
            "quantity": 3,
            "unit_price": 22000,
            "line_total": 66000,
        }
    ]
    assert order["total_price"] == 66000


def test_orders_empty_database_returns_empty_list(client) -> None:
    with session_scope() as session:
        # Delete in FK-safe order (items -> orders); customers/products stay.
        session.query(OrderItem).delete()
        session.query(Order).delete()

    response = client.get("/api/orders")

    assert response.status_code == 200
    body = response.json()
    assert body["orders"] == []
    assert body["count"] == 0


def test_orders_database_error_is_503_without_internals(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_list_orders(session):
        raise SQLAlchemyError("connection refused: hush-secret-host")

    monkeypatch.setattr(
        "app.db.repository.list_orders", broken_list_orders
    )

    response = client.get("/api/orders")

    assert response.status_code == 503
    assert response.json()["detail"] == "Orders are unavailable (database error)."
    # Internal details (host, driver message) never reach the client.
    assert "hush-secret-host" not in response.text


def test_order_created_through_create_order_appears_in_api(client) -> None:
    from app.tools.business_tools import create_order

    created = create_order(
        "Rina Melati",
        [
            {"product_name": "Kopi Susu", "quantity": 3},
            {"product_name": "Croissant", "quantity": 1},
        ],
    )
    assert created["success"] is True
    assert created["order"]["order_id"] == "ORD-0005"

    body = client.get("/api/orders").json()

    # The new order leads the list (newest first) with the same payload
    # the tool returned — one source of truth for the table and the chat.
    assert body["count"] == 5
    api_order = body["orders"][0]
    assert api_order["order_id"] == "ORD-0005"
    assert api_order["customer"]["name"] == "Rina Melati"
    assert api_order["status"] == "completed"
    assert api_order["items"] == [
        {"product_name": "Kopi Susu", "quantity": 3,
         "unit_price": 18000, "line_total": 54000},
        {"product_name": "Croissant", "quantity": 1,
         "unit_price": 25000, "line_total": 25000},
    ]
    assert api_order["total_price"] == 79000 == created["order"]["total_price"]


def test_orders_and_inventory_stay_consistent_after_a_sale(client) -> None:
    from app.tools.business_tools import create_order

    created = create_order(
        "Budi Santoso",
        [{"product_name": "Kopi Susu", "quantity": 5}],
    )
    assert created["success"] is True

    orders_body = client.get("/api/orders").json()
    inventory_body = client.get("/api/inventory").json()

    # The sale is visible in both read-only endpoints...
    api_order = next(
        o for o in orders_body["orders"] if o["order_id"] == "ORD-0005"
    )
    assert api_order["items"][0]["quantity"] == 5
    api_stock = next(
        p for p in inventory_body["products"] if p["name"] == "Kopi Susu"
    )["stock"]
    # ...with exactly the stock the transaction deducted (24 - 5).
    assert api_stock == 19 == check_stock("Kopi Susu")["stock"]
    # The stored total matches the priced line (no drift between tables).
    line = api_order["items"][0]
    assert line["unit_price"] * line["quantity"] == line["line_total"]
    assert api_order["total_price"] == line["line_total"]
