"""API tests for GET /api/inventory (read-only product listing).

The endpoint must serve the exact same PostgreSQL rows the
``check_stock`` tool reads — same repository, same field semantics —
so the inventory table and the chat answer can never disagree. These
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
from app.db.models import Order, OrderItem, Product
from app.main import app
from app.tools.business_tools import check_stock, reset_mock_data


@pytest.fixture(autouse=True)
def _business_database(test_database):
    """Inventory reads run against the PostgreSQL test database."""
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


def test_inventory_lists_seeded_products(client) -> None:
    response = client.get("/api/inventory")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 5
    assert [p["name"] for p in body["products"]] == [
        "Kopi Susu",
        "Americano",
        "Croissant",
        "Matcha Latte",
        "Teh Manis",
    ]
    kopi = body["products"][0]
    assert kopi["price"] == 18000
    assert kopi["stock"] == 24
    assert kopi["low_stock_threshold"] == 10
    assert kopi["in_stock"] is True
    assert kopi["low_stock"] is False


def test_inventory_matches_check_stock_after_a_sale(client) -> None:
    # A real order through the public tool deducts stock in PostgreSQL.
    from app.tools.business_tools import create_order

    created = create_order(
        "Budi Santoso",
        [{"product_name": "Kopi Susu", "quantity": 5}],
    )
    assert created["success"] is True

    body = client.get("/api/inventory").json()
    api_row = next(p for p in body["products"] if p["name"] == "Kopi Susu")
    tool_row = check_stock("Kopi Susu")

    # Same source of truth: the table and the chat tool agree.
    assert api_row["stock"] == 19 == tool_row["stock"]
    assert api_row["price"] == tool_row["price"]
    assert api_row["in_stock"] == tool_row["in_stock"]
    assert api_row["low_stock"] == tool_row["low_stock"]


def test_inventory_low_and_out_of_stock_flags(client) -> None:
    from app.tools.business_tools import update_stock

    body = client.get("/api/inventory").json()
    croissant = next(p for p in body["products"] if p["name"] == "Croissant")
    assert croissant["stock"] == 8
    assert croissant["low_stock"] is True  # 0 < 8 <= 10 (same rule as check_stock)
    assert croissant["in_stock"] is True

    # Drain Teh Manis entirely: out of stock is NOT "low stock".
    update_stock("Teh Manis", -30)
    body = client.get("/api/inventory").json()
    teh = next(p for p in body["products"] if p["name"] == "Teh Manis")
    assert teh["stock"] == 0
    assert teh["in_stock"] is False
    assert teh["low_stock"] is False


def test_inventory_empty_database_returns_empty_list(client) -> None:
    with session_scope() as session:
        # Delete in FK-safe order (items -> orders -> products).
        session.query(OrderItem).delete()
        session.query(Order).delete()
        session.query(Product).delete()

    response = client.get("/api/inventory")

    assert response.status_code == 200
    body = response.json()
    assert body["products"] == []
    assert body["count"] == 0


def test_inventory_database_error_is_503_without_internals(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_list_products(session):
        raise SQLAlchemyError("connection refused: hush-secret-host")

    monkeypatch.setattr(
        "app.db.repository.list_products", broken_list_products
    )

    response = client.get("/api/inventory")

    assert response.status_code == 503
    assert response.json()["detail"] == "Inventory is unavailable (database error)."
    # Internal details (host, driver message) never reach the client.
    assert "hush-secret-host" not in response.text
