"""API tests for GET /api/reports (read-only sales summary).

The endpoint must serve the exact same PostgreSQL aggregates the
``get_sales_report`` tool computes — same ``sales_window`` repository
helper, same completed-orders-only rule — so the report cards and the
chat answer can never disagree. These tests run against the isolated
scratch database (see conftest.py); the main business database is
never touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.db.database import session_scope
from app.db.models import Order, OrderItem
from app.main import app
from app.tools.business_tools import reset_mock_data


@pytest.fixture(autouse=True)
def _business_database(test_database):
    """Report reads run against the PostgreSQL test database."""
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


def _window(body: dict, period: str) -> dict:
    return next(w for w in body["windows"] if w["period"] == period)


def test_reports_summarize_seeded_database(client) -> None:
    response = client.get("/api/reports")

    assert response.status_code == 200
    body = response.json()
    assert body["generated_at"]  # always carries a computation timestamp
    assert [w["period"] for w in body["windows"]] == [
        "daily",
        "weekly",
        "monthly",
    ]

    # Seed ages: ORD-0001 2h, ORD-0002 3d, ORD-0003 10d, ORD-0004 20d.
    daily = _window(body, "daily")
    assert daily["total_orders"] == 1
    assert daily["total_items_sold"] == 2
    assert daily["total_revenue"] == 36000  # 2 x Kopi Susu @18000

    weekly = _window(body, "weekly")
    assert weekly["total_orders"] == 2
    assert weekly["total_items_sold"] == 3
    assert weekly["total_revenue"] == 60000  # + 1 x Matcha Latte @24000

    monthly = _window(body, "monthly")
    assert monthly["total_orders"] == 4
    assert monthly["total_items_sold"] == 11
    assert monthly["total_revenue"] == 176000  # + 3x Americano, 5x Teh Manis

    # Every seeded order is completed.
    assert body["status_summary"] == {"by_status": {"completed": 4}, "total": 4}


def test_reports_product_aggregation_matches_order_items(client) -> None:
    body = client.get("/api/reports").json()
    monthly = _window(body, "monthly")

    # The API aggregation equals a direct GROUP BY over order_items
    # joined to completed orders — the tables are the source of truth.
    with session_scope() as session:
        expected = dict(
            session.execute(
                select(OrderItem.product_name, func.sum(OrderItem.quantity))
                .join(Order, OrderItem.order_id == Order.id)
                .where(Order.status == "completed")
                .group_by(OrderItem.product_name)
            ).all()
        )
    assert monthly["products_sold"] == expected == {
        "Kopi Susu": 2,
        "Matcha Latte": 1,
        "Americano": 3,
        "Teh Manis": 5,
    }
    assert monthly["top_product"] == "Teh Manis"  # 5 units, the maximum


def test_reports_exclude_refunded_orders_from_windows(client) -> None:
    # Mark today's order refunded (direct ORM write on the scratch DB).
    with session_scope() as session:
        session.get(Order, "ORD-0001").status = "refunded"

    body = client.get("/api/reports").json()

    # Status breakdown reports every stored status as-is...
    assert body["status_summary"]["by_status"] == {
        "completed": 3,
        "refunded": 1,
    }
    assert body["status_summary"]["total"] == 4
    # ...while sales windows keep counting completed orders only.
    daily = _window(body, "daily")
    assert daily["total_orders"] == 0
    assert daily["total_revenue"] == 0
    assert daily["top_product"] is None
    monthly = _window(body, "monthly")
    assert monthly["total_orders"] == 3
    assert monthly["total_revenue"] == 140000  # 176000 - 36000


def test_reports_empty_database_returns_zeroed_report(client) -> None:
    with session_scope() as session:
        # Delete in FK-safe order (items -> orders); customers/products stay.
        session.query(OrderItem).delete()
        session.query(Order).delete()

    response = client.get("/api/reports")

    assert response.status_code == 200
    body = response.json()
    assert body["status_summary"] == {"by_status": {}, "total": 0}
    for window in body["windows"]:
        assert window["total_orders"] == 0
        assert window["total_items_sold"] == 0
        assert window["total_revenue"] == 0
        assert window["products_sold"] == {}
        assert window["top_product"] is None


def test_reports_database_error_is_503_without_internals(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_sales_window(session, *, now, window_days):
        raise SQLAlchemyError("connection refused: hush-secret-host")

    monkeypatch.setattr(
        "app.db.repository.sales_window", broken_sales_window
    )

    response = client.get("/api/reports")

    assert response.status_code == 503
    assert response.json()["detail"] == "Reports are unavailable (database error)."
    # Internal details (host, driver message) never reach the client.
    assert "hush-secret-host" not in response.text


def test_reports_reflect_order_created_through_create_order(client) -> None:
    from app.tools.business_tools import create_order

    before = client.get("/api/reports").json()

    created = create_order(
        "Rina Melati",
        [{"product_name": "Croissant", "quantity": 1}],
    )
    assert created["success"] is True

    after = client.get("/api/reports").json()

    # The sale lands in every window that already had data, and the
    # totals move by exactly the order's amount.
    for period in ("daily", "weekly", "monthly"):
        was = _window(before, period)
        now = _window(after, period)
        assert now["total_orders"] == was["total_orders"] + 1
        assert now["total_revenue"] == was["total_revenue"] + 25000
        assert (
            now["products_sold"].get("Croissant")
            == was["products_sold"].get("Croissant", 0) + 1
        )
    # Teh Manis (5) stays the monthly top product; Croissant only sold 1.
    assert _window(after, "monthly")["top_product"] == "Teh Manis"
    assert after["status_summary"] == {
        "by_status": {"completed": 5},
        "total": 5,
    }
