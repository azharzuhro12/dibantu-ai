"""Unit tests for the mock business tools (Step 3)."""

from __future__ import annotations

import pytest

from app.tools import (
    check_stock,
    create_order,
    get_low_stock,
    get_sales_report,
    reset_mock_data,
    search_customer,
    update_stock,
)


@pytest.fixture(autouse=True)
def fresh_store(test_database):
    """Start every test from the seeded store (PostgreSQL test DB)."""
    reset_mock_data()
    yield


# --- check_stock -----------------------------------------------------------


def test_check_stock_success():
    result = check_stock("Kopi Susu")
    assert result["success"] is True
    assert result["product_name"] == "Kopi Susu"
    assert result["stock"] == 24
    assert result["price"] == 18000
    assert result["in_stock"] is True


def test_check_stock_matches_case_insensitively():
    result = check_stock("  kopi susu ")
    assert result["success"] is True
    assert result["product_name"] == "Kopi Susu"


def test_check_stock_flags_low_stock():
    assert check_stock("Croissant")["low_stock"] is True
    assert check_stock("Americano")["low_stock"] is False


def test_check_stock_product_not_found():
    result = check_stock("Es Kopi Ninja")
    assert result["success"] is False
    assert result["error"] == "PRODUCT_NOT_FOUND"
    assert "Kopi Susu" in result["available_products"]


def test_check_stock_rejects_blank_name():
    result = check_stock("   ")
    assert result["success"] is False
    assert result["error"] == "VALIDATION_ERROR"


# --- search_customer -------------------------------------------------------


def test_search_customer_success():
    result = search_customer("budi")
    assert result["success"] is True
    assert result["count"] == 1
    assert result["customers"][0]["name"] == "Budi Santoso"
    assert result["customers"][0]["phone"] == "0812-1111-2222"


def test_search_customer_not_found():
    result = search_customer("Zelda")
    assert result["success"] is True
    assert result["count"] == 0
    assert result["customers"] == []


# --- create_order ----------------------------------------------------------


def test_create_order_success():
    result = create_order(
        "Sinta Putri",
        [
            {"product_name": "Kopi Susu", "quantity": 2},
            {"product_name": "Croissant", "quantity": 1},
        ],
    )
    assert result["success"] is True
    assert result["new_customer"] is True

    order = result["order"]
    assert order["order_id"] == "ORD-0005"
    assert order["customer_name"] == "Sinta Putri"
    assert order["total_price"] == 2 * 18000 + 1 * 25000

    # Stock is deducted and the walk-in customer becomes searchable.
    assert check_stock("Kopi Susu")["stock"] == 22
    assert check_stock("Croissant")["stock"] == 7
    found = search_customer("Sinta")
    assert found["count"] == 1
    assert found["customers"][0]["total_orders"] == 1


def test_create_order_updates_existing_customer_stats():
    result = create_order(
        "Budi Santoso", [{"product_name": "Teh Manis", "quantity": 2}]
    )
    assert result["success"] is True
    assert result["new_customer"] is False
    customer = search_customer("Budi Santoso")["customers"][0]
    assert customer["total_orders"] == 13  # seeded 12 + this order
    assert customer["total_spent"] == 412000 + 20000


def test_create_order_insufficient_stock_is_atomic():
    result = create_order(
        "Budi Santoso",
        [
            {"product_name": "Kopi Susu", "quantity": 1},  # would fit
            {"product_name": "Croissant", "quantity": 99},  # does not
        ],
    )
    assert result["success"] is False
    assert result["error"] == "INSUFFICIENT_STOCK"
    assert result["product_name"] == "Croissant"
    assert result["available"] == 8
    # Nothing was applied: the fitting line kept its stock and no
    # order was recorded.
    assert check_stock("Kopi Susu")["stock"] == 24
    assert get_sales_report("daily")["total_orders"] == 1


def test_create_order_unknown_product():
    result = create_order(
        "Budi Santoso", [{"product_name": "Pizza", "quantity": 1}]
    )
    assert result["success"] is False
    assert result["error"] == "PRODUCT_NOT_FOUND"


def test_create_order_rejects_empty_items():
    result = create_order("Budi Santoso", [])
    assert result["success"] is False
    assert result["error"] == "VALIDATION_ERROR"


def test_create_order_rejects_non_positive_quantity():
    result = create_order(
        "Budi Santoso", [{"product_name": "Kopi Susu", "quantity": 0}]
    )
    assert result["success"] is False
    assert result["error"] == "VALIDATION_ERROR"


# --- update_stock ----------------------------------------------------------


def test_update_stock_restock_success():
    result = update_stock("Croissant", 10)
    assert result["success"] is True
    assert result["previous_stock"] == 8
    assert result["new_stock"] == 18
    assert check_stock("Croissant")["stock"] == 18


def test_update_stock_deduct_success():
    result = update_stock("teh manis", -5)
    assert result["success"] is True
    assert result["new_stock"] == 25


def test_update_stock_never_goes_negative():
    result = update_stock("Croissant", -20)
    assert result["success"] is False
    assert result["error"] == "NEGATIVE_STOCK"
    assert check_stock("Croissant")["stock"] == 8  # unchanged


def test_update_stock_product_not_found():
    result = update_stock("Pizza", 5)
    assert result["success"] is False
    assert result["error"] == "PRODUCT_NOT_FOUND"


# --- get_low_stock ---------------------------------------------------------


def test_get_low_stock_success():
    result = get_low_stock(16)
    assert result["success"] is True
    assert result["count"] == 2
    names = [product["product_name"] for product in result["products"]]
    assert names == ["Croissant", "Matcha Latte"]  # lowest stock first


def test_get_low_stock_can_return_empty():
    result = get_low_stock(1)
    assert result["success"] is True
    assert result["products"] == []


# --- get_sales_report ------------------------------------------------------


def test_get_sales_report_daily():
    result = get_sales_report("daily")
    assert result["success"] is True
    assert result["period"] == "daily"
    assert result["total_orders"] == 1
    assert result["total_revenue"] == 36000
    assert result["top_product"] == "Kopi Susu"


def test_get_sales_report_weekly_and_monthly():
    weekly = get_sales_report("weekly")
    assert weekly["total_orders"] == 2
    assert weekly["total_revenue"] == 60000

    monthly = get_sales_report("monthly")
    assert monthly["total_orders"] == 4
    assert monthly["total_revenue"] == 176000


def test_get_sales_report_counts_new_orders():
    create_order("Dewi Lestari", [{"product_name": "Teh Manis", "quantity": 1}])
    result = get_sales_report("daily")
    assert result["total_orders"] == 2
    assert result["total_revenue"] == 36000 + 10000


def test_get_sales_report_invalid_period():
    result = get_sales_report("yearly")
    assert result["success"] is False
    assert result["error"] == "INVALID_PERIOD"
