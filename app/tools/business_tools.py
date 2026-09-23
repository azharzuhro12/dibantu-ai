"""Business tools for DibantuAI (Step 3 → Step 11: PostgreSQL-backed).

Every tool keeps the exact input/output contract of the original
in-memory implementation — validations, error codes, messages, and
success payloads are unchanged — but all state now lives in
PostgreSQL via the ``app.db`` repository layer. The agent and its tool
schemas are unaware of the database.

Failures (validation errors, unknown products, insufficient stock,
database unavailability, ...) are still reported as
``{"success": False, "error": ..., "message": ...}`` dicts instead of
exceptions, so the agent can relay them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.db import database, repository
from app.db.database import DatabaseNotConfiguredError
from app.db.models import Product, money

__all__ = [
    "check_stock",
    "create_order",
    "get_low_stock",
    "get_sales_report",
    "reset_mock_data",
    "search_customer",
    "update_stock",
]

#: Periods understood by get_sales_report (rolling windows ending "now").
VALID_PERIODS: tuple[str, ...] = ("daily", "weekly", "monthly")

#: Default low-stock flag level used at seed time; check_stock reads the
#: per-product ``low_stock_threshold`` column (seeded to this value).
LOW_STOCK_THRESHOLD = 10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validation_error(message: str) -> dict[str, Any]:
    return {"success": False, "error": "VALIDATION_ERROR", "message": message}


def _product_not_found(name: str, available: list[str]) -> dict[str, Any]:
    return {
        "success": False,
        "error": "PRODUCT_NOT_FOUND",
        "message": f"Produk '{name.strip()}' tidak ditemukan.",
        "available_products": available,
    }


def _database_error(exc: Exception) -> dict[str, Any]:
    """Translate a database failure into an agent-visible error dict.

    The exception text is deliberately NOT included: connection errors
    can embed the full DATABASE_URL (with credentials).
    """
    if isinstance(exc, DatabaseNotConfiguredError):
        message = "DATABASE_URL is not configured."
    else:
        message = (
            "Business database is unreachable. Try again once it is up."
        )
    return {"success": False, "error": "DATABASE_ERROR", "message": message}


def _lock_product(session, product_name: str) -> Product | None:
    """Find a product case-insensitively and lock its row FOR UPDATE."""
    wanted = product_name.strip().casefold()
    return session.scalar(
        select(Product)
        .where(func.lower(Product.name) == wanted)
        .with_for_update()
    )


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def check_stock(product_name: str) -> dict[str, Any]:
    """Return the current stock and price for a single product."""
    if not isinstance(product_name, str) or not product_name.strip():
        return _validation_error("Nama produk wajib diisi.")
    try:
        with database.session_scope() as session:
            product = repository.find_product(session, product_name)
            if product is None:
                return _product_not_found(
                    product_name, repository.list_product_names(session)
                )
            stock = product.stock_quantity
            return {
                "success": True,
                "product_name": product.name,
                "stock": stock,
                "price": money(product.price),
                "in_stock": stock > 0,
                "low_stock": 0 < stock <= product.low_stock_threshold,
            }
    except (SQLAlchemyError, DatabaseNotConfiguredError, OSError) as exc:
        return _database_error(exc)


def create_order(customer_name: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a new order and deduct stock for every line item.

    The whole operation is one database transaction: the product rows
    are locked (FOR UPDATE), every line is validated against the locked
    stock, and only then are the order, its items, the stock deduction,
    and the customer aggregates written. A failure anywhere leaves no
    order, no items, and no stock change (the transaction rolls back).
    Unknown customers are registered automatically (walk-in) and
    flagged via ``new_customer``.
    """
    if not isinstance(customer_name, str) or not customer_name.strip():
        return _validation_error("Nama customer wajib diisi.")
    if not isinstance(items, list) or not items:
        return _validation_error("items harus list berisi minimal satu item.")

    try:
        with database.session_scope() as session:
            # Resolve + lock every referenced product up front (ordered
            # by id to avoid deadlocks), so validation sees committed
            # stock and concurrent orders cannot oversell.
            requested: list[tuple[str, int]] = []  # (raw name, quantity)
            for item in items:
                if (
                    not isinstance(item, dict)
                    or "product_name" not in item
                    or "quantity" not in item
                ):
                    return _validation_error(
                        "Setiap item harus dict dengan kunci 'product_name' dan 'quantity'."
                    )
                product_name = item["product_name"]
                quantity = item["quantity"]
                if not isinstance(product_name, str) or not product_name.strip():
                    return _validation_error("product_name pada item wajib diisi.")
                # bool is a subclass of int, so reject it explicitly.
                if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
                    return _validation_error(
                        f"quantity untuk '{product_name}' harus bilangan bulat positif."
                    )
                requested.append((product_name, quantity))

            # casefolded name -> locked product row
            by_fold: dict[str, Product] = {}
            for product_name, _ in requested:
                fold = product_name.strip().casefold()
                if fold in by_fold:
                    continue
                product = _lock_product(session, product_name)
                if product is None:
                    return _product_not_found(
                        product_name, repository.list_product_names(session)
                    )
                by_fold[fold] = product

            # Validate cumulative demand per product before any write.
            needed: dict[str, int] = {}
            for product_name, quantity in requested:
                product = by_fold[product_name.strip().casefold()]
                needed[product.name] = needed.get(product.name, 0) + quantity
                available = product.stock_quantity
                if available < needed[product.name]:
                    return {
                        "success": False,
                        "error": "INSUFFICIENT_STOCK",
                        "message": (
                            f"Stok '{product.name}' tidak cukup: diminta "
                            f"{needed[product.name]}, tersedia {available}."
                        ),
                        "product_name": product.name,
                        "requested": needed[product.name],
                        "available": available,
                    }

            customer, is_new = repository.register_customer(
                session, customer_name.strip()
            )
            order = repository.insert_order(
                session,
                order_id=repository.next_order_id(session),
                customer=customer,
                lines=[
                    (by_fold[pn.strip().casefold()], q) for pn, q in requested
                ],
            )
            session.flush()
            return {
                "success": True,
                "order": repository.order_as_dict(order),
                "new_customer": is_new,
            }
    except (SQLAlchemyError, DatabaseNotConfiguredError, OSError) as exc:
        return _database_error(exc)


def update_stock(product_name: str, quantity_change: int) -> dict[str, Any]:
    """Add to (positive) or subtract from (negative) a product's stock."""
    if not isinstance(product_name, str) or not product_name.strip():
        return _validation_error("Nama produk wajib diisi.")
    # bool is a subclass of int, so reject it explicitly.
    if isinstance(quantity_change, bool) or not isinstance(quantity_change, int):
        return _validation_error("quantity_change harus bilangan bulat.")
    if quantity_change == 0:
        return _validation_error("quantity_change tidak boleh 0.")
    try:
        with database.session_scope() as session:
            product = _lock_product(session, product_name)
            if product is None:
                return _product_not_found(
                    product_name, repository.list_product_names(session)
                )
            previous_stock = product.stock_quantity
            new_stock = previous_stock + quantity_change
            if new_stock < 0:
                return {
                    "success": False,
                    "error": "NEGATIVE_STOCK",
                    "message": (
                        f"Penurunan stok {-quantity_change} melebihi stok "
                        f"'{product.name}' yang tersisa {previous_stock}."
                    ),
                    "product_name": product.name,
                    "current_stock": previous_stock,
                    "requested_change": quantity_change,
                }
            product.stock_quantity = new_stock
            return {
                "success": True,
                "product_name": product.name,
                "previous_stock": previous_stock,
                "quantity_change": quantity_change,
                "new_stock": new_stock,
            }
    except (SQLAlchemyError, DatabaseNotConfiguredError, OSError) as exc:
        return _database_error(exc)


def search_customer(name: str) -> dict[str, Any]:
    """Search customers whose name contains ``name`` (case-insensitive)."""
    if not isinstance(name, str) or not name.strip():
        return _validation_error("Nama customer wajib diisi.")
    try:
        with database.session_scope() as session:
            matches = repository.search_customers(session, name.strip())
            return {
                "success": True,
                "query": name.strip(),
                "count": len(matches),
                "customers": [
                    repository.customer_as_dict(customer)
                    for customer in matches
                ],
            }
    except (SQLAlchemyError, DatabaseNotConfiguredError, OSError) as exc:
        return _database_error(exc)


def get_sales_report(period: str = "daily") -> dict[str, Any]:
    """Summarize orders inside a rolling window ending now.

    ``period`` must be one of ``VALID_PERIODS``: daily = last 24 hours,
    weekly = last 7 days, monthly = last 30 days. All figures are
    derived live from the orders/order_items tables — nothing is
    stored twice.
    """
    if not isinstance(period, str) or period.strip().casefold() not in VALID_PERIODS:
        return {
            "success": False,
            "error": "INVALID_PERIOD",
            "message": f"period harus salah satu dari: {', '.join(VALID_PERIODS)}.",
        }
    normalized = period.strip().casefold()
    window_days = {"daily": 1, "weekly": 7, "monthly": 30}[normalized]
    try:
        with database.session_scope() as session:
            summary = repository.sales_window(
                session, now=datetime.now(), window_days=window_days
            )
            return {
                "success": True,
                "period": normalized,
                "window_days": window_days,
                **summary,
            }
    except (SQLAlchemyError, DatabaseNotConfiguredError, OSError) as exc:
        return _database_error(exc)


def get_low_stock(threshold: int) -> dict[str, Any]:
    """List products whose stock is below ``threshold`` (lowest first)."""
    # bool is a subclass of int, so reject it explicitly.
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
        return _validation_error("threshold harus bilangan bulat >= 0.")
    try:
        with database.session_scope() as session:
            products = repository.low_stock_products(session, threshold)
            return {
                "success": True,
                "threshold": threshold,
                "count": len(products),
                "products": [
                    {
                        "product_name": product.name,
                        "stock": product.stock_quantity,
                        "price": money(product.price),
                    }
                    for product in products
                ],
            }
    except (SQLAlchemyError, DatabaseNotConfiguredError, OSError) as exc:
        return _database_error(exc)


def reset_mock_data() -> None:
    """Restore products, customers, and orders to their seeded state.

    Kept under its historical name for tests and the evaluation; now
    truncates and re-seeds the PostgreSQL store.
    """
    from app.db.seed import reset_database

    with database.session_scope() as session:
        reset_database(session, now=datetime.now())
