"""Sensitive business actions for DibantuAI (Step 16).

Real, transactional implementations of the three actions that require
human approval (``refund_order``, ``cancel_order``, ``bulk_stock_update``).
The agent can NEVER call these directly: the agent loop intercepts
sensitive action requests and turns them into pending approvals
(see ``app/agent/agent.py``), and only the approval executor — after a
human approves — dispatches the stored payload here
(see ``app/approval/executor.py``). That is also why these functions
are deliberately NOT registered in the agent tool registry.

Every function follows the ``business_tools`` contract: validation and
business failures are ``{"success": False, "error": ..., "message": ...}``
dicts (never exceptions), and success payloads carry the resulting
state. Each mutation is one database transaction: the order (or every
referenced product) is locked ``FOR UPDATE`` first, validated, then
written together with the stock restore and the customer aggregates.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.db import database, repository
from app.db.database import DatabaseNotConfiguredError
from app.db.models import Order, Product, money

__all__ = [
    "MAX_BULK_UPDATES",
    "bulk_stock_update",
    "cancel_order",
    "refund_order",
]

#: Upper bound for one bulk_stock_update request (defensive cap).
MAX_BULK_UPDATES = 50

#: Statuses that block a refund/cancel (only "completed" orders may move).
_TERMINAL_ORDER_STATUSES = ("refunded", "cancelled")


def _validation_error(message: str) -> dict[str, Any]:
    return {"success": False, "error": "VALIDATION_ERROR", "message": message}


def _order_not_found(order_id: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": "ORDER_NOT_FOUND",
        "message": f"Pesanan '{order_id}' tidak ditemukan.",
    }


def _database_error(exc: Exception) -> dict[str, Any]:
    """Translate a database failure into an agent-visible error dict.

    The exception text is deliberately NOT included: connection errors
    can embed the full DATABASE_URL (with credentials).
    """
    if isinstance(exc, DatabaseNotConfiguredError):
        message = "DATABASE_URL is not configured."
    else:
        message = "Business database is unreachable. Try again once it is up."
    return {"success": False, "error": "DATABASE_ERROR", "message": message}


def _lock_order(session: Any, order_id: str) -> Order | None:
    """Load one order, locking its row FOR UPDATE (items load lazily)."""
    return session.scalar(
        select(Order).where(Order.id == order_id).with_for_update()
    )


def _lock_order_products(session: Any, order: Order) -> dict[int, Product]:
    """Lock every product the order references (ordered by id); id -> row."""
    product_ids = sorted({item.product_id for item in order.items})
    return {
        product.id: product
        for product in repository.lock_products_by_ids(session, product_ids)
    }


def _undo_order_effects(
    order: Order, products: dict[int, Product], amount: Decimal
) -> None:
    """Restore the ordered stock and adjust customer aggregates.

    The caller holds the locks; this runs inside the caller's
    transaction. ``total_spent`` never goes below zero — the customer's
    aggregates may predate the seeded orders and not include this
    order's total.
    """
    for item in order.items:
        product = products.get(item.product_id)
        if product is not None:
            product.stock_quantity += item.quantity
    customer = order.customer
    if customer is not None:
        if customer.total_orders > 0:
            customer.total_orders -= 1
        customer.total_spent = max(Decimal(0), customer.total_spent - amount)


def _terminal_status_error(order_id: str, status: str, verb: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": f"ORDER_ALREADY_{status.upper()}",
        "message": (
            f"Pesanan '{order_id}' sudah berstatus '{status}' dan "
            f"tidak bisa di{verb}."
        ),
    }


def refund_order(order_id: str, amount: int | float | None = None) -> dict[str, Any]:
    """Refund one order, fully or partially, in one transaction.

    ``amount`` is optional — omitting it refunds whatever remains
    (``total_price - refunded_amount``). Two regimes:

    * FULL refund (``amount`` omitted, or exactly the remaining
      balance): the ordered stock is restored, the customer aggregates
      are undone, and the order is marked ``refunded`` — it leaves the
      sales reports. Terminal: no further refund is possible.
    * PARTIAL refund (``0 < amount < remaining``): only money moves —
      ``total_spent`` is reduced and ``refunded_amount`` accumulates.
      The order stays ``completed`` (the sale stands, goods are
      considered kept) and further partial refunds of the remainder
      remain possible, each still gated by its own approval. The
      cumulative ``refunded_amount`` can never exceed the total.

    Only ``completed`` orders can be refunded, so refunding a
    refunded/cancelled order fails cleanly — defense in depth on top of
    the approval state machine, which already prevents double
    execution of one approval.
    """
    if not isinstance(order_id, str) or not order_id.strip():
        return _validation_error("order_id wajib diisi.")
    order_id = order_id.strip()
    if amount is not None:
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            return _validation_error("amount harus berupa angka.")
        if amount <= 0:
            return _validation_error("amount harus lebih besar dari 0.")
    try:
        with database.session_scope() as session:
            order = _lock_order(session, order_id)
            if order is None:
                return _order_not_found(order_id)
            if order.status in _TERMINAL_ORDER_STATUSES:
                return _terminal_status_error(order_id, order.status, "refund")
            remaining = order.total_price - order.refunded_amount
            if remaining <= 0:  # pragma: no cover - status guard covers it
                return _terminal_status_error(order_id, "refunded", "refund")
            refund_amount = (
                Decimal(str(amount)) if amount is not None else remaining
            )
            if refund_amount > remaining:
                return {
                    "success": False,
                    "error": "REFUND_EXCEEDS_TOTAL",
                    "message": (
                        f"Amount refund {money(refund_amount)} melebihi sisa "
                        f"yang bisa direfund {money(remaining)} "
                        f"(total {money(order.total_price)}, sudah direfund "
                        f"{money(order.refunded_amount)})."
                    ),
                }
            previous_status = order.status
            previous_refunded = order.refunded_amount
            full_refund = refund_amount == remaining
            restocked: list[dict[str, Any]] = []
            if full_refund:
                products = _lock_order_products(session, order)
                _undo_order_effects(order, products, remaining)
                order.refunded_amount = order.total_price
                order.status = "refunded"
                restocked = [
                    {"product_name": item.product_name, "quantity": item.quantity}
                    for item in order.items
                ]
            else:
                customer = order.customer
                if customer is not None:
                    customer.total_spent = max(
                        Decimal(0), customer.total_spent - refund_amount
                    )
                order.refunded_amount = previous_refunded + refund_amount
            session.flush()
            return {
                "success": True,
                "action": "refund_order",
                "order_id": order.id,
                "previous_status": previous_status,
                "status": order.status,
                "refund_amount": money(refund_amount),
                "refunded_total": money(order.refunded_amount),
                "order_total": money(order.total_price),
                "partial": not full_refund,
                "restocked_items": restocked,
            }
    except (SQLAlchemyError, DatabaseNotConfiguredError, OSError) as exc:
        return _database_error(exc)


def cancel_order(order_id: str) -> dict[str, Any]:
    """Cancel one order: mark it cancelled and undo its business effects.

    Same transactional shape as ``refund_order`` (locked order + locked
    products, stock restored, aggregates adjusted), but the full order
    total is always undone — partial cancellation would need order line
    edits and is out of scope.
    """
    if not isinstance(order_id, str) or not order_id.strip():
        return _validation_error("order_id wajib diisi.")
    order_id = order_id.strip()
    try:
        with database.session_scope() as session:
            order = _lock_order(session, order_id)
            if order is None:
                return _order_not_found(order_id)
            if order.status in _TERMINAL_ORDER_STATUSES:
                return _terminal_status_error(order_id, order.status, "batalkan")
            previous_status = order.status
            products = _lock_order_products(session, order)
            _undo_order_effects(order, products, order.total_price)
            order.status = "cancelled"
            session.flush()
            return {
                "success": True,
                "action": "cancel_order",
                "order_id": order.id,
                "previous_status": previous_status,
                "status": order.status,
                "order_total": money(order.total_price),
                "restocked_items": [
                    {"product_name": item.product_name, "quantity": item.quantity}
                    for item in order.items
                ],
            }
    except (SQLAlchemyError, DatabaseNotConfiguredError, OSError) as exc:
        return _database_error(exc)


def bulk_stock_update(updates: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply several signed stock changes in one transaction.

    Every referenced product is resolved and locked up front (ordered by
    id to avoid deadlocks), the cumulative change per product is
    validated against the locked stock, and only then are all changes
    applied — an invalid line rolls the whole request back.
    """
    if not isinstance(updates, list) or not updates:
        return _validation_error("updates harus list berisi minimal satu item.")
    if len(updates) > MAX_BULK_UPDATES:
        return _validation_error(
            f"updates maksimal {MAX_BULK_UPDATES} item per request."
        )
    parsed: list[tuple[str, int]] = []
    for update in updates:
        if (
            not isinstance(update, dict)
            or "product_name" not in update
            or "quantity_change" not in update
        ):
            return _validation_error(
                "Setiap update harus dict dengan kunci 'product_name' dan 'quantity_change'."
            )
        product_name = update["product_name"]
        quantity_change = update["quantity_change"]
        if not isinstance(product_name, str) or not product_name.strip():
            return _validation_error("product_name pada update wajib diisi.")
        if (
            isinstance(quantity_change, bool)
            or not isinstance(quantity_change, int)
            or quantity_change == 0
        ):
            return _validation_error(
                f"quantity_change untuk '{product_name}' harus bilangan bulat bukan 0."
            )
        parsed.append((product_name, quantity_change))
    try:
        with database.session_scope() as session:
            by_fold: dict[str, Product] = {}
            for product_name, _ in parsed:
                fold = product_name.strip().casefold()
                if fold in by_fold:
                    continue
                product = session.scalar(
                    select(Product)
                    .where(func.lower(Product.name) == fold)
                    .with_for_update()
                )
                if product is None:
                    return {
                        "success": False,
                        "error": "PRODUCT_NOT_FOUND",
                        "message": f"Produk '{product_name.strip()}' tidak ditemukan.",
                        "available_products": repository.list_product_names(session),
                    }
                by_fold[fold] = product

            # Validate the cumulative change per product before any write.
            deltas: dict[str, int] = {}
            for product_name, quantity_change in parsed:
                product = by_fold[product_name.strip().casefold()]
                delta = deltas.get(product.name, 0) + quantity_change
                if product.stock_quantity + delta < 0:
                    return {
                        "success": False,
                        "error": "INSUFFICIENT_STOCK",
                        "message": (
                            f"Stok '{product.name}' tidak cukup: perubahan "
                            f"kumulatif {delta} dari {product.stock_quantity} "
                            "akan menghasilkan stok negatif."
                        ),
                        "product_name": product.name,
                    }
                deltas[product.name] = delta

            applied: list[dict[str, Any]] = []
            for name, delta in deltas.items():
                product = by_fold[name.strip().casefold()]
                previous_stock = product.stock_quantity
                product.stock_quantity = previous_stock + delta
                applied.append(
                    {
                        "product_name": name,
                        "previous_stock": previous_stock,
                        "quantity_change": delta,
                        "new_stock": product.stock_quantity,
                    }
                )
            session.flush()
            return {
                "success": True,
                "action": "bulk_stock_update",
                "updated_count": len(applied),
                "updates": applied,
            }
    except (SQLAlchemyError, DatabaseNotConfiguredError, OSError) as exc:
        return _database_error(exc)
