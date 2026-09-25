"""Tool registry for DibantuAI (Step 4A → Step 12: + knowledge base).

Maps the business tools (and, since Step 12, the RAG knowledge tool) to
the Anthropic tool-use format so the chat agent can advertise them via
``get_tool_schemas()`` and dispatch calls via ``get_tool(name)``.
``reset_mock_data`` is a test helper and is intentionally not
registered.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from app.rag.knowledge_tool import search_knowledge_base

from .business_tools import (
    VALID_PERIODS,
    check_stock,
    create_order,
    get_low_stock,
    get_sales_report,
    search_customer,
    update_stock,
)

__all__ = ["get_tool", "list_tools", "get_tool_schemas"]

#: name -> callable, in the order tools are advertised to the agent.
_TOOLS: dict[str, Callable[..., Any]] = {
    "check_stock": check_stock,
    "create_order": create_order,
    "update_stock": update_stock,
    "search_customer": search_customer,
    "get_sales_report": get_sales_report,
    "get_low_stock": get_low_stock,
    "search_knowledge_base": search_knowledge_base,
}

#: name -> Anthropic-compatible schema (``name`` / ``description`` /
#: ``input_schema``), kept in the same order as ``_TOOLS``.
_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "check_stock": {
        "name": "check_stock",
        "description": (
            "Return the current stock and price for a single product, "
            "including in-stock and low-stock flags."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "Product name to look up (case-insensitive).",
                },
            },
            "required": ["product_name"],
        },
    },
    "create_order": {
        "name": "create_order",
        "description": (
            "Create a new order and deduct stock for every line item. "
            "Unknown customers are registered automatically (walk-in)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Name of the customer placing the order.",
                },
                "items": {
                    "type": "array",
                    "description": "Order lines; at least one item is required.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product_name": {
                                "type": "string",
                                "description": (
                                    "Product to order (case-insensitive)."
                                ),
                            },
                            "quantity": {
                                "type": "integer",
                                "description": "Quantity to order (positive).",
                            },
                        },
                        "required": ["product_name", "quantity"],
                    },
                    "minItems": 1,
                },
            },
            "required": ["customer_name", "items"],
        },
    },
    "update_stock": {
        "name": "update_stock",
        "description": (
            "Add to (positive) or subtract from (negative) a product's stock."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "Product whose stock changes (case-insensitive).",
                },
                "quantity_change": {
                    "type": "integer",
                    "description": "Signed change applied to the stock; must not be 0.",
                },
            },
            "required": ["product_name", "quantity_change"],
        },
    },
    "search_customer": {
        "name": "search_customer",
        "description": (
            "Search customers whose name contains the given query "
            "(case-insensitive) and return their records."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Substring to search for in customer names.",
                },
            },
            "required": ["name"],
        },
    },
    "get_sales_report": {
        "name": "get_sales_report",
        "description": (
            "Summarize orders inside a rolling window ending now: "
            "daily = last 24 hours, weekly = last 7 days, monthly = last 30 days."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "enum": list(VALID_PERIODS),
                    "description": "Report window; defaults to 'daily'.",
                },
            },
            "required": [],
        },
    },
    "get_low_stock": {
        "name": "get_low_stock",
        "description": (
            "List products whose stock is below the given threshold, "
            "lowest stock first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "integer",
                    "description": "Stock level to compare against (>= 0).",
                    "minimum": 0,
                },
            },
            "required": ["threshold"],
        },
    },
    "search_knowledge_base": {
        "name": "search_knowledge_base",
        "description": (
            "Search the business knowledge base (documented policies, "
            "procedures, and operational rules) and return the most "
            "relevant passages with their source documents. Use this "
            "for policy or procedure questions such as refunds, "
            "cancellations, stock rules, payment rules, or operating "
            "rules. Do NOT use it for live stock, orders, customers, or "
            "sales data — those come from the database tools."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The policy/procedure question or keywords to "
                        "look up in the knowledge base."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        "Number of passages to return (1-10; default 3)."
                    ),
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}


def get_tool(name: str) -> Callable[..., Any] | None:
    """Return the tool callable registered under ``name``, or ``None``."""
    return _TOOLS.get(name)


def list_tools() -> list[str]:
    """Return the names of all registered tools in registration order."""
    return list(_TOOLS)


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return Anthropic-compatible schemas for every registered tool.

    Each entry has exactly the keys ``name``, ``description`` and
    ``input_schema`` (a JSON-Schema object). The list is a deep copy, so
    callers may mutate it freely.
    """
    return copy.deepcopy([_TOOL_SCHEMAS[name] for name in _TOOLS])
