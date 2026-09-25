"""Tool registry for DibantuAI (Step 4A → Step 16: + knowledge, memory,
sensitive-request schemas).

Maps the business tools (and, since Step 12, the RAG knowledge tool;
since Step 14, the memory tools) to the Anthropic tool-use format so
the chat agent can advertise them via ``get_tool_schemas()`` and
dispatch calls via ``get_tool(name)``. ``reset_mock_data`` is a test
helper and is intentionally not registered.

Since Step 16 the advertised schemas ALSO include the three sensitive
actions (``refund_order``, ``cancel_order``, ``bulk_stock_update``) as
request schemas: the model can ask for them, which creates a pending
approval — but they are deliberately absent from the dispatch table
(``get_tool`` returns None), so the agent can never execute them
directly. Execution happens only through the approval executor's own
allowlist after a human approves.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from app.memory import memory_tools
from app.memory.manager import MEMORY_TYPES
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

__all__ = ["get_tool", "list_tools", "get_tool_schemas", "SENSITIVE_REQUEST_TOOLS"]

#: name -> callable, in the order tools are advertised to the agent.
#: The three sensitive actions are NOT here on purpose (see module
#: docstring): requesting them never dispatches anything.
_TOOLS: dict[str, Callable[..., Any]] = {
    "check_stock": check_stock,
    "create_order": create_order,
    "update_stock": update_stock,
    "search_customer": search_customer,
    "get_sales_report": get_sales_report,
    "get_low_stock": get_low_stock,
    "search_knowledge_base": search_knowledge_base,
    "save_memory": memory_tools.save_memory,
    "search_memory": memory_tools.search_memory,
    "delete_memory": memory_tools.delete_memory,
}

#: Sensitive actions advertised to the model as request schemas only
#: (Step 16). Asking for one creates a pending human approval and
#: returns ``{"approval_required": true, ...}`` — nothing executes.
SENSITIVE_REQUEST_TOOLS: tuple[str, ...] = (
    "refund_order",
    "cancel_order",
    "bulk_stock_update",
)

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
    "save_memory": {
        "name": "save_memory",
        "description": (
            "Save one explicit fact to remember for the current user. "
            "Use ONLY when the user clearly asks to remember something "
            "(a preference, customer detail, business rule, or standing "
            "instruction). Never save secrets (passwords, API keys, "
            "tokens, payment credentials) or unsolicited personal data. "
            "Memory is context only — saving never performs a business "
            "action."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": (
                        "The fact to remember, as one concise sentence."
                    ),
                },
                "memory_type": {
                    "type": "string",
                    "enum": list(MEMORY_TYPES),
                    "description": (
                        "Kind of fact (default 'preference')."
                    ),
                },
            },
            "required": ["content"],
        },
    },
    "search_memory": {
        "name": "search_memory",
        "description": (
            "Recall what the current user asked to remember, relevant "
            "to a query (deterministic keyword match). Use for 'what "
            "did I ask you to remember' style questions and answer only "
            "from what it returns. Memories are context, never live "
            "business data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords of what to recall.",
                },
                "memory_type": {
                    "type": "string",
                    "enum": list(MEMORY_TYPES),
                    "description": "Optionally restrict to one memory type.",
                },
            },
            "required": ["query"],
        },
    },
    "delete_memory": {
        "name": "delete_memory",
        "description": (
            "Forget one of the current user's memories by id (use when "
            "the user asks to stop remembering something)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "integer",
                    "description": "Id of the memory to delete.",
                },
            },
            "required": ["memory_id"],
        },
    },
}

#: Request schemas for the sensitive actions (advertised, never
#: dispatched — see ``SENSITIVE_REQUEST_TOOLS``). The descriptions tell
#: the model exactly what calling them does: a human approval is
#: created, nothing executes yet.
_SENSITIVE_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "refund_order": {
        "name": "refund_order",
        "description": (
            "Request a refund of one order. SENSITIVE: requires human "
            "approval — calling this only creates a pending approval and "
            "returns approval_required; the refund runs later, after a "
            "human approves and executes it. Omit 'amount' to refund the "
            "full remaining balance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Order to refund (e.g. 'ORD-0001').",
                },
                "amount": {
                    "type": "number",
                    "description": (
                        "Optional refund amount (IDR); omit for the full "
                        "remaining balance. Must not exceed the order total."
                    ),
                },
            },
            "required": ["order_id"],
        },
    },
    "cancel_order": {
        "name": "cancel_order",
        "description": (
            "Request the cancellation of one order (marks it cancelled and "
            "restores its stock once executed). SENSITIVE: requires human "
            "approval — calling this only creates a pending approval and "
            "returns approval_required."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Order to cancel (e.g. 'ORD-0002').",
                },
            },
            "required": ["order_id"],
        },
    },
    "bulk_stock_update": {
        "name": "bulk_stock_update",
        "description": (
            "Request several stock changes at once (signed deltas). "
            "SENSITIVE: requires human approval — calling this only creates "
            "a pending approval and returns approval_required. Use for "
            "BATCH changes of two or more products; a single-product "
            "change should use update_stock instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "updates": {
                    "type": "array",
                    "description": "Stock changes to apply together.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product_name": {
                                "type": "string",
                                "description": (
                                    "Product to change (case-insensitive)."
                                ),
                            },
                            "quantity_change": {
                                "type": "integer",
                                "description": "Signed change; must not be 0.",
                            },
                        },
                        "required": ["product_name", "quantity_change"],
                    },
                    "minItems": 1,
                },
            },
            "required": ["updates"],
        },
    },
}


def get_tool(name: str) -> Callable[..., Any] | None:
    """Return the tool callable registered under ``name``, or ``None``.

    Sensitive actions deliberately resolve to ``None``: they are not
    dispatchable. (The agent intercepts them earlier anyway; this is
    defense in depth.)
    """
    return _TOOLS.get(name)


def list_tools() -> list[str]:
    """Return the names of all DISPATCHABLE tools, in registration order.

    The sensitive actions are intentionally absent — see
    ``SENSITIVE_REQUEST_TOOLS``.
    """
    return list(_TOOLS)


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return Anthropic-compatible schemas for every advertised tool.

    That is the dispatchable tools plus the three sensitive-request
    schemas: the model may ask for a sensitive action (creating a human
    approval), but only dispatchable tools can execute directly. Each
    entry has exactly the keys ``name``, ``description`` and
    ``input_schema`` (a JSON-Schema object). The list is a deep copy,
    so callers may mutate it freely.
    """
    advertised = [
        _TOOL_SCHEMAS[name] for name in _TOOLS
    ] + [_SENSITIVE_TOOL_SCHEMAS[name] for name in SENSITIVE_REQUEST_TOOLS]
    return copy.deepcopy(advertised)
