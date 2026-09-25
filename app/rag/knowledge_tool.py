"""``search_knowledge_base`` — the RAG agent tool (Step 12).

Exposes the knowledge base to the GLM tool-calling loop exactly like
the business tools: a plain callable returning a JSON-serialisable
dict, never raising at the agent. Errors come back as
``{"success": False, "error": ..., "message": ...}`` so the model can
tell the user honestly that the knowledge base is unavailable instead
of the loop crashing.
"""

from __future__ import annotations

from typing import Any

from .errors import RagError
from .retriever import MAX_TOP_K, MIN_TOP_K
from .service import get_rag_service

__all__ = ["search_knowledge_base"]


def search_knowledge_base(
    query: str, top_k: int | None = None
) -> dict[str, Any]:
    """Search the business knowledge base and return cited passages.

    Use for policy/procedure questions (refunds, stock rules, order
    rules, payment rules) — never for live stock, orders, or customers,
    which live in the business database tools.
    """
    if not isinstance(query, str) or not query.strip():
        return {
            "success": False,
            "error": "VALIDATION_ERROR",
            "message": "Query is required.",
        }
    if top_k is not None and not isinstance(top_k, int):
        return {
            "success": False,
            "error": "VALIDATION_ERROR",
            "message": "top_k must be an integer.",
        }
    if isinstance(top_k, int):
        top_k = max(MIN_TOP_K, min(MAX_TOP_K, top_k))
    try:
        return get_rag_service().search(query, top_k=top_k)
    except ValueError as exc:
        return {
            "success": False,
            "error": "VALIDATION_ERROR",
            "message": str(exc),
        }
    except RagError as exc:
        # Missing deps, unavailable model, store problems — surfaced as
        # a plain message (no paths, keys, or tracebacks leak to the
        # model or the user).
        return {
            "success": False,
            "error": "KNOWLEDGE_BASE_UNAVAILABLE",
            "message": (
                "The knowledge base is not available right now. "
                f"Detail: {exc}"
            ),
        }
