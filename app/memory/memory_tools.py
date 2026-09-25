"""Memory agent tools: save_memory, search_memory, delete_memory (Step 14).

Exposed to the GLM tool-calling loop exactly like the business and
knowledge tools: plain callables returning JSON-serialisable dicts,
never raising at the agent. All three operate on the owner scope of
the CURRENT request (``current_owner_key``) — the model cannot choose
an owner, so it can never read or write another owner's memories.

Memory is context only: these tools touch no business data, no RAG
documents, no approvals, and no secrets (secret-like content is
refused by the manager's screen).
"""

from __future__ import annotations

from typing import Any

from app.memory import manager
from app.memory.manager import SENSITIVE_CONTENT_ERROR

__all__ = ["delete_memory", "save_memory", "search_memory"]


def save_memory(content: str, memory_type: str = "preference") -> dict[str, Any]:
    """Remember one explicit fact for the current user.

    Use ONLY when the user explicitly asks to remember something. Never
    save secrets (passwords, API keys, tokens, payment credentials) or
    unsolicited personal details.
    """
    if not isinstance(content, str) or not isinstance(memory_type, str):
        return {
            "success": False,
            "error": "VALIDATION_ERROR",
            "message": "content and memory_type must be strings.",
        }
    try:
        memory = manager.create_memory(
            owner_key=manager.current_owner_key(),
            memory_type=memory_type,
            content=content,
        )
    except manager.MemoryValidationError as exc:
        error = (
            SENSITIVE_CONTENT_ERROR
            if "secret" in str(exc) or "card number" in str(exc)
            else "VALIDATION_ERROR"
        )
        return {"success": False, "error": error, "message": str(exc)}
    except Exception as exc:  # noqa: BLE001 - store failure must not crash the loop
        return {
            "success": False,
            "error": "MEMORY_STORE_UNAVAILABLE",
            "message": f"The memory store is not available right now. ({type(exc).__name__})",
        }
    return {
        "success": True,
        "memory_id": memory.memory_id,
        "memory_type": memory.memory_type,
        "content": memory.content,
        "message": f"Saved as memory {memory.memory_id}.",
    }


def search_memory(query: str, memory_type: str | None = None) -> dict[str, Any]:
    """Recall the current user's memories relevant to ``query``.

    Deterministic keyword matching — no semantic search. Answer recall
    questions only from what this returns; if nothing relevant is
    found, say the user has no such memory.
    """
    if not isinstance(query, str) or not query.strip():
        return {
            "success": False,
            "error": "VALIDATION_ERROR",
            "message": "query is required.",
        }
    if memory_type is not None and memory_type not in manager.MEMORY_TYPES:
        return {
            "success": False,
            "error": "VALIDATION_ERROR",
            "message": (
                "memory_type must be one of: "
                f"{', '.join(manager.MEMORY_TYPES)}."
            ),
        }
    try:
        memories = manager.search_memories(
            owner_key=manager.current_owner_key(),
            query=query,
            memory_type=memory_type,
            limit=5,
        )
    except Exception as exc:  # noqa: BLE001 - store failure must not crash the loop
        return {
            "success": False,
            "error": "MEMORY_STORE_UNAVAILABLE",
            "message": f"The memory store is not available right now. ({type(exc).__name__})",
        }
    return {
        "success": True,
        "query": query,
        "results": [
            {
                "memory_id": memory.memory_id,
                "memory_type": memory.memory_type,
                "content": memory.content,
                "created_at": memory.created_at,
            }
            for memory in memories
        ],
        "count": len(memories),
        "message": (
            "No relevant memories found."
            if not memories
            else f"{len(memories)} relevant memories found."
        ),
    }


def delete_memory(memory_id: int) -> dict[str, Any]:
    """Forget one of the current user's memories by id."""
    if not isinstance(memory_id, int):
        return {
            "success": False,
            "error": "VALIDATION_ERROR",
            "message": "memory_id must be an integer.",
        }
    try:
        deleted = manager.delete_memory(
            memory_id, owner_key=manager.current_owner_key()
        )
    except manager.MemoryNotFoundError:
        return {
            "success": False,
            "error": "NOT_FOUND",
            "message": (
                f"Memory {memory_id} was not found for this user."
            ),
        }
    except Exception as exc:  # noqa: BLE001 - store failure must not crash the loop
        return {
            "success": False,
            "error": "MEMORY_STORE_UNAVAILABLE",
            "message": f"The memory store is not available right now. ({type(exc).__name__})",
        }
    return {
        "success": True,
        "deleted_memory_id": deleted.memory_id,
        "message": f"Memory {deleted.memory_id} deleted.",
    }
