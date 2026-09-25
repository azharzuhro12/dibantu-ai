"""Persistent agent memory package for DibantuAI (Step 14).

Owner-scoped structured facts the user explicitly asked the agent to
remember, stored in PostgreSQL (``agent_memories``) so they survive
restarts. Retrieval is deterministic (keyword + recency). The memory
tools (save/search/delete) bind to the current request's owner scope;
memory is context only and can never execute business actions.
"""

from .manager import (
    DEFAULT_OWNER_KEY,
    MEMORY_TYPES,
    Memory,
    MemoryError,
    MemoryNotFoundError,
    MemoryValidationError,
    create_memory,
    current_owner_key,
    delete_memory,
    get_memory,
    list_memories,
    memory_context_block,
    owner_scope,
    reset_memories,
    search_memories,
    update_memory,
)
from .memory_tools import delete_memory as delete_memory_tool
from .memory_tools import save_memory as save_memory_tool
from .memory_tools import search_memory as search_memory_tool
from .repository import delete_all_memories

__all__ = [
    "DEFAULT_OWNER_KEY",
    "MEMORY_TYPES",
    "Memory",
    "MemoryError",
    "MemoryNotFoundError",
    "MemoryValidationError",
    "create_memory",
    "current_owner_key",
    "delete_all_memories",
    "delete_memory",
    "delete_memory_tool",
    "get_memory",
    "list_memories",
    "memory_context_block",
    "owner_scope",
    "reset_memories",
    "save_memory_tool",
    "search_memories",
    "search_memory_tool",
    "update_memory",
]
