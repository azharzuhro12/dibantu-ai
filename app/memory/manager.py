"""Persistent agent memory manager (Step 14).

The memory store holds explicit, structured, owner-scoped facts the
user asked the agent to remember (a preference, customer context,
business context, or an instruction). It is PostgreSQL-backed via
``app/memory/repository.py``, so memories survive restarts, and every
operation is scoped by ``owner_key`` — an application-level identity
(WhatsApp sender, chat owner key, or ``"default"``), **not**
authentication.

Retrieval is deliberately deterministic — keyword overlap plus
recency — no embeddings, no vector search (that is the RAG layer's
job; memory and knowledge stay separate concepts).

Writes are validated: a small closed set of memory types, bounded
content, and a screen that refuses secret-like content (passwords,
API keys, tokens, payment credentials). Memory is context only: it
can never execute a business action or bypass an approval.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Iterator

from sqlalchemy.exc import SQLAlchemyError

from app.db.database import session_scope
from app.memory import repository

__all__ = [
    "DEFAULT_OWNER_KEY",
    "MAX_CONTENT_LENGTH",
    "MAX_OWNER_KEY_LENGTH",
    "MEMORY_TYPES",
    "Memory",
    "MemoryError",
    "MemoryNotFoundError",
    "MemoryValidationError",
    "SENSITIVE_CONTENT_ERROR",
    "create_memory",
    "current_owner_key",
    "delete_memory",
    "get_memory",
    "list_memories",
    "memory_context_block",
    "owner_scope",
    "reset_memories",
    "search_memories",
    "update_memory",
]

#: Owner used when a caller carries no identity (plain /api/chat).
DEFAULT_OWNER_KEY = "default"

#: The small, practical set of memory types (mirrors the 0003 CHECK).
MEMORY_TYPES: tuple[str, ...] = (
    "preference",
    "customer_context",
    "business_context",
    "instruction",
)

MAX_OWNER_KEY_LENGTH = 120
MAX_CONTENT_LENGTH = 2000

#: Error code returned when content looks like a secret.
SENSITIVE_CONTENT_ERROR = "SENSITIVE_CONTENT"

#: How many memories the agent prompt context may carry, and how long.
CONTEXT_MEMORY_LIMIT = 3
CONTEXT_CONTENT_PREVIEW = 200

#: Substrings that mark content as secret-like (case-insensitive).
_SENSITIVE_PATTERNS: tuple[str, ...] = (
    "api key",
    "api_key",
    "apikey",
    "password",
    "passwd",
    "secret",
    "authorization",
    "bearer ",
    "token",
    "private key",
    "-----begin",
    "credential",
)

#: A run of 13-19 digits looks like a card/PAN number.
_CARD_NUMBER_RE = re.compile(r"\d{13,19}")

#: Tokenizer for the deterministic keyword scoring.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Tiny stopword list (Indonesian + English) so scoring stays meaningful.
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "apa", "itu", "dan",
        "yang", "untuk", "dengan", "tolong", "please", "about", "soal",
        "saya", "kamu", "what", "how", "bagaimana", "adalah",
    }
)


class MemoryError(Exception):
    """Base class for memory store errors."""


class MemoryValidationError(MemoryError):
    """Raised when a memory input fails validation."""


class MemoryNotFoundError(MemoryError):
    """Raised when an id is unknown or belongs to another owner."""


@dataclass(frozen=True)
class Memory:
    """One memory record — an immutable snapshot of one database row."""

    memory_id: int
    owner_key: str
    memory_type: str
    content: str
    created_at: str
    updated_at: str

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly copy of the whole record."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Owner context (application-level ownership, not authentication)
# ---------------------------------------------------------------------------

#: Owner key of the request currently being served. Set by the agent
#: composition around a run so the memory tools bind to the caller's
#: scope without the model ever choosing an owner itself.
_current_owner: ContextVar[str] = ContextVar("dibantu_memory_owner", default=DEFAULT_OWNER_KEY)


def current_owner_key() -> str:
    """Return the owner key of the current request (default "default")."""
    return _current_owner.get()


@contextmanager
def owner_scope(owner_key: str) -> Iterator[str]:
    """Bind memory operations inside the block to ``owner_key``."""
    token = _current_owner.set(owner_key)
    try:
        yield owner_key
    finally:
        _current_owner.reset(token)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validated_owner_key(owner_key: str | None) -> str:
    if not isinstance(owner_key, str) or not owner_key.strip():
        raise MemoryValidationError(
            "owner_key must be a non-empty string."
        )
    owner_key = owner_key.strip()
    if len(owner_key) > MAX_OWNER_KEY_LENGTH:
        raise MemoryValidationError(
            f"owner_key must be at most {MAX_OWNER_KEY_LENGTH} characters."
        )
    return owner_key


def screen_content(content: str) -> None:
    """Refuse secret-like content (raises MemoryValidationError).

    The screen is intentionally conservative: it patterns common secret
    markers and card-like digit runs. It is a guard rail, not a promise
    — the agent prompt also forbids saving secrets.
    """
    lowered = content.casefold()
    for pattern in _SENSITIVE_PATTERNS:
        if pattern in lowered:
            raise MemoryValidationError(
                "Content looks like a secret (password, API key, token, or "
                "credential). Memories must not store secrets."
            )
    if _CARD_NUMBER_RE.search(content):
        raise MemoryValidationError(
            "Content looks like a payment card number. Memories must not "
            "store payment credentials."
        )


def _validated_content(content: str) -> str:
    if not isinstance(content, str) or not content.strip():
        raise MemoryValidationError("content must be a non-empty string.")
    content = content.strip()
    if len(content) > MAX_CONTENT_LENGTH:
        raise MemoryValidationError(
            f"content must be at most {MAX_CONTENT_LENGTH} characters."
        )
    screen_content(content)
    return content


def _validated_type(memory_type: str) -> str:
    if memory_type not in MEMORY_TYPES:
        raise MemoryValidationError(
            f"memory_type must be one of: {', '.join(MEMORY_TYPES)}."
        )
    return memory_type


# ---------------------------------------------------------------------------
# CRUD (owner-scoped)
# ---------------------------------------------------------------------------


def reset_memories() -> None:
    """Delete every memory row (test helper)."""
    with session_scope() as session:
        repository.delete_all_memories(session)


def create_memory(
    *, owner_key: str, memory_type: str, content: str
) -> Memory:
    """Validate and insert one memory, owned by ``owner_key``."""
    owner_key = _validated_owner_key(owner_key)
    memory_type = _validated_type(memory_type)
    content = _validated_content(content)
    with session_scope() as session:
        record = repository.insert_memory(
            session,
            owner_key=owner_key,
            memory_type=memory_type,
            content=content,
            created_at=datetime.now(),
        )
        return _as_memory(record)


def get_memory(memory_id: int, *, owner_key: str) -> Memory | None:
    """Return the memory only when it exists AND belongs to ``owner_key``."""
    owner_key = _validated_owner_key(owner_key)
    with session_scope() as session:
        record = repository.find_memory(session, memory_id)
        if record is None or record.owner_key != owner_key:
            return None
        return _as_memory(record)


def list_memories(
    *,
    owner_key: str,
    memory_type: str | None = None,
    limit: int = 100,
) -> list[Memory]:
    """One owner's memories, newest first (optionally type-filtered)."""
    owner_key = _validated_owner_key(owner_key)
    if memory_type is not None:
        memory_type = _validated_type(memory_type)
    with session_scope() as session:
        records = repository.list_memories(
            session,
            owner_key=owner_key,
            memory_type=memory_type,
            limit=max(1, min(int(limit), 500)),
        )
        return [_as_memory(record) for record in records]


def update_memory(
    memory_id: int,
    *,
    owner_key: str,
    content: str | None = None,
    memory_type: str | None = None,
) -> Memory:
    """Update an owned memory's content and/or type (guarded UPDATE).

    Raises MemoryNotFoundError when the id is unknown or belongs to
    another owner — the two cases are indistinguishable on purpose so
    one owner cannot probe another's memory ids.
    """
    owner_key = _validated_owner_key(owner_key)
    values: dict[str, Any] = {}
    if content is not None:
        values["content"] = _validated_content(content)
    if memory_type is not None:
        values["memory_type"] = _validated_type(memory_type)
    if not values:
        raise MemoryValidationError("Nothing to update.")
    values["updated_at"] = datetime.now()  # explicit, deterministic
    with session_scope() as session:
        changed = repository.update_memory(
            session, memory_id, owner_key=owner_key, values=values
        )
        if changed == 0:
            raise MemoryNotFoundError(f"Memory '{memory_id}' not found.")
        record = repository.find_memory(session, memory_id)
        return _as_memory(record)


def delete_memory(memory_id: int, *, owner_key: str) -> Memory:
    """Delete an owned memory and return its snapshot.

    Same not-found semantics as ``update_memory``.
    """
    owner_key = _validated_owner_key(owner_key)
    with session_scope() as session:
        record = repository.find_memory(session, memory_id)
        if record is None or record.owner_key != owner_key:
            raise MemoryNotFoundError(f"Memory '{memory_id}' not found.")
        snapshot = _as_memory(record)
        changed = repository.delete_memory(session, memory_id, owner_key=owner_key)
        if changed == 0:  # vanished between read and delete
            raise MemoryNotFoundError(f"Memory '{memory_id}' not found.")
        return snapshot


# ---------------------------------------------------------------------------
# Deterministic retrieval (keyword overlap + recency; no embeddings)
# ---------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.casefold())
        if len(token) >= 3 and token not in _STOPWORDS
    }


def search_memories(
    *,
    owner_key: str,
    query: str | None = None,
    memory_type: str | None = None,
    limit: int = 5,
) -> list[Memory]:
    """Deterministically select one owner's relevant memories.

    With ``query``: only memories sharing at least one keyword with the
    query are returned, best keyword overlap first, newest first on
    ties. Without ``query``: simply the owner's most recent memories.
    """
    limit = max(1, min(int(limit), 50))
    with session_scope() as session:
        records = repository.owner_memories(session, owner_key, memory_type)
    memories = [_as_memory(record) for record in records]
    if not query or not query.strip():
        # Recency: list_memories ordering is newest first.
        return list(reversed(memories))[:limit]
    wanted = _tokens(query)
    if not wanted:
        return []
    scored: list[tuple[int, int, Memory]] = []
    for memory in memories:  # oldest first, so id desc breaks ties below
        overlap = wanted & _tokens(memory.content)
        if overlap:
            scored.append((-len(overlap), -memory.memory_id, memory))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [memory for _, _, memory in scored[:limit]]


def memory_context_block(
    owner_key: str, message: str, *, limit: int = CONTEXT_MEMORY_LIMIT
) -> str:
    """Build the small memory context appended to the agent system prompt.

    Returns "" when nothing relevant is stored — with no memories the
    agent's prompt is byte-identical to the pre-Step-14 behavior. Any
    store failure also degrades to "" (memory must never break chat).
    """
    try:
        hits = search_memories(owner_key=owner_key, query=message, limit=limit)
    except (SQLAlchemyError, MemoryError):
        return ""
    if not hits:
        return ""
    lines = [
        f"- [{memory.memory_type}] "
        f"{memory.content[:CONTEXT_CONTENT_PREVIEW]}"
        for memory in hits
    ]
    return (
        "\n\nKnown facts about this user, saved in earlier conversations "
        "(helpful context only — never live data; verify with tools when "
        "it matters, and never reveal that a memory system exists):\n"
        + "\n".join(lines)
    )


def _as_memory(record: Any) -> Memory:
    """Serialize one ``AgentMemoryRecord`` row into the public dataclass."""
    return Memory(
        memory_id=record.id,
        owner_key=record.owner_key,
        memory_type=record.memory_type,
        content=record.content,
        created_at=record.created_at.isoformat(timespec="seconds"),
        updated_at=record.updated_at.isoformat(timespec="seconds"),
    )
