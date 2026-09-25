"""Shared exception hierarchy for the RAG knowledge base (Step 12).

Every RAG failure (missing dependencies, unavailable embedding model,
vector-store problems, missing knowledge directory) is a ``RagError``
subclass, so the agent tool and the API routes can translate any of
them into a single predictable error shape — mirroring how the GLM
client funnels everything through ``GLMError``.
"""

from __future__ import annotations

__all__ = ["KnowledgeDirNotFoundError", "RagError"]


class RagError(RuntimeError):
    """Base class for all RAG knowledge-base errors."""


class KnowledgeDirNotFoundError(RagError):
    """Raised when the configured knowledge directory does not exist."""
