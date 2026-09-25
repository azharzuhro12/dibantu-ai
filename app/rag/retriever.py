"""Semantic retrieval over the knowledge vector store (Step 12).

Pipeline: question -> embedding -> ChromaDB cosine similarity search ->
top-k chunks as structured ``RetrievalResult`` rows (content, source,
section, chunk id, similarity). Purely mechanical — answer generation
stays with GLM in the agent loop; this module never calls any LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

from .embeddings import EmbeddingProvider
from .vector_store import ChromaVectorStore

__all__ = ["MIN_TOP_K", "MAX_TOP_K", "RetrievalResult", "KnowledgeRetriever"]

#: Bounds for the number of passages returned per query.
MIN_TOP_K = 1
MAX_TOP_K = 10


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved knowledge passage with its citation metadata."""

    content: str
    source: str
    section: str
    chunk_id: str
    similarity: float


class KnowledgeRetriever:
    """Embeds queries and searches a ``ChromaVectorStore``."""

    def __init__(
        self, store: ChromaVectorStore, embeddings: EmbeddingProvider
    ) -> None:
        self._store = store
        self._embeddings = embeddings

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievalResult]:
        """Return the ``top_k`` passages nearest to ``query``.

        ``top_k`` is clamped into ``[MIN_TOP_K, MAX_TOP_K]`` so a bad
        value from the model can never blow up the store query.
        """
        k = max(MIN_TOP_K, min(MAX_TOP_K, top_k))
        vector = self._embeddings.embed_query(query)
        return [
            RetrievalResult(
                content=hit["content"],
                source=hit["source"],
                section=hit["section"],
                chunk_id=hit["chunk_id"],
                similarity=hit["similarity"],
            )
            for hit in self._store.query(vector, top_k=k)
        ]
