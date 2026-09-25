"""ChromaDB-backed persistent vector store for the RAG knowledge base.

A thin wrapper over one Chroma persistent collection (cosine space).
``chromadb`` is imported lazily inside the constructor so the rest of
the application — including the agent and its tool registry — imports
and runs fine in environments without the heavy RAG dependencies; the
failure then surfaces as a ``VectorStoreError`` when the knowledge base
is actually used.

The collection records which embedding provider and dimension built it
(Chroma collection metadata). Opening a store built by a different
provider raises immediately instead of silently mixing vector spaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .chunker import KnowledgeChunk
from .errors import RagError

__all__ = ["COLLECTION_NAME", "ChromaVectorStore", "VectorStoreError"]


class VectorStoreError(RagError):
    """Raised when the vector store cannot be opened or queried."""


#: Single Chroma collection used for all knowledge chunks.
COLLECTION_NAME = "dibantu_knowledge"

#: Client cache: one PersistentClient per directory. Reusing clients
#: avoids re-opening the same SQLite/Rust core twice in one process.
_CLIENT_CACHE: dict[str, Any] = {}


class ChromaVectorStore:
    """Persistent knowledge-chunk collection rooted at ``directory``."""

    def __init__(
        self,
        directory: str | Path,
        *,
        provider_name: str,
        provider_dimension: int,
    ) -> None:
        self.directory = str(Path(directory))
        self._collection = self._open_collection(provider_name, provider_dimension)

    # -- construction ------------------------------------------------------

    def _open_collection(
        self, provider_name: str, provider_dimension: int
    ) -> Any:
        """Open (or create) the knowledge collection and verify its origin."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:  # pragma: no cover - env-specific
            raise VectorStoreError(
                "chromadb is not installed; install the RAG dependencies "
                "(see requirements.txt)."
            ) from exc
        try:
            client = _CLIENT_CACHE.get(self.directory)
            if client is None:
                client = chromadb.PersistentClient(
                    path=self.directory,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                _CLIENT_CACHE[self.directory] = client
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_provider": provider_name,
                    "embedding_dimension": provider_dimension,
                },
            )
        except Exception as exc:  # noqa: BLE001 - chroma failures vary
            raise VectorStoreError(
                f"Could not open vector store at '{self.directory}': "
                f"{type(exc).__name__}"
            ) from exc
        self._verify_provider(collection, provider_name, provider_dimension)
        return collection

    def _verify_provider(
        self, collection: Any, provider_name: str, provider_dimension: int
    ) -> None:
        """Refuse to mix vector spaces from different embedding providers."""
        metadata = dict(collection.metadata or {})
        existing_provider = metadata.get("embedding_provider")
        existing_dimension = metadata.get("embedding_dimension")
        if existing_provider is None:
            return  # legacy/empty collection: nothing recorded yet
        if existing_provider != provider_name or existing_dimension != (
            provider_dimension
        ):
            raise VectorStoreError(
                f"Vector store at '{self.directory}' was built with "
                f"'{existing_provider}' embeddings (dimension "
                f"{existing_dimension}), but '{provider_name}' "
                f"(dimension {provider_dimension}) is configured. Delete "
                "the store directory and re-run ingestion."
            )

    # -- writes ------------------------------------------------------------

    def replace_source(
        self,
        source: str,
        chunks: list[KnowledgeChunk],
        vectors: list[list[float]],
        doc_hash: str,
    ) -> None:
        """Atomically (re)write every chunk of one document.

        Deletes the source's existing chunks first, then adds the new
        ones with ``doc_hash`` stamped on each, which makes ingestion
        idempotent per document: unchanged documents are skipped earlier
        (see the service), changed documents are fully replaced so no
        stale chunks survive.
        """
        self.delete_source(source)
        if not chunks:
            return
        self._collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=vectors,
            metadatas=[
                {
                    "source": chunk.source,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
                    "doc_hash": doc_hash,
                }
                for chunk in chunks
            ],
        )

    def delete_source(self, source: str) -> None:
        """Remove every chunk belonging to one document."""
        self._collection.delete(where={"source": source})

    # -- reads -------------------------------------------------------------

    def get_doc_hash(self, source: str) -> str | None:
        """Return the recorded content hash of a stored document."""
        got = self._collection.get(
            where={"source": source}, limit=1, include=["metadatas"]
        )
        if not got["ids"]:
            return None
        return got["metadatas"][0].get("doc_hash")

    def count(self) -> int:
        """Total number of stored chunks."""
        return self._collection.count()

    def list_sources(self) -> list[str]:
        """Sorted names of every document present in the store."""
        got = self._collection.get(include=["metadatas"])
        return sorted({meta["source"] for meta in got["metadatas"]})

    def query(
        self, vector: list[float], *, top_k: int
    ) -> list[dict[str, Any]]:
        """Return the ``top_k`` nearest chunks as raw result dicts."""
        if self.count() == 0 or top_k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        ids = result.get("ids") or [[]]
        for i, chunk_id in enumerate(ids[0]):
            distance = float(result["distances"][0][i])
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "content": result["documents"][0][i],
                    "source": result["metadatas"][0][i].get("source", ""),
                    "section": result["metadatas"][0][i].get("section", ""),
                    # cosine distance -> similarity in [-1, 1]
                    "similarity": round(1.0 - distance, 4),
                }
            )
        return hits
