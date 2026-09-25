"""High-level RAG service: ingestion and search (Step 12).

``RagService`` owns the whole retrieval pipeline — knowledge directory
in, ChromaDB vector store out — and is what both the agent tool and the
API endpoints talk to. Construction is cheap; the embedding model and
the Chroma client load lazily on first actual use, so importing this
module (via the tool registry) never pulls in torch or chromadb.

Ingestion is idempotent per document: each stored chunk carries its
document's content hash (``doc_hash``), so re-running ingestion skips
unchanged documents, fully replaces changed ones (no stale chunks), and
reports exactly what happened.

``get_rag_service()`` returns a process-wide singleton configured from
the app settings; ``set_rag_service`` / ``reset_rag_service`` are the
seams tests and the evaluator use to point the stack at an isolated
scratch store — the same idea as the scratch-database isolation in
``evaluation/db_isolation.py``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings

from .chunker import KnowledgeChunk, chunk_document
from .embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingProvider,
    build_embeddings,
)
from .errors import KnowledgeDirNotFoundError, RagError
from .retriever import KnowledgeRetriever
from .vector_store import ChromaVectorStore

__all__ = [
    "IngestReport",
    "RagService",
    "get_rag_service",
    "load_knowledge_documents",
    "reset_rag_service",
    "set_rag_service",
]


@dataclass(frozen=True)
class IngestReport:
    """What one ingestion run did, in plain counts."""

    documents_found: int
    documents_created: int
    documents_updated: int
    documents_skipped: int
    chunks_created: int
    total_chunks: int
    store_dir: str
    sources: tuple[str, ...]

    @property
    def documents_processed(self) -> int:
        """Documents actually re-chunked (created or updated)."""
        return self.documents_created + self.documents_updated


def load_knowledge_documents(directory: str | Path) -> list[tuple[str, str]]:
    """Read every ``*.md`` file directly inside ``directory``.

    Returns ``(file_name, text)`` pairs sorted by name — deterministic
    across runs and machines. Only the configured directory is ever
    read; nothing user-supplied can point ingestion at another path.
    """
    path = Path(directory)
    if not path.is_dir():
        raise KnowledgeDirNotFoundError(
            f"Knowledge directory '{path}' does not exist."
        )
    documents: list[tuple[str, str]] = []
    for file in sorted(path.glob("*.md")):
        documents.append((file.name, file.read_text(encoding="utf-8")))
    return documents


class RagService:
    """Ingestion + search over one knowledge directory and vector store."""

    def __init__(
        self,
        *,
        knowledge_dir: str | Path,
        store_dir: str | Path,
        embeddings_provider: str = "local",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        top_k: int = 3,
    ) -> None:
        self.knowledge_dir = str(Path(knowledge_dir))
        self.store_dir = str(Path(store_dir))
        self.embeddings_provider = embeddings_provider
        self.embedding_model = embedding_model
        self.top_k = top_k
        self._embeddings: EmbeddingProvider | None = None
        self._store: ChromaVectorStore | None = None

    # -- lazy component access --------------------------------------------

    @property
    def embeddings(self) -> EmbeddingProvider:
        """The configured embedding provider (built on first use)."""
        if self._embeddings is None:
            self._embeddings = build_embeddings(
                self.embeddings_provider, self.embedding_model
            )
        return self._embeddings

    @property
    def store(self) -> ChromaVectorStore:
        """The persistent Chroma store (opened on first use)."""
        if self._store is None:
            self._store = ChromaVectorStore(
                self.store_dir,
                provider_name=self.embeddings.name,
                provider_dimension=self.embeddings.dimension,
            )
        return self._store

    # -- ingestion ---------------------------------------------------------

    def ingest(self) -> IngestReport:
        """(Re)ingest every knowledge document into the vector store.

        Idempotent: a document whose content hash already matches the
        stored one is skipped; a changed document has all its chunks
        replaced. Embedding every chunk of a new/changed document is
        the only expensive step (one model pass per chunk, locally).
        """
        documents = load_knowledge_documents(self.knowledge_dir)
        store = self.store
        embeddings = self.embeddings

        created = updated = skipped = chunks_created = 0
        for source, text in documents:
            doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            existing = store.get_doc_hash(source)
            if existing == doc_hash:
                skipped += 1
                continue
            if existing is None:
                created += 1
            else:
                updated += 1
            chunks: list[KnowledgeChunk] = chunk_document(text, source)
            vectors = embeddings.embed_documents(
                [chunk.content for chunk in chunks]
            )
            store.replace_source(source, chunks, vectors, doc_hash)
            chunks_created += len(chunks)

        return IngestReport(
            documents_found=len(documents),
            documents_created=created,
            documents_updated=updated,
            documents_skipped=skipped,
            chunks_created=chunks_created,
            total_chunks=store.count(),
            store_dir=self.store_dir,
            sources=tuple(store.list_sources()),
        )

    # -- search ------------------------------------------------------------

    def search(self, query: str, *, top_k: int | None = None) -> dict[str, Any]:
        """Retrieve the passages relevant to ``query``.

        Returns the same dict shape the agent tool passes to the model:
        ``{"success", "query", "results": [...], "sources": [...]}``,
        each result carrying ``content``/``source``/``section``/
        ``chunk_id``/``similarity``. An empty store is a success with a
        hint to run ingestion, not an error — the agent should tell the
        user the knowledge is unavailable, not that the system crashed.
        """
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Query is required.")
        if self.store.count() == 0:
            return {
                "success": True,
                "query": cleaned,
                "results": [],
                "sources": [],
                "message": (
                    "Knowledge base is empty. Run ingestion first "
                    "(python -m app.rag.ingest or POST /api/knowledge/ingest)."
                ),
            }
        retriever = KnowledgeRetriever(self.store, self.embeddings)
        effective_k = self.top_k if top_k is None else top_k
        results = retriever.retrieve(cleaned, top_k=effective_k)
        sources: list[str] = []
        for result in results:
            if result.source not in sources:
                sources.append(result.source)
        return {
            "success": True,
            "query": cleaned,
            "results": [
                {
                    "content": result.content,
                    "source": result.source,
                    "section": result.section,
                    "chunk_id": result.chunk_id,
                    "similarity": result.similarity,
                }
                for result in results
            ],
            "sources": sources,
        }


# ---------------------------------------------------------------------------
# Process-wide singleton (settings-driven, test/eval-overridable)
# ---------------------------------------------------------------------------

_service: RagService | None = None


def get_rag_service() -> RagService:
    """Return the shared RagService, built from settings on first use."""
    global _service
    if _service is None:
        settings: Settings = get_settings()
        _service = RagService(
            knowledge_dir=settings.knowledge_dir,
            store_dir=settings.rag_store_dir,
            embeddings_provider=settings.rag_embeddings,
            embedding_model=settings.rag_embedding_model,
            top_k=settings.rag_top_k,
        )
    return _service


def set_rag_service(service: RagService | None) -> None:
    """Override the shared service (tests, evaluator) or reset with None."""
    global _service
    _service = service


def reset_rag_service() -> None:
    """Restore settings-driven construction on next ``get_rag_service``."""
    set_rag_service(None)
