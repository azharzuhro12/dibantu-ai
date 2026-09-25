"""Tests for RAG ingestion (Step 12): loading, chunking, embedding,
ChromaDB persistence — and the idempotency contract."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from app.rag.errors import KnowledgeDirNotFoundError
from app.rag.service import RagService, load_knowledge_documents

DEMO_DOCS = ROOT / "data" / "knowledge"
DEMO_DOC_NAMES = {
    "business_policies.md",
    "inventory_policy.md",
    "order_policy.md",
    "refund_policy.md",
}


def test_load_knowledge_documents_reads_sorted_markdown() -> None:
    documents = load_knowledge_documents(DEMO_DOCS)
    names = [name for name, _ in documents]
    assert names == sorted(names)
    assert set(names) == DEMO_DOC_NAMES


def test_load_knowledge_documents_rejects_missing_dir(tmp_path) -> None:
    with pytest.raises(KnowledgeDirNotFoundError):
        load_knowledge_documents(tmp_path / "nowhere")


def test_first_ingest_creates_chunks(tmp_path) -> None:
    """A fresh store ingests every demo document once."""
    service = RagService(
        knowledge_dir=str(DEMO_DOCS),
        store_dir=str(tmp_path / "chroma"),
        embeddings_provider="hashing",
    )
    report = service.ingest()
    assert report.documents_found == 4
    assert report.documents_created == 4
    assert report.documents_updated == 0
    assert report.documents_skipped == 0
    assert report.documents_processed == 4
    assert report.chunks_created == report.total_chunks > 0
    assert set(report.sources) == DEMO_DOC_NAMES
    assert report.store_dir == str(tmp_path / "chroma")


def test_reingest_after_fixture_skips_everything(rag_service) -> None:
    """The fixture already ingested; a re-run skips all documents."""
    report = rag_service.ingest()
    assert report.documents_found == 4
    assert report.documents_created == 0
    assert report.documents_skipped == 4
    assert report.chunks_created == 0
    assert report.total_chunks > 0


def test_repeated_ingest_is_idempotent(rag_service) -> None:
    """Running ingestion twice must not create duplicate chunks."""
    first = rag_service.ingest()  # fixture ingested, so this is a re-run
    second = rag_service.ingest()
    assert first.total_chunks == second.total_chunks
    assert second.documents_skipped == 4
    assert second.chunks_created == 0
    assert rag_service.store.count() == first.total_chunks


def test_changed_document_is_replaced_not_duplicated(tmp_path) -> None:
    """Editing a doc replaces its chunks; stale chunks do not survive."""
    docs = tmp_path / "knowledge"
    docs.mkdir()
    (docs / "one.md").write_text(
        "# Satu\n\n" + "\n\n".join(f"Isi paragraf {i}." for i in range(5)),
        encoding="utf-8",
    )
    service = RagService(
        knowledge_dir=str(docs),
        store_dir=str(tmp_path / "chroma"),
        embeddings_provider="hashing",
    )
    first = service.ingest()
    baseline = first.total_chunks
    assert first.documents_created == 1

    # Same content again: skipped, store unchanged.
    unchanged = service.ingest()
    assert unchanged.documents_skipped == 1
    assert unchanged.total_chunks == baseline

    # Grow the document: fully replaced, not appended.
    (docs / "one.md").write_text(
        "# Satu\n\n" + "\n\n".join(f"Isi paragraf baru {i}." for i in range(40)),
        encoding="utf-8",
    )
    updated = service.ingest()
    assert updated.documents_updated == 1
    assert updated.documents_created == 0
    assert updated.total_chunks > baseline  # more content -> more chunks
    # Full replace: every stored chunk of this doc was written this run,
    # so no stale pre-edit chunk can survive.
    assert updated.chunks_created == updated.total_chunks
    assert service.store.count() == updated.total_chunks
    assert service.store.list_sources() == ["one.md"]


def test_empty_knowledge_dir_is_a_noop(tmp_path) -> None:
    (tmp_path / "empty").mkdir()
    service = RagService(
        knowledge_dir=str(tmp_path / "empty"),
        store_dir=str(tmp_path / "chroma"),
        embeddings_provider="hashing",
    )
    report = service.ingest()
    assert report.documents_found == 0
    assert report.total_chunks == 0
    assert report.sources == ()
