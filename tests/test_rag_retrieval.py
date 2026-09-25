"""Tests for RAG retrieval (Step 12): semantic search, metadata,
sources, top_k bounds, and the empty-store path."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from app.rag.service import RagService


def test_refund_question_retrieves_refund_policy(rag_service) -> None:
    result = rag_service.search(
        "Apa aturan refund untuk pesanan yang sudah dibayar?"
    )
    assert result["success"] is True
    assert result["results"], "expected at least one passage"
    assert result["results"][0]["source"] == "refund_policy.md"
    assert "refund" in result["results"][0]["content"].lower()


def test_low_stock_question_retrieves_inventory_policy(rag_service) -> None:
    result = rag_service.search("ambang stok rendah standar berapa?")
    assert result["results"]
    sources = [hit["source"] for hit in result["results"]]
    assert "inventory_policy.md" in sources[:2]


def test_results_carry_full_citation_metadata(rag_service) -> None:
    result = rag_service.search("proses refund", top_k=3)
    for hit in result["results"]:
        assert hit["content"].strip()
        assert hit["source"].endswith(".md")
        assert isinstance(hit["section"], str)
        assert hit["chunk_id"]
        assert -1.0 <= hit["similarity"] <= 1.0


def test_sources_list_is_unique_and_ordered(rag_service) -> None:
    result = rag_service.search("refund", top_k=5)
    sources = result["sources"]
    assert len(sources) == len(set(sources))
    for hit in result["results"]:
        assert hit["source"] in sources


def test_top_k_is_honoured_and_clamped(rag_service) -> None:
    assert len(rag_service.search("refund", top_k=1)["results"]) == 1
    assert len(rag_service.search("refund", top_k=2)["results"]) == 2
    # Out-of-range values are clamped, never an error for the model.
    assert len(rag_service.search("refund", top_k=99)["results"]) <= 10
    assert len(rag_service.search("refund", top_k=0)["results"]) == 1


def test_default_top_k_comes_from_settings(rag_service) -> None:
    assert rag_service.top_k == 3
    assert len(rag_service.search("refund")["results"]) == 3


def test_empty_store_returns_hint_not_error(tmp_path) -> None:
    service = RagService(
        knowledge_dir=str(ROOT / "data" / "knowledge"),
        store_dir=str(tmp_path / "chroma"),
        embeddings_provider="hashing",
    )  # never ingested
    result = service.search("refund")
    assert result["success"] is True
    assert result["results"] == []
    assert result["sources"] == []
    assert "ingest" in result["message"].lower()


def test_blank_query_raises_value_error(rag_service) -> None:
    with pytest.raises(ValueError):
        rag_service.search("   ")


def test_vector_store_rejects_mixed_providers(tmp_path) -> None:
    """A store built with hashing embeddings refuses a 'local' reopen."""
    from app.rag.vector_store import ChromaVectorStore, VectorStoreError

    store_dir = str(tmp_path / "chroma")
    ChromaVectorStore(store_dir, provider_name="hashing", provider_dimension=384)
    with pytest.raises(VectorStoreError, match="hashing"):
        ChromaVectorStore(store_dir, provider_name="local", provider_dimension=384)
