"""Tests for app.rag.embeddings (Step 12): local + hashing providers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from app.rag.embeddings import (
    EMBEDDING_DIMENSION,
    EmbeddingError,
    HashingEmbeddings,
    build_embeddings,
)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def test_hashing_embeddings_are_deterministic() -> None:
    provider = HashingEmbeddings()
    assert provider.embed_query("refund pesanan") == provider.embed_query(
        "refund pesanan"
    )
    assert provider.embed_documents(["a", "b"]) == provider.embed_documents(
        ["a", "b"]
    )


def test_hashing_embeddings_have_expected_dimension() -> None:
    provider = HashingEmbeddings()
    assert provider.dimension == EMBEDDING_DIMENSION == 384
    assert len(provider.embed_query("apa aturan refund")) == 384


def test_hashing_embeddings_are_l2_normalised() -> None:
    provider = HashingEmbeddings()
    vector = provider.embed_query("kebijakan stok rendah")
    norm = math.sqrt(sum(value * value for value in vector))
    assert norm == pytest.approx(1.0, abs=1e-9)


def test_hashing_embeddings_empty_text_is_zero_vector() -> None:
    provider = HashingEmbeddings()
    assert provider.embed_query("!!! ???") == [0.0] * 384


def test_hashing_embeddings_rank_shared_vocabulary_closer() -> None:
    provider = HashingEmbeddings()
    query = provider.embed_query("aturan refund pesanan yang sudah dibayar")
    relevant = provider.embed_query(
        "Refund hanya diajukan untuk pesanan yang sudah dibayar."
    )
    unrelated = provider.embed_query(
        "Kedai buka setiap hari pukul 08.00 sampai 21.00."
    )
    assert _cosine(query, relevant) > _cosine(query, unrelated)


def test_build_embeddings_rejects_unknown_provider() -> None:
    with pytest.raises(EmbeddingError):
        build_embeddings("openai")


def test_build_embeddings_returns_hashing_provider() -> None:
    provider = build_embeddings("hashing")
    assert isinstance(provider, HashingEmbeddings)
    assert provider.name == "hashing"


def test_local_embeddings_load_model_and_encode() -> None:
    """The real sentence-transformers model (skipped when unavailable).

    First run downloads all-MiniLM-L6-v2 (~90 MB) into the local
    HuggingFace cache; offline machines simply skip — every other test
    uses the deterministic hashing provider instead.
    """
    pytest.importorskip("sentence_transformers")
    from app.rag.embeddings import LocalEmbeddings

    try:
        provider = LocalEmbeddings()
    except EmbeddingError:
        pytest.skip("embedding model not downloadable in this environment")

    assert provider.name == "local"
    assert provider.dimension == 384
    first = provider.embed_query("kebijakan refund")
    second = provider.embed_query("kebijakan refund")
    assert first == pytest.approx(second, abs=1e-6)
    batch = provider.embed_documents(["kebijakan refund", "stok rendah"])
    assert len(batch) == 2
    assert all(len(vector) == 384 for vector in batch)
    norm = math.sqrt(sum(value * value for value in first))
    assert norm == pytest.approx(1.0, abs=1e-4)
