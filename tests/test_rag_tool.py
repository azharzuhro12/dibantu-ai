"""Tests for the search_knowledge_base agent tool (Step 12): registry
schema, validation, integration with the real RAG service, and error
dicts (never exceptions) for unavailable knowledge bases."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from app.rag.knowledge_tool import search_knowledge_base
from app.rag.service import RagService, reset_rag_service, set_rag_service
from app.tools import registry


def test_tool_is_registered_with_business_tools() -> None:
    assert "search_knowledge_base" in registry.list_tools()
    assert registry.get_tool("search_knowledge_base") is search_knowledge_base


def test_tool_schema_shape() -> None:
    schema = {
        s["name"]: s for s in registry.get_tool_schemas()
    }["search_knowledge_base"]
    assert schema["input_schema"]["required"] == ["query"]
    properties = schema["input_schema"]["properties"]
    assert properties["query"]["type"] == "string"
    assert properties["top_k"]["type"] == "integer"
    assert properties["top_k"]["minimum"] == 1
    assert properties["top_k"]["maximum"] == 10
    assert "knowledge" in schema["description"].lower()


def test_tool_returns_cited_passages(rag_service) -> None:
    result = search_knowledge_base("aturan refund pesanan yang sudah dibayar")
    assert result["success"] is True
    assert result["results"]
    assert result["results"][0]["source"] == "refund_policy.md"
    assert "refund_policy.md" in result["sources"]


def test_tool_validates_query() -> None:
    for bad in ("", "   ", None, 123):
        result = search_knowledge_base(bad)  # type: ignore[arg-type]
        assert result["success"] is False
        assert result["error"] == "VALIDATION_ERROR"


def test_tool_validates_top_k_type() -> None:
    result = search_knowledge_base("refund", top_k="3")  # type: ignore[arg-type]
    assert result["success"] is False
    assert result["error"] == "VALIDATION_ERROR"


def test_tool_clamps_top_k_instead_of_erroring(rag_service) -> None:
    result = search_knowledge_base("refund", top_k=99)
    assert result["success"] is True
    assert len(result["results"]) <= 10


def test_tool_on_empty_store_reports_availability(tmp_path) -> None:
    service = RagService(
        knowledge_dir=str(ROOT / "data" / "knowledge"),
        store_dir=str(tmp_path / "chroma"),
        embeddings_provider="hashing",
    )
    set_rag_service(service)
    try:
        result = search_knowledge_base("refund")
        assert result["success"] is True
        assert result["results"] == []
        assert "ingest" in result["message"].lower()
    finally:
        reset_rag_service()


def test_tool_reports_unavailable_stack_as_error_dict(tmp_path) -> None:
    """Misconfigured stack -> structured error, never an exception."""
    service = RagService(
        knowledge_dir=str(ROOT / "data" / "knowledge"),
        store_dir=str(tmp_path / "chroma"),
        embeddings_provider="no-such-provider",
    )
    set_rag_service(service)
    try:
        result = search_knowledge_base("refund")
        assert result["success"] is False
        assert result["error"] == "KNOWLEDGE_BASE_UNAVAILABLE"
        assert "message" in result
    finally:
        reset_rag_service()


def test_tool_result_is_json_serialisable_for_the_agent(rag_service) -> None:
    import json

    payload = json.dumps(
        search_knowledge_base("refund"), ensure_ascii=False, default=str
    )
    assert "refund_policy.md" in payload
