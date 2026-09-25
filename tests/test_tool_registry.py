"""Tests for app.tools.registry (Step 4A → Step 14: + knowledge, memory)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from app.memory import memory_tools
from app.rag.knowledge_tool import search_knowledge_base
from app.tools import business_tools, registry

EXPECTED_TOOLS = {
    "check_stock",
    "create_order",
    "update_stock",
    "search_customer",
    "get_sales_report",
    "get_low_stock",
    "search_knowledge_base",
    "save_memory",
    "search_memory",
    "delete_memory",
}

EXPECTED_REQUIRED = {
    "check_stock": {"product_name"},
    "create_order": {"customer_name", "items"},
    "update_stock": {"product_name", "quantity_change"},
    "search_customer": {"name"},
    "get_sales_report": set(),
    "get_low_stock": {"threshold"},
    "search_knowledge_base": {"query"},
    "save_memory": {"content"},
    "search_memory": {"query"},
    "delete_memory": {"memory_id"},
}


def test_list_tools_registers_all_ten_tools():
    assert set(registry.list_tools()) == EXPECTED_TOOLS
    assert len(registry.list_tools()) == 10


def test_reset_mock_data_is_not_exposed():
    assert "reset_mock_data" not in registry.list_tools()


def test_list_tools_returns_a_copy():
    names = registry.list_tools()
    names.append("rogue_tool")
    assert "rogue_tool" not in registry.list_tools()


@pytest.mark.parametrize(
    "name, func",
    [
        ("check_stock", business_tools.check_stock),
        ("create_order", business_tools.create_order),
        ("update_stock", business_tools.update_stock),
        ("search_customer", business_tools.search_customer),
        ("get_sales_report", business_tools.get_sales_report),
        ("get_low_stock", business_tools.get_low_stock),
        ("search_knowledge_base", search_knowledge_base),
        ("save_memory", memory_tools.save_memory),
        ("search_memory", memory_tools.search_memory),
        ("delete_memory", memory_tools.delete_memory),
    ],
)
def test_get_tool_returns_the_underlying_function(name, func):
    assert registry.get_tool(name) is func


def test_get_tool_unknown_name_returns_none():
    assert registry.get_tool("does_not_exist") is None
    assert registry.get_tool("") is None


def test_get_tool_schemas_covers_every_registered_tool():
    schemas = registry.get_tool_schemas()
    advertised = set(EXPECTED_TOOLS) | set(registry.SENSITIVE_REQUEST_TOOLS)
    assert {schema["name"] for schema in schemas} == advertised
    assert len(schemas) == 13  # 10 dispatchable + 3 sensitive-request


def test_sensitive_actions_are_advertised_but_never_dispatchable():
    """Requesting a sensitive action creates an approval; it can never
    be dispatched through the registry (Step 16)."""
    for name in registry.SENSITIVE_REQUEST_TOOLS:
        assert registry.get_tool(name) is None
        assert name not in registry.list_tools()
        assert name in {schema["name"] for schema in registry.get_tool_schemas()}


def test_schemas_use_the_anthropic_tool_format():
    for schema in registry.get_tool_schemas():
        assert set(schema) == {"name", "description", "input_schema"}
        assert isinstance(schema["name"], str)
        assert isinstance(schema["description"], str) and schema["description"]
        input_schema = schema["input_schema"]
        assert input_schema["type"] == "object"
        assert isinstance(input_schema["properties"], dict)
        assert set(input_schema["required"]) <= set(input_schema["properties"])


def test_required_fields_match_the_tool_signatures():
    schemas = {s["name"]: s for s in registry.get_tool_schemas()}
    for name, required in EXPECTED_REQUIRED.items():
        assert set(schemas[name]["input_schema"]["required"]) == required


def test_period_property_exposes_the_valid_periods():
    schema = {s["name"]: s for s in registry.get_tool_schemas()}["get_sales_report"]
    period = schema["input_schema"]["properties"]["period"]
    assert period["enum"] == list(business_tools.VALID_PERIODS)
    assert "daily" in period["enum"]


def test_get_tool_schemas_returns_deep_copies():
    first = {s["name"]: s for s in registry.get_tool_schemas()}
    first["check_stock"]["description"] = "MUTATED"
    first["check_stock"]["input_schema"]["required"] = []

    second = {s["name"]: s for s in registry.get_tool_schemas()}
    assert second["check_stock"]["description"] != "MUTATED"
    assert second["check_stock"]["input_schema"]["required"] == ["product_name"]


def test_registered_tools_are_wired_to_the_real_module():
    result = registry.get_tool("check_stock")("croissant")
    assert result["success"] is True
