"""Tests for GLM tool calling (Step 4B).

Part 1 tests ``_parse_response`` as a pure function: distinguishing
plain text replies from tool_use blocks in Anthropic-compatible
payloads. Part 2 tests ``complete_with_tools`` over a mock HTTP
transport: the request payload must carry the ``tools`` field and the
response must parse into a GLMResponse. No real network is involved.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from app.agent.glm_client import (
    GLMAPIError,
    GLMAuthError,
    GLMClient,
    GLMResponse,
    _parse_response,
)
from app.config import Settings
from app.tools.registry import get_tool_schemas


def _text_block(text: str) -> dict[str, Any]:
    """Build a text content block."""
    return {"type": "text", "text": text}


def _tool_block(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Build a tool_use content block."""
    return {
        "type": "tool_use",
        "id": f"toolu_{name}",
        "name": name,
        "input": tool_input,
    }


def _response(
    content: list[dict[str, Any]], stop_reason: str | None = "end_turn"
) -> dict[str, Any]:
    """Build a minimal Anthropic-style response body."""
    body: dict[str, Any] = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": content,
    }
    if stop_reason is not None:
        body["stop_reason"] = stop_reason
    return body


def test_text_response_parses_text_and_stop_reason() -> None:
    result = _parse_response(_response([_text_block("Stok kopi: 10 unit.")]))

    assert isinstance(result, GLMResponse)
    assert result.text == "Stok kopi: 10 unit."
    assert result.tool_name is None
    assert result.tool_input is None
    assert result.stop_reason == "end_turn"


def test_multiple_text_blocks_are_concatenated() -> None:
    data = _response([_text_block("Stok kopi: "), _text_block("10 unit.")])

    assert _parse_response(data).text == "Stok kopi: 10 unit."


def test_tool_use_response_parses_name_and_input() -> None:
    data = _response(
        [
            _text_block("Saya cek dulu ya."),
            _tool_block("check_stock", {"product_name": "kopi"}),
        ],
        stop_reason="tool_use",
    )

    result = _parse_response(data)

    assert result.text == "Saya cek dulu ya."
    assert result.tool_name == "check_stock"
    assert result.tool_input == {"product_name": "kopi"}
    assert result.stop_reason == "tool_use"


def test_tool_use_without_text_is_valid() -> None:
    data = _response(
        [_tool_block("get_sales_report", {})],
        stop_reason="tool_use",
    )

    result = _parse_response(data)

    assert result.text is None
    assert result.tool_name == "get_sales_report"
    assert result.tool_input == {}
    assert result.stop_reason == "tool_use"


def test_first_tool_use_block_wins() -> None:
    data = _response(
        [
            _tool_block("check_stock", {"product_name": "kopi"}),
            _tool_block("create_order", {"product_name": "teh"}),
        ],
        stop_reason="tool_use",
    )

    result = _parse_response(data)

    assert result.tool_name == "check_stock"
    assert result.tool_input == {"product_name": "kopi"}


def test_missing_input_defaults_to_empty_dict() -> None:
    block = _tool_block("get_low_stock", {})
    del block["input"]
    data = _response([block], stop_reason="tool_use")

    result = _parse_response(data)

    assert result.tool_input == {}


def test_missing_stop_reason_is_none() -> None:
    data = _response([_text_block("Halo!")], stop_reason=None)

    assert _parse_response(data).stop_reason is None


def test_no_text_or_tool_use_raises() -> None:
    with pytest.raises(GLMAPIError):
        _parse_response(_response([]))


def test_malformed_payload_without_content_raises() -> None:
    with pytest.raises(GLMAPIError):
        _parse_response({"id": "msg_test", "type": "message"})


# ---------------------------------------------------------------------------
# HTTP tests (Step 4B, part 2): complete_with_tools over a mock transport.
# ---------------------------------------------------------------------------


def _make_client(
    response: httpx.Response, captured: list[httpx.Request]
) -> GLMClient:
    """Build a GLMClient whose HTTP calls hit a recording mock transport."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return response

    settings = Settings(
        glm_api_key="test-key",
        glm_base_url="https://glm.test.example",
        glm_model="glm-test-model",
    )
    return GLMClient(settings, transport=httpx.MockTransport(handler))


def _sent_payload(captured: list[httpx.Request]) -> dict[str, Any]:
    """Decode the JSON body of the single captured request."""
    assert len(captured) == 1
    return json.loads(captured[0].content)


def test_complete_with_tools_sends_tools_in_payload() -> None:
    captured: list[httpx.Request] = []
    body = _response(
        [
            _text_block("Saya cek dulu ya."),
            _tool_block("check_stock", {"product_name": "kopi"}),
        ],
        stop_reason="tool_use",
    )
    client = _make_client(httpx.Response(200, json=body), captured)
    tools = get_tool_schemas()

    result = asyncio.run(
        client.complete_with_tools(
            [{"role": "user", "content": "Berapa stok kopi?"}],
            system="You are a helpful assistant.",
            tools=tools,
        )
    )

    payload = _sent_payload(captured)
    assert payload["tools"] == tools
    expected_messages = [{"role": "user", "content": "Berapa stok kopi?"}]
    assert payload["messages"] == expected_messages
    assert payload["system"] == "You are a helpful assistant."
    assert isinstance(result, GLMResponse)
    assert result.tool_name == "check_stock"
    assert result.tool_input == {"product_name": "kopi"}
    assert result.stop_reason == "tool_use"


def test_complete_with_tools_parses_text_response() -> None:
    captured: list[httpx.Request] = []
    body = _response([_text_block("Stok kopi: 10 unit.")])
    client = _make_client(httpx.Response(200, json=body), captured)

    result = asyncio.run(
        client.complete_with_tools(
            [{"role": "user", "content": "Berapa stok kopi?"}],
            tools=get_tool_schemas(),
        )
    )

    assert isinstance(result, GLMResponse)
    assert result.text == "Stok kopi: 10 unit."
    assert result.tool_name is None
    assert result.stop_reason == "end_turn"


def test_complete_with_tools_uses_same_http_and_auth_as_complete() -> None:
    captured: list[httpx.Request] = []
    body = _response([_text_block("Oke.")])
    client = _make_client(httpx.Response(200, json=body), captured)

    asyncio.run(
        client.complete_with_tools(
            [{"role": "user", "content": "Halo"}],
            tools=[],
        )
    )

    request = captured[0]
    assert str(request.url) == "https://glm.test.example/v1/messages"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"


def test_complete_with_tools_raises_auth_error_on_401() -> None:
    client = _make_client(
        httpx.Response(401, json={"error": "unauthorized"}), []
    )

    with pytest.raises(GLMAuthError):
        asyncio.run(
            client.complete_with_tools(
                [{"role": "user", "content": "Halo"}],
                tools=get_tool_schemas(),
            )
        )
