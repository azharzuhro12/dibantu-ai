"""Unit tests for the tool-calling loop in ``app.agent.agent``.

All tests drive ``Agent`` with a fake GLM client so no network calls
are made: the direct text-only reply, a single tool execution, chained
multi-step tool calls, a request for an unknown tool, a tool that
raises an exception, and the tool-iteration cap forcing a final reply.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.agent.agent import Agent


@dataclass
class FakeGLMResponse:
    """Minimal stand-in for ``GLMResponse`` as consumed by ``Agent``."""

    text: str = ""
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None


class FakeGLMClient:
    """Test double yielding queued responses and recording every call."""

    def __init__(self, responses: list[FakeGLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str,
        tools: list[dict[str, Any]],
    ) -> FakeGLMResponse:
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "system": system,
                "tools": tools,
            }
        )
        return self._responses.pop(0)


def test_agent_returns_text_response_without_tool() -> None:
    """A text-only GLM reply is returned immediately, with one GLM call."""
    fake = FakeGLMClient([FakeGLMResponse(text="Semua sistem berjalan normal.")])
    agent = Agent(fake, tools={}, schemas=[])

    result = asyncio.run(agent.run("Apa kabar?"))

    assert result == "Semua sistem berjalan normal."
    assert len(fake.calls) == 1
    assert fake.calls[0]["messages"] == [{"role": "user", "content": "Apa kabar?"}]


def test_agent_runs_one_tool_then_returns_final_text() -> None:
    """A tool request is executed and its result feeds the final reply."""
    fake = FakeGLMClient(
        [
            FakeGLMResponse(tool_name="get_weather", tool_input={"city": "Jakarta"}),
            FakeGLMResponse(text="Cuaca di Jakarta cerah."),
        ]
    )
    seen: list[dict[str, Any]] = []

    def get_weather(city: str) -> str:
        seen.append({"city": city})
        return "cerah"

    agent = Agent(fake, tools={"get_weather": get_weather}, schemas=[])

    result = asyncio.run(agent.run("Bagaimana cuaca di Jakarta?"))

    assert result == "Cuaca di Jakarta cerah."
    assert seen == [{"city": "Jakarta"}]
    assert len(fake.calls) == 2
    second_messages = fake.calls[1]["messages"]
    assert second_messages[0] == {
        "role": "user",
        "content": "Bagaimana cuaca di Jakarta?",
    }
    assert second_messages[1] == {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_get_weather_0",
                "name": "get_weather",
                "input": {"city": "Jakarta"},
            }
        ],
    }
    assert second_messages[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_get_weather_0",
                "content": "cerah",
            }
        ],
    }


def test_agent_handles_multi_step_tool_calls() -> None:
    """Two chained tool requests run in order before the final text reply."""
    fake = FakeGLMClient(
        [
            FakeGLMResponse(tool_name="get_weather", tool_input={"city": "Jakarta"}),
            FakeGLMResponse(
                tool_name="convert_temperature", tool_input={"celsius": 30}
            ),
            FakeGLMResponse(text="Suhu di Jakarta 30C, setara 86F."),
        ]
    )
    seen: list[tuple[str, dict[str, Any]]] = []

    def get_weather(city: str) -> str:
        seen.append(("get_weather", {"city": city}))
        return "cerah, 30C"

    def convert_temperature(celsius: int) -> str:
        seen.append(("convert_temperature", {"celsius": celsius}))
        return f"{celsius * 9 // 5 + 32}F"

    agent = Agent(
        fake,
        tools={"get_weather": get_weather, "convert_temperature": convert_temperature},
        schemas=[],
    )

    result = asyncio.run(agent.run("Berapa suhu Jakarta dalam Fahrenheit?"))

    assert result == "Suhu di Jakarta 30C, setara 86F."
    assert seen == [
        ("get_weather", {"city": "Jakarta"}),
        ("convert_temperature", {"celsius": 30}),
    ]
    assert len(fake.calls) == 3
    third_messages = fake.calls[2]["messages"]
    assert [message["role"] for message in third_messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert third_messages[1]["content"][0]["id"] == "toolu_get_weather_0"
    assert third_messages[3]["content"][0]["id"] == "toolu_convert_temperature_1"
    assert third_messages[4]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_convert_temperature_1",
        "content": "86F",
    }


def test_agent_handles_unknown_tool() -> None:
    """An unknown tool request becomes an error result, not a crash."""
    fake = FakeGLMClient(
        [
            FakeGLMResponse(tool_name="teleport", tool_input={"city": "Bandung"}),
            FakeGLMResponse(text="Maaf, saya tidak bisa melakukan teleportasi."),
        ]
    )

    agent = Agent(fake, tools={}, schemas=[])

    result = asyncio.run(agent.run("Teleport saya ke Bandung."))

    assert result == "Maaf, saya tidak bisa melakukan teleportasi."
    assert len(fake.calls) == 2
    second_messages = fake.calls[1]["messages"]
    assert second_messages[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_teleport_0",
                "content": '{"error": "Unknown tool: teleport"}',
            }
        ],
    }


def test_agent_handles_tool_exception() -> None:
    """A raising tool becomes an error result; the loop still finishes."""
    fake = FakeGLMClient(
        [
            FakeGLMResponse(tool_name="get_weather", tool_input={"city": "Jakarta"}),
            FakeGLMResponse(text="Maaf, layanan cuaca sedang bermasalah."),
        ]
    )

    def broken_weather(city: str) -> str:
        raise RuntimeError("weather service unavailable")

    agent = Agent(fake, tools={"get_weather": broken_weather}, schemas=[])

    result = asyncio.run(agent.run("Bagaimana cuaca di Jakarta?"))

    assert result == "Maaf, layanan cuaca sedang bermasalah."
    assert len(fake.calls) == 2
    second_messages = fake.calls[1]["messages"]
    assert second_messages[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_get_weather_0",
                "content": '{"error": "RuntimeError: weather service unavailable"}',
            }
        ],
    }


def test_agent_respects_max_tool_iterations() -> None:
    """The loop stops after the iteration cap with one final GLM call."""
    fake = FakeGLMClient(
        [
            FakeGLMResponse(tool_name="get_weather", tool_input={"city": "Jakarta"}),
            FakeGLMResponse(tool_name="get_weather", tool_input={"city": "Bandung"}),
            FakeGLMResponse(text="Saya sudah memeriksa cuaca dua kota."),
        ]
    )
    executions: list[dict[str, Any]] = []

    def get_weather(city: str) -> str:
        executions.append({"city": city})
        return "cerah"

    agent = Agent(
        fake, tools={"get_weather": get_weather}, schemas=[], max_tool_iterations=2
    )

    result = asyncio.run(agent.run("Cek cuaca semua kota sampai selesai."))

    assert result == "Saya sudah memeriksa cuaca dua kota."
    assert len(executions) == 2
    assert len(fake.calls) == 3
    final_messages = fake.calls[2]["messages"]
    assert [message["role"] for message in final_messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert final_messages[2]["content"][0]["tool_use_id"] == "toolu_get_weather_0"
    assert final_messages[4]["content"][0]["tool_use_id"] == "toolu_get_weather_1"
