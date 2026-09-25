"""Persistent agent memory tests (Step 14).

Covers the PostgreSQL-backed memory store: CRUD, durability across
engine recreation (the testable slice of a backend restart), owner
isolation, deterministic keyword retrieval, agent integration (the
small context block + owner-scoped tools), the secret screen, the
memory API endpoints, and memory's inability to touch business data
or approvals (it is context only).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.agent.agent import AGENT_SYSTEM_PROMPT, DibantuAgent
from app.config import Settings
from app.main import app
from app.models.schemas import AgentRequest
from app.memory import (
    DEFAULT_OWNER_KEY,
    MEMORY_TYPES,
    MemoryNotFoundError,
    MemoryValidationError,
    create_memory,
    current_owner_key,
    delete_memory,
    get_memory,
    list_memories,
    memory_context_block,
    owner_scope,
    reset_memories,
    search_memories,
    update_memory,
)
from app.tools.business_tools import reset_mock_data


@pytest.fixture(autouse=True)
def clean_stores(test_database):
    """Start every test with an empty memory store and seeded business data."""
    reset_memories()
    reset_mock_data()
    yield
    reset_memories()
    reset_mock_data()


def make_memory(**overrides: Any):
    """Create a preference memory with sensible defaults."""
    params: dict[str, Any] = {
        "owner_key": "owner-a",
        "memory_type": "preference",
        "content": "Customer Budi prefers kopi susu tanpa gula.",
    }
    params.update(overrides)
    return create_memory(**params)


# ---------------------------------------------------------------------------
# A. CRUD
# ---------------------------------------------------------------------------


def test_create_and_get_memory() -> None:
    memory = make_memory()

    assert memory.memory_id >= 1
    assert memory.owner_key == "owner-a"
    assert memory.memory_type == "preference"
    assert memory.content == "Customer Budi prefers kopi susu tanpa gula."
    assert memory.created_at
    assert memory.updated_at
    fetched = get_memory(memory.memory_id, owner_key="owner-a")
    assert fetched == memory  # snapshot round-trips field-for-field


def test_list_memories_newest_first_and_type_filter() -> None:
    first = make_memory()
    second = make_memory(
        memory_type="business_context", content="Toko tutup hari Minggu."
    )

    listed = list_memories(owner_key="owner-a")
    assert [m.memory_id for m in listed] == [second.memory_id, first.memory_id]

    only_prefs = list_memories(owner_key="owner-a", memory_type="preference")
    assert [m.memory_id for m in only_prefs] == [first.memory_id]


def test_update_memory_content_and_type() -> None:
    memory = make_memory()

    updated = update_memory(
        memory.memory_id,
        owner_key="owner-a",
        content="Customer Budi prefers matcha latte.",
        memory_type="customer_context",
    )

    assert updated.memory_id == memory.memory_id
    assert updated.content == "Customer Budi prefers matcha latte."
    assert updated.memory_type == "customer_context"
    assert get_memory(memory.memory_id, owner_key="owner-a") == updated


def test_delete_memory() -> None:
    memory = make_memory()

    deleted = delete_memory(memory.memory_id, owner_key="owner-a")

    assert deleted.memory_id == memory.memory_id
    assert get_memory(memory.memory_id, owner_key="owner-a") is None
    with pytest.raises(MemoryNotFoundError):
        delete_memory(memory.memory_id, owner_key="owner-a")


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(MemoryValidationError):
        create_memory(owner_key="owner-a", memory_type="hobby", content="x")
    with pytest.raises(MemoryValidationError):
        create_memory(owner_key="", memory_type="preference", content="x")
    with pytest.raises(MemoryValidationError):
        create_memory(owner_key="owner-a", memory_type="preference", content="   ")
    with pytest.raises(MemoryValidationError):
        create_memory(
            owner_key="owner-a", memory_type="preference", content="x" * 2001
        )
    with pytest.raises(MemoryValidationError):
        update_memory(1, owner_key="owner-a")  # nothing to update


# ---------------------------------------------------------------------------
# B. Persistence across engine recreation
# ---------------------------------------------------------------------------


def test_memory_survives_engine_dispose_and_recreate() -> None:
    from app.db.database import dispose_engines

    memory = make_memory()

    dispose_engines()  # drop every pooled connection and cached engine

    fetched = get_memory(memory.memory_id, owner_key="owner-a")
    assert fetched == memory
    assert [m.memory_id for m in list_memories(owner_key="owner-a")] == [
        memory.memory_id
    ]


# ---------------------------------------------------------------------------
# C. Owner isolation
# ---------------------------------------------------------------------------


def test_owner_b_cannot_read_or_mutate_owner_a() -> None:
    memory = make_memory()

    # Reads: invisible to another owner.
    assert get_memory(memory.memory_id, owner_key="owner-b") is None
    assert list_memories(owner_key="owner-b") == []
    assert search_memories(owner_key="owner-b", query="Budi") == []

    # Mutations: refused with the same NotFound as a truly unknown id.
    with pytest.raises(MemoryNotFoundError):
        update_memory(memory.memory_id, owner_key="owner-b", content="hacked")
    with pytest.raises(MemoryNotFoundError):
        delete_memory(memory.memory_id, owner_key="owner-b")

    # The memory is untouched.
    assert get_memory(memory.memory_id, owner_key="owner-a") == memory


# ---------------------------------------------------------------------------
# D. Deterministic retrieval
# ---------------------------------------------------------------------------


def test_search_returns_relevant_memory_and_ignores_unrelated() -> None:
    budi = make_memory(content="Customer Budi prefers kopi susu tanpa gula.")
    make_memory(memory_type="business_context", content="Toko tutup hari Minggu.")
    make_memory(memory_type="instruction", content="Always answer in Indonesian.")

    hits = search_memories(owner_key="owner-a", query="apa preference Budi?")

    assert [m.memory_id for m in hits] == [budi.memory_id]
    # Deterministic: the same query returns the same result.
    again = search_memories(owner_key="owner-a", query="apa preference Budi?")
    assert again == hits


def test_search_scores_by_keyword_overlap_then_recency() -> None:
    weak = make_memory(content="Budi sometimes orders kopi susu.")
    strong = make_memory(content="Budi suka kopi susu tanpa gula setiap pagi.")

    hits = search_memories(owner_key="owner-a", query="Budi kopi susu gula")

    # More keyword overlap wins even though `weak` was inserted first.
    assert [m.memory_id for m in hits] == [strong.memory_id, weak.memory_id]


def test_search_without_query_returns_recent_memories() -> None:
    first = make_memory()
    second = make_memory(content="Second fact.")

    recent = search_memories(owner_key="owner-a", limit=1)

    assert [m.memory_id for m in recent] == [second.memory_id]
    assert first.memory_id != second.memory_id


def test_memory_type_filter_in_search() -> None:
    make_memory(content="Budi likes early delivery.")
    context = make_memory(
        memory_type="customer_context", content="Budi is a regular since 2024."
    )

    hits = search_memories(
        owner_key="owner-a", query="Budi", memory_type="customer_context"
    )

    assert [m.memory_id for m in hits] == [context.memory_id]


def test_context_block_is_empty_without_hits_and_bounded_with_hits() -> None:
    assert memory_context_block("owner-a", "Berapa stok kopi susu?") == ""

    make_memory(content="Budi prefers kopi susu tanpa gula.")
    block = memory_context_block("owner-a", "kopi susu untuk Budi")

    assert "Known facts" in block
    assert "Budi prefers kopi susu tanpa gula." in block
    assert "[preference]" in block
    # Bounded: at most CONTEXT_MEMORY_LIMIT entries.
    assert block.count("\n- [") <= 3


# ---------------------------------------------------------------------------
# E. Agent integration
# ---------------------------------------------------------------------------


@dataclass
class FakeGLMResponse:
    text: str = ""
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None


class FakeGLMClient:
    """Records the system prompt; returns queued responses."""

    def __init__(self, responses: list[FakeGLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete_with_tools(
        self, messages: list[dict[str, Any]], *, system: str, tools: list[dict[str, Any]]
    ) -> FakeGLMResponse:
        self.calls.append(
            {"messages": [dict(m) for m in messages], "system": system}
        )
        return self._responses.pop(0)


def make_agent(fake: FakeGLMClient) -> DibantuAgent:
    return DibantuAgent(
        Settings(glm_api_key="test-key", glm_base_url="https://x.test", glm_model="m"),
        client=fake,
    )


def test_agent_without_memories_gets_the_plain_system_prompt() -> None:
    fake = FakeGLMClient(
        [
            FakeGLMResponse(
                tool_name="check_stock", tool_input={"product_name": "Kopi Susu"}
            ),
            FakeGLMResponse(text="Stok Kopi Susu 24 unit."),
        ]
    )

    asyncio.run(
        make_agent(fake).run(
            AgentRequest(
                message="Berapa stok kopi susu?"
            )
        )
    )

    assert fake.calls[0]["system"] == AGENT_SYSTEM_PROMPT  # byte-identical


def test_agent_with_memory_gets_small_relevant_context() -> None:
    make_memory()
    fake = FakeGLMClient([FakeGLMResponse(text="Budi suka kopi susu tanpa gula.")])

    asyncio.run(
        make_agent(fake).run(
            AgentRequest(
                message="Kopi susu untuk Budi", owner_key="owner-a"
            )
        )
    )

    system = fake.calls[0]["system"]
    assert system.startswith(AGENT_SYSTEM_PROMPT)
    assert "Customer Budi prefers kopi susu tanpa gula." in system
    assert len(system) < len(AGENT_SYSTEM_PROMPT) + 600  # small context only


def test_memory_tool_inside_run_is_owner_scoped() -> None:
    fake = FakeGLMClient(
        [
            FakeGLMResponse(
                tool_name="save_memory",
                tool_input={"content": "Owner A likes morning delivery."},
            ),
            FakeGLMResponse(text="Tersimpan."),
        ]
    )

    asyncio.run(
        make_agent(fake).run(
            AgentRequest(
                message="Ingat: saya suka pengiriman pagi.", owner_key="owner-a"
            )
        )
    )

    stored = list_memories(owner_key="owner-a")
    assert len(stored) == 1
    assert stored[0].content == "Owner A likes morning delivery."
    assert list_memories(owner_key="owner-b") == []


def test_owner_scope_context_manager() -> None:
    assert current_owner_key() == DEFAULT_OWNER_KEY
    with owner_scope("wa-628123"):
        assert current_owner_key() == "wa-628123"
    assert current_owner_key() == DEFAULT_OWNER_KEY


# ---------------------------------------------------------------------------
# F. Memory safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        "My password is hunter2trustno1",
        "API key sk-abc123def456ghi789",
        "bearer token abc123",
        "-----BEGIN RSA PRIVATE KEY-----",
        "Card number 4111111111111111",
    ],
)
def test_secret_like_content_is_refused(content: str) -> None:
    with pytest.raises(MemoryValidationError, match="secret|card"):
        create_memory(owner_key="owner-a", memory_type="preference", content=content)


def test_save_memory_tool_returns_error_dict_for_secrets() -> None:
    from app.tools.registry import get_tool

    result = get_tool("save_memory")("my api key is abc123")

    assert result["success"] is False
    assert result["error"] == "SENSITIVE_CONTENT"


def test_memory_tools_never_touch_business_data_or_approvals() -> None:
    from app.tools.registry import get_tool
    from app.tools.business_tools import get_sales_report
    from app.approval import list_pending

    before = get_sales_report("monthly")

    get_tool("save_memory")("Reminder: reorder croissant every Friday.")
    saved = get_tool("search_memory")("croissant")
    deleted = get_tool("delete_memory")(saved["results"][0]["memory_id"])

    assert saved["success"] is True
    assert deleted["success"] is True
    # No business or approval side effects whatsoever.
    assert get_sales_report("monthly") == before
    assert list_pending() == []


# ---------------------------------------------------------------------------
# G. Restart-like behavior + API
# ---------------------------------------------------------------------------


def test_memories_survive_backend_restart_simulation() -> None:
    from app.db.database import dispose_engines

    memory = make_memory()
    dispose_engines()

    with TestClient(app) as client:  # fresh ASGI lifespan, fresh sessions
        body = client.get("/api/memory", params={"owner_key": "owner-a"}).json()

    assert body["count"] == 1
    assert body["memories"][0]["memory_id"] == memory.memory_id
    assert body["memories"][0]["content"] == memory.content


def test_api_create_list_and_delete_memory() -> None:
    with TestClient(app) as client:
        created = client.post(
            "/api/memory",
            json={
                "owner_key": "owner-a",
                "memory_type": "instruction",
                "content": "Always confirm orders twice.",
            },
        )
        assert created.status_code == 200
        memory_id = created.json()["memory_id"]

        listed = client.get(
            "/api/memory", params={"owner_key": "owner-a", "memory_type": "instruction"}
        ).json()
        assert listed["count"] == 1

        # Wrong owner: invisible AND undeletable (404, not 403 — no probing).
        assert (
            client.get("/api/memory", params={"owner_key": "owner-b"}).json()["count"]
            == 0
        )
        gone = client.delete(
            f"/api/memory/{memory_id}", params={"owner_key": "owner-b"}
        )
        assert gone.status_code == 404

        deleted = client.delete(
            f"/api/memory/{memory_id}", params={"owner_key": "owner-a"}
        )
        assert deleted.status_code == 200
        assert deleted.json()["memory_id"] == memory_id
        assert client.delete(
            f"/api/memory/{memory_id}", params={"owner_key": "owner-a"}
        ).status_code == 404


def test_api_rejects_invalid_payloads() -> None:
    with TestClient(app) as client:
        bad_type = client.post(
            "/api/memory",
            json={"owner_key": "o", "memory_type": "dream", "content": "x"},
        )
        assert bad_type.status_code == 422

        secret = client.post(
            "/api/memory",
            json={
                "owner_key": "o",
                "memory_type": "preference",
                "content": "my api key is abc123",
            },
        )
        assert secret.status_code == 422
        assert "secret" in secret.json()["detail"].lower()


def test_api_chat_owner_key_flows_to_memory_context() -> None:
    from app.api.routes import get_agent

    make_memory()
    fake = FakeGLMClient([FakeGLMResponse(text="Siap.")])
    app.dependency_overrides[get_agent] = lambda: make_agent(fake)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/chat",
                json={
                    "message": "Kopi susu untuk Budi",
                    "owner_key": "owner-a",
                },
            )
        assert response.status_code == 200
    finally:
        app.dependency_overrides.clear()

    assert "Customer Budi prefers kopi susu tanpa gula." in fake.calls[0]["system"]


def test_memory_types_are_the_documented_set() -> None:
    assert MEMORY_TYPES == (
        "preference",
        "customer_context",
        "business_context",
        "instruction",
    )
