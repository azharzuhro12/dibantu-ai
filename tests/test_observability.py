"""Agent observability & tracing tests (Step 15).

Covers the PostgreSQL-backed tracing: run/event lifecycle (start,
complete, fail, durations), durability across engine recreation, agent
integration (runs + typed tool events in iteration order, LLM events
with usage passthrough), failure paths (failed tool event, failed run,
tracing failures never crash business logic), the read-only API with
filters and 404s, and the security properties (no secrets, prompts,
raw responses, or stack traces ever persisted).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.agent.agent import Agent, DibantuAgent
from app.config import Settings
from app.main import app
from app.observability import (
    EVENT_NAMES,
    TraceRecorder,
    classify_tool,
    complete_event,
    complete_run,
    fail_event,
    fail_run,
    get_run,
    get_run_events,
    list_runs,
    record_event,
    sanitize_metadata,
    start_event,
    start_run,
)
from app.tools.business_tools import reset_mock_data


@pytest.fixture(autouse=True)
def clean_stores(test_database):
    """Start every test with empty observability tables."""
    _reset_observability()
    reset_mock_data()
    yield
    _reset_observability()
    reset_mock_data()


def _reset_observability() -> None:
    from app.db.database import session_scope
    from app.db.models import AgentEventRecord, AgentRunRecord

    with session_scope() as session:
        session.query(AgentEventRecord).delete()
        session.query(AgentRunRecord).delete()


# ---------------------------------------------------------------------------
# Test doubles (same shape the approval/memory suites use)
# ---------------------------------------------------------------------------


@dataclass
class FakeGLMResponse:
    text: str = ""
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    usage: dict[str, int] | None = None


class FakeGLMClient:
    """Returns queued responses; records the system prompt per call."""

    def __init__(self, responses: list[FakeGLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete_with_tools(
        self, messages: list[dict[str, Any]], *, system: str, tools: list[dict[str, Any]]
    ) -> FakeGLMResponse:
        self.calls.append({"messages": [dict(m) for m in messages], "system": system})
        return self._responses.pop(0)


class ExplodingTool:
    """A registered-shape tool that always raises."""

    def __call__(self, **_: Any) -> Any:
        raise RuntimeError("boom: payment information 4111111111111111")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "ExplodingTool"


def make_agent(fake: FakeGLMClient) -> DibantuAgent:
    return DibantuAgent(
        Settings(glm_api_key="test-key", glm_base_url="https://x.test", glm_model="glm-test"),
        client=fake,
    )


# ---------------------------------------------------------------------------
# A. Run lifecycle
# ---------------------------------------------------------------------------


def test_run_lifecycle_start_complete_with_duration() -> None:
    run_id = start_run(source="api", owner_key="owner-a", request_preview="x" * 500)

    run = get_run(run_id)
    assert run is not None
    assert run.status == "running"
    assert run.source == "api"
    assert run.owner_key == "owner-a"
    assert len(run.request_preview) <= 160  # preview is bounded
    assert run.duration_ms is None

    complete_run(run_id, duration_ms=1234)

    run = get_run(run_id)
    assert run.status == "completed"
    assert run.duration_ms == 1234
    assert run.completed_at is not None


def test_run_lifecycle_fail_records_error_type_only() -> None:
    run_id = start_run(source="api")

    fail_run(run_id, error_type="GLMAuthError", duration_ms=55)

    run = get_run(run_id)
    assert run.status == "failed"
    assert run.error_type == "GLMAuthError"
    assert "Traceback" not in (run.error_type or "")


def test_run_completion_is_guarded_against_double_close() -> None:
    run_id = start_run(source="api")
    complete_run(run_id, duration_ms=1)
    complete_run(run_id, duration_ms=2)  # second close: guarded no-op
    run = get_run(run_id)
    assert run.status == "completed"
    assert run.duration_ms == 1  # first decision stands


def test_run_events_bookends() -> None:
    run_id = start_run(source="api")
    complete_run(run_id, duration_ms=5)

    names = [e.event_name for e in get_run_events(run_id)]
    assert names == ["RUN_STARTED", "RUN_COMPLETED"]


def test_failed_run_events_bookends() -> None:
    run_id = start_run(source="api")
    fail_run(run_id, error_type="ValueError", duration_ms=5)

    events = get_run_events(run_id)
    assert [e.event_name for e in events] == ["RUN_STARTED", "RUN_FAILED"]
    assert events[-1].status == "failed"
    assert events[-1].error_type == "ValueError"


# ---------------------------------------------------------------------------
# B. Event lifecycle (granular start/complete/fail service)
# ---------------------------------------------------------------------------


def test_event_lifecycle_start_complete() -> None:
    run_id = start_run(source="api")
    event_id = start_event(run_id, "TOOL", "TOOL_CALL", iteration=2, metadata={"tool": "check_stock"})

    event = [e for e in get_run_events(run_id) if e.event_id == event_id][0]
    assert event.status == "started"
    assert event.completed_at is None

    complete_event(event_id, duration_ms=42, metadata={"tool": "check_stock", "result": "success"})

    event = [e for e in get_run_events(run_id) if e.event_id == event_id][0]
    assert event.status == "success"
    assert event.duration_ms == 42
    assert event.metadata["tool"] == "check_stock"


def test_event_lifecycle_fail() -> None:
    run_id = start_run(source="api")
    event_id = start_event(run_id, "LLM", "LLM_CALL", iteration=1)

    fail_event(event_id, duration_ms=9, error_type="GLMTimeoutError")

    event = [e for e in get_run_events(run_id) if e.event_id == event_id][0]
    assert event.status == "failed"
    assert event.error_type == "GLMTimeoutError"
    assert "Traceback" not in (event.error_type or "")


def test_record_event_single_insert() -> None:
    run_id = start_run(source="api")

    event_id = record_event(
        run_id, "TOOL", "TOOL_CALL", iteration=1, duration_ms=7, metadata={"tool": "x"}
    )

    event = [e for e in get_run_events(run_id) if e.event_id == event_id][0]
    assert event.status == "success"
    assert event.duration_ms == 7
    assert event.started_at == event.completed_at  # already-closed event


# ---------------------------------------------------------------------------
# C. Persistence across engine recreation
# ---------------------------------------------------------------------------


def test_runs_and_events_survive_engine_dispose() -> None:
    from app.db.database import dispose_engines

    run_id = start_run(source="api", owner_key="owner-a")
    record_event(run_id, "TOOL", "TOOL_CALL", iteration=1, duration_ms=3, metadata={"tool": "y"})

    dispose_engines()

    run = get_run(run_id)
    assert run is not None and run.owner_key == "owner-a"
    events = get_run_events(run_id)
    assert [e.event_name for e in events] == ["RUN_STARTED", "TOOL_CALL"]


def test_runs_survive_backend_restart_simulation() -> None:
    from app.db.database import dispose_engines

    run_id = start_run(source="chat", owner_key="owner-a")
    complete_run(run_id, duration_ms=10)
    dispose_engines()

    with TestClient(app) as client:  # fresh ASGI lifespan
        body = client.get("/api/observability/runs", params={"owner_key": "owner-a"}).json()

    assert body["count"] == 1
    assert body["runs"][0]["run_id"] == run_id


# ---------------------------------------------------------------------------
# D. Agent integration
# ---------------------------------------------------------------------------


def test_normal_agent_request_creates_completed_run_with_events() -> None:
    fake = FakeGLMClient(
        [
            FakeGLMResponse(
                tool_name="check_stock", tool_input={"product_name": "Kopi Susu"}
            ),
            FakeGLMResponse(text="Stok Kopi Susu 24 unit.", usage={"input_tokens": 10, "output_tokens": 5}),
        ]
    )

    result = asyncio.run(make_agent(fake).run(_request("Berapa stok kopi susu?")))

    assert result.run_id and result.run_id.startswith("run-")
    run = get_run(result.run_id)
    assert run.status == "completed"
    assert run.source == "api"
    assert run.duration_ms is not None

    events = get_run_events(result.run_id)
    names = [(e.event_type, e.event_name) for e in events]
    assert names == [
        ("RUN", "RUN_STARTED"),
        ("LLM", "LLM_CALL"),
        ("TOOL", "TOOL_CALL"),
        ("LLM", "LLM_CALL"),
        ("RUN", "RUN_COMPLETED"),
    ]
    tool_event = events[2]
    assert tool_event.metadata["tool"] == "check_stock"
    assert tool_event.iteration == 1
    assert tool_event.status == "success"
    # Usage passthrough: only the counts the fake actually returned.
    llm_final = events[3]
    assert llm_final.metadata["input_tokens"] == 10
    assert llm_final.metadata["output_tokens"] == 5
    assert llm_final.metadata["model"] == "glm-test"


def test_multi_step_workflow_preserves_iteration_sequence() -> None:
    fake = FakeGLMClient(
        [
            FakeGLMResponse(
                tool_name="search_memory", tool_input={"query": "Budi"}
            ),
            FakeGLMResponse(
                tool_name="check_stock", tool_input={"product_name": "Kopi Susu"}
            ),
            FakeGLMResponse(text="Selesai."),
        ]
    )

    result = asyncio.run(make_agent(fake).run(_request("Pesan untuk Budi")))

    events = get_run_events(result.run_id)
    typed = [(e.event_type, e.event_name, e.iteration) for e in events]
    assert typed == [
        ("RUN", "RUN_STARTED", None),
        ("LLM", "LLM_CALL", 1),
        ("MEMORY", "MEMORY_SEARCH", 1),
        ("LLM", "LLM_CALL", 2),
        ("TOOL", "TOOL_CALL", 2),
        ("LLM", "LLM_CALL", 3),
        ("RUN", "RUN_COMPLETED", None),
    ]
    memory_event = events[2]
    assert memory_event.metadata["tool"] == "search_memory"


def test_sensitive_action_traced_as_approval_requested() -> None:
    from app.approval import list_pending, reset_approvals

    reset_approvals()
    fake = FakeGLMClient(
        [
            FakeGLMResponse(
                tool_name="refund_order",
                tool_input={"order_id": "ORD-0001", "amount": 36000},
            ),
            FakeGLMResponse(text="Menunggu persetujuan."),
        ]
    )

    result = asyncio.run(make_agent(fake).run(_request("Refund ORD-0001")))

    events = get_run_events(result.run_id)
    approval_event = [e for e in events if e.event_name == "APPROVAL_REQUESTED"][0]
    assert approval_event.event_type == "APPROVAL"
    assert approval_event.metadata["approval_id"] == list_pending()[0].approval_id
    # The payload was never stored in the trace.
    assert "amount" not in json.dumps(approval_event.metadata)
    assert "ORD-0001" not in json.dumps(approval_event.metadata)


def test_classify_tool_mapping() -> None:
    assert classify_tool("search_memory") == ("MEMORY", "MEMORY_SEARCH")
    assert classify_tool("save_memory") == ("MEMORY", "MEMORY_SAVE")
    assert classify_tool("delete_memory") == ("MEMORY", "MEMORY_DELETE")
    assert classify_tool("search_knowledge_base") == ("RAG", "RAG_SEARCH")
    assert classify_tool("refund_order") == ("APPROVAL", "APPROVAL_REQUESTED")
    assert classify_tool("check_stock") == ("TOOL", "TOOL_CALL")
    assert classify_tool("create_order") == ("TOOL", "TOOL_CALL")


# ---------------------------------------------------------------------------
# E. Failure handling
# ---------------------------------------------------------------------------


def test_failed_tool_creates_failed_event_but_run_completes() -> None:
    agent = make_agent(
        FakeGLMClient(
            [
                FakeGLMResponse(tool_name="explode", tool_input={}),
                FakeGLMResponse(text="Tool-nya gagal."),
            ]
        )
    )
    agent._agent._tools = {"explode": ExplodingTool()}

    result = asyncio.run(agent.run(_request("Panggil tool meledak")))

    events = get_run_events(result.run_id)
    failed = [e for e in events if e.status == "failed"][0]
    assert failed.event_name == "TOOL_CALL"
    assert failed.error_type == "RuntimeError"
    assert "boom" not in json.dumps(failed.metadata)  # no exception message
    assert get_run(result.run_id).status == "completed"  # agent survived


def test_agent_failure_marks_run_failed_and_propagates() -> None:
    class FailingGLM:
        async def complete_with_tools(self, messages, *, system, tools):
            raise RuntimeError("provider down")

    agent = make_agent(FailingGLM())

    with pytest.raises(RuntimeError):
        asyncio.run(agent.run(_request("Halo")))

    runs = list_runs(source="api")
    assert len(runs) == 1
    assert runs[0].status == "failed"
    assert runs[0].error_type == "RuntimeError"
    events = get_run_events(runs[0].run_id)
    assert [e.event_name for e in events] == ["RUN_STARTED", "LLM_CALL", "RUN_FAILED"]
    assert events[1].status == "failed"


def test_tracing_failure_never_crashes_business_logic(monkeypatch) -> None:
    def broken_start_run(**_: Any) -> str:
        raise RuntimeError("observability db gone")

    monkeypatch.setattr("app.observability.manager.start_run", broken_start_run)

    fake = FakeGLMClient(
        [
            FakeGLMResponse(tool_name="check_stock", tool_input={"product_name": "Kopi Susu"}),
            FakeGLMResponse(text="Stok 24."),
        ]
    )

    result = asyncio.run(make_agent(fake).run(_request("Stok kopi susu?")))  # no raise

    assert result.reply == "Stok 24."


def test_tracing_event_failure_never_crashes(monkeypatch) -> None:
    def broken_record_event(*_: Any, **__: Any) -> int:
        raise RuntimeError("event insert failed")

    monkeypatch.setattr("app.observability.manager.record_event", broken_record_event)

    fake = FakeGLMClient(
        [
            FakeGLMResponse(tool_name="check_stock", tool_input={"product_name": "Kopi Susu"}),
            FakeGLMResponse(text="Stok 24."),
        ]
    )

    result = asyncio.run(make_agent(fake).run(_request("Stok?")))

    assert result.reply == "Stok 24."
    assert get_run(result.run_id).status == "completed"


# ---------------------------------------------------------------------------
# F. Read-only API
# ---------------------------------------------------------------------------


def test_api_lists_filters_and_returns_runs() -> None:
    run_a = start_run(source="chat", owner_key="owner-a")
    complete_run(run_a, duration_ms=10)
    run_b = start_run(source="webhook_whatsapp", owner_key="owner-b")
    complete_run(run_b, duration_ms=20)
    start_run(source="chat", owner_key="owner-c")  # left running

    with TestClient(app) as client:
        all_runs = client.get("/api/observability/runs").json()
        assert all_runs["count"] == 3

        only_chat = client.get(
            "/api/observability/runs", params={"source": "chat"}
        ).json()
        assert only_chat["count"] == 2

        only_completed = client.get(
            "/api/observability/runs", params={"status": "completed"}
        ).json()
        assert only_completed["count"] == 2

        scoped = client.get(
            "/api/observability/runs", params={"owner_key": "owner-b"}
        ).json()
        assert [r["run_id"] for r in scoped["runs"]] == [run_b]

        limited = client.get(
            "/api/observability/runs", params={"limit": 2}
        ).json()
        assert limited["count"] == 2

        invalid_status = client.get(
            "/api/observability/runs", params={"status": "exploded"}
        )
        assert invalid_status.status_code == 422


def test_api_run_detail_and_events_and_404() -> None:
    run_id = start_run(source="chat", owner_key="owner-a")
    record_event(run_id, "MEMORY", "MEMORY_SEARCH", iteration=1, duration_ms=2, metadata={"tool": "search_memory"})
    record_event(run_id, "TOOL", "TOOL_CALL", iteration=2, duration_ms=3, metadata={"tool": "check_stock"})
    complete_run(run_id, duration_ms=30)

    with TestClient(app) as client:
        detail = client.get(f"/api/observability/runs/{run_id}").json()
        assert detail["run_id"] == run_id
        assert detail["event_count"] == 4  # START + 2 events + COMPLETED

        events = client.get(f"/api/observability/runs/{run_id}/events").json()
        assert events["count"] == 4
        assert [e["event_name"] for e in events["events"]] == [
            "RUN_STARTED", "MEMORY_SEARCH", "TOOL_CALL", "RUN_COMPLETED",
        ]

        only_tool = client.get(
            f"/api/observability/runs/{run_id}/events", params={"event_type": "TOOL"}
        ).json()
        assert [e["event_name"] for e in only_tool["events"]] == ["TOOL_CALL"]

        assert client.get("/api/observability/runs/run-nope").status_code == 404
        assert (
            client.get("/api/observability/runs/run-nope/events").status_code == 404
        )


def test_api_chat_returns_run_id_and_events_are_queryable() -> None:
    from app.api.routes import get_agent

    fake = FakeGLMClient(
        [
            FakeGLMResponse(tool_name="check_stock", tool_input={"product_name": "Kopi Susu"}),
            FakeGLMResponse(text="Stok 24."),
        ]
    )
    app.dependency_overrides[get_agent] = lambda: make_agent(fake)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/chat", json={"message": "Berapa stok kopi susu?", "owner_key": "owner-a"}
            )
        assert response.status_code == 200
        run_id = response.json()["run_id"]

        with TestClient(app) as client:
            events = client.get(f"/api/observability/runs/{run_id}/events").json()
        assert events["count"] == 5
        assert events["events"][1]["event_name"] == "LLM_CALL"
    finally:
        app.dependency_overrides.clear()


def test_approval_decision_creates_its_own_micro_run() -> None:
    from app.approval import create_pending_approval, reset_approvals

    reset_approvals()
    approval = create_pending_approval(
        action="refund_order", requested_by="agent", payload={"order_id": "ORD-0001"}
    )

    # Tracing is wired at the API layer, so decide through the endpoint.
    with TestClient(app) as client:
        response = client.post(f"/api/approvals/{approval.approval_id}/approve")
    assert response.status_code == 200

    decision_runs = list_runs(source="approval_api")
    assert len(decision_runs) == 1
    assert decision_runs[0].status == "completed"
    events = get_run_events(decision_runs[0].run_id)
    assert [e.event_name for e in events] == [
        "RUN_STARTED", "APPROVAL_APPROVED", "RUN_COMPLETED",
    ]
    assert events[1].metadata["approval_id"] == approval.approval_id
    reset_approvals()


# ---------------------------------------------------------------------------
# G. Security: nothing sensitive is persisted
# ---------------------------------------------------------------------------


def test_no_secrets_prompts_or_raw_responses_in_records() -> None:
    fake = FakeGLMClient(
        [
            FakeGLMResponse(tool_name="check_stock", tool_input={"product_name": "Kopi Susu"}),
            FakeGLMResponse(text="Jawaban rahasia 4111111111111111."),
        ]
    )

    result = asyncio.run(make_agent(fake).run(_request("Stok? api_key=sk-SUPERSECRET123")))

    run = get_run(result.run_id)
    events = get_run_events(result.run_id)
    blob = json.dumps([run.as_dict()] + [e.as_dict() for e in events])
    for forbidden in (
        "sk-SUPERSECRET123",
        "Authorization",
        "Bearer ",
        "password",
        "Jawaban rahasia",  # no model replies
        "Traceback",  # no stack traces
        "Stok?",  # no full prompts (preview may hold a prefix, bounded)
    ):
        assert forbidden not in blob, forbidden


def test_request_preview_is_bounded_and_screened() -> None:
    run_id = start_run(source="api", request_preview="A" * 10_000)
    assert len(get_run(run_id).request_preview) == 160

    secret_run = start_run(
        source="api", request_preview="my api key is sk-SUPERSECRET123"
    )
    assert get_run(secret_run).request_preview is None  # screened out

    card_run = start_run(source="api", request_preview="kartu 4111111111111111")
    assert get_run(card_run).request_preview is None


def test_sanitize_metadata_bounds_everything() -> None:
    sanitized = sanitize_metadata(
        {"k" * 100: "v" * 10_000, "second": {"nested": "dict"}}
    )
    assert sanitized is not None
    assert all(len(str(v)) <= 400 for v in sanitized.values())
    assert all(len(k) <= 40 for k in sanitized)
    assert sanitize_metadata(None) is None
    assert sanitize_metadata({}) is None


def test_usage_never_invented() -> None:
    fake = FakeGLMClient(
        [FakeGLMResponse(text="Halo.")]  # no usage on the fake
    )

    result = asyncio.run(make_agent(fake).run(_request("Hai")))

    llm_events = [
        e for e in get_run_events(result.run_id) if e.event_name == "LLM_CALL"
    ]
    assert llm_events
    for event in llm_events:
        assert "input_tokens" not in (event.metadata or {})
        assert "output_tokens" not in (event.metadata or {})


def test_event_names_vocabulary_is_closed() -> None:
    assert "LLM_CALL" in EVENT_NAMES
    assert "MEMORY_SEARCH" in EVENT_NAMES
    assert "RAG_SEARCH" in EVENT_NAMES
    assert "APPROVAL_REQUESTED" in EVENT_NAMES
    assert set(EVENT_NAMES) == {
        "RUN_STARTED", "RUN_COMPLETED", "RUN_FAILED",
        "LLM_CALL", "TOOL_CALL",
        "MEMORY_SEARCH", "MEMORY_SAVE", "MEMORY_DELETE",
        "RAG_SEARCH",
        "APPROVAL_REQUESTED", "APPROVAL_APPROVED", "APPROVAL_REJECTED",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _request(message: str, **overrides: Any):
    from app.models.schemas import AgentRequest

    return AgentRequest(message=message, **overrides)
