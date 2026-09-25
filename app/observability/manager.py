"""Agent observability manager and trace recorder (Step 15).

Lightweight, PostgreSQL-only tracing for the AI agent — no external
observability platform. One ``agent_runs`` row per execution plus one
``agent_events`` row per traced operation (LLM calls, tool calls,
memory/RAG/approval operations), all tied together by a stable
``run_id`` that is returned to API callers.

Safety rules baked in:

- Metadata is small, safe JSON only: no prompts, no raw model
  responses, no chain-of-thought, no secrets, no stack traces
  (``error_type`` is an exception class name).
- ``TraceRecorder`` swallows every observability failure and logs a
  warning: tracing must NEVER break or slow down the business request
  (critical rule — observability is secondary to the operation).
- Token usage is stored only when the provider actually returned it;
  otherwise null (counts are never invented).
"""

from __future__ import annotations

import logging
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Any, Iterator

from app.approval import SENSITIVE_ACTIONS
from app.db.database import session_scope
from app.observability import repository

logger = logging.getLogger("dibantu.observability")

__all__ = [
    "EVENT_NAMES",
    "EVENT_TYPES",
    "RUN_STATUSES",
    "AgentEvent",
    "AgentRun",
    "TraceRecorder",
    "classify_tool",
    "complete_event",
    "complete_run",
    "fail_event",
    "fail_run",
    "get_run",
    "get_run_events",
    "list_runs",
    "new_run_id",
    "record_event",
    "sanitize_metadata",
    "start_event",
    "start_run",
    "trace_approval_decision",
]

#: Closed vocabularies (mirrored by the 0004 migration CHECKs).
EVENT_TYPES: tuple[str, ...] = ("RUN", "LLM", "TOOL", "MEMORY", "RAG", "APPROVAL")
RUN_STATUSES: tuple[str, ...] = ("running", "completed", "failed")
EVENT_STATUSES: tuple[str, ...] = ("started", "success", "failed")
EVENT_NAMES: tuple[str, ...] = (
    "RUN_STARTED",
    "RUN_COMPLETED",
    "RUN_FAILED",
    "LLM_CALL",
    "TOOL_CALL",
    "MEMORY_SEARCH",
    "MEMORY_SAVE",
    "MEMORY_DELETE",
    "RAG_SEARCH",
    "APPROVAL_REQUESTED",
    "APPROVAL_APPROVED",
    "APPROVAL_REJECTED",
)

#: How much of the user message is kept as the run's request preview.
REQUEST_PREVIEW_LENGTH = 160

#: Markers that make a request preview unsafe to persist at all (the
#: user may have pasted a secret into the chat); a matching preview is
#: stored as NULL rather than truncated.
_PREVIEW_SENSITIVE_MARKERS: tuple[str, ...] = (
    "api key",
    "api_key",
    "apikey",
    "password",
    "bearer ",
    "authorization",
    "token",
    "secret",
    "private key",
    "-----begin",
    "sk-",
)

_PREVIEW_CARD_RE = re.compile(r"\d{13,19}")


def _safe_preview(message: str | None) -> str | None:
    """Bounded preview of the user message, or None when unsafe."""
    text = (message or "").strip()
    if not text:
        return None
    lowered = text.casefold()
    if any(marker in lowered for marker in _PREVIEW_SENSITIVE_MARKERS):
        return None
    if _PREVIEW_CARD_RE.search(text):
        return None
    return text[:REQUEST_PREVIEW_LENGTH]

#: Keys metadata may carry; anything else is dropped (defense in depth
#: against accidental sensitive payloads).
_MAX_METADATA_KEYS = 12
_MAX_METADATA_CHARS = 400


class ObservabilityError(Exception):
    """Base class for observability errors (never raised at business code)."""


def new_run_id() -> str:
    """A fresh public run id (run-<16 hex chars>)."""
    return f"run-{uuid.uuid4().hex[:16]}"


def classify_tool(tool_name: str) -> tuple[str, str]:
    """Map a tool name to its (event_type, event_name) pair.

    Memory/RAG/approval operations get typed events without touching
    their business code: the classification happens purely at the
    agent's instrumentation point.
    """
    if tool_name == "search_memory":
        return ("MEMORY", "MEMORY_SEARCH")
    if tool_name == "save_memory":
        return ("MEMORY", "MEMORY_SAVE")
    if tool_name == "delete_memory":
        return ("MEMORY", "MEMORY_DELETE")
    if tool_name == "search_knowledge_base":
        return ("RAG", "RAG_SEARCH")
    if tool_name in SENSITIVE_ACTIONS:
        return ("APPROVAL", "APPROVAL_REQUESTED")
    return ("TOOL", "TOOL_CALL")


def sanitize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep metadata compact and safe: shallow, bounded copies only.

    JSON scalars (int/float/bool/None) keep their type — token counts
    stay numbers; everything else is stringified and bounded. This does
    not attempt deep content inspection (callers must not put sensitive
    values in); it bounds what a buggy caller could persist.
    """
    if not metadata:
        return None
    safe: dict[str, Any] = {}
    for key, value in list(metadata.items())[:_MAX_METADATA_KEYS]:
        if isinstance(value, bool) or value is None:
            safe[str(key)[:40]] = value
        elif isinstance(value, (int, float)):
            safe[str(key)[:40]] = value
        else:
            safe[str(key)[:40]] = str(value)[:_MAX_METADATA_CHARS]
    return safe or None


# ---------------------------------------------------------------------------
# Domain snapshots
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentRun:
    """One agent run row (immutable snapshot)."""

    run_id: str
    source: str
    owner_key: str | None
    status: str
    request_preview: str | None
    started_at: str
    completed_at: str | None
    duration_ms: int | None
    error_type: str | None
    event_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source": self.source,
            "owner_key": self.owner_key,
            "status": self.status,
            "request_preview": self.request_preview,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "error_type": self.error_type,
            "event_count": self.event_count,
        }


@dataclass(frozen=True)
class AgentEvent:
    """One traced operation (immutable snapshot)."""

    event_id: int
    run_id: str
    event_type: str
    event_name: str
    status: str
    iteration: int | None
    started_at: str
    completed_at: str | None
    duration_ms: int | None
    metadata: dict[str, Any] | None
    error_type: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "event_type": self.event_type,
            "event_name": self.event_name,
            "status": self.status,
            "iteration": self.iteration,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "error_type": self.error_type,
        }


# ---------------------------------------------------------------------------
# Service: run/event lifecycle (module-level, spec-complete)
# ---------------------------------------------------------------------------


def start_run(
    *,
    source: str,
    owner_key: str | None = None,
    request_preview: str | None = None,
) -> str:
    """Insert one running run row; returns its fresh run_id."""
    run_id = new_run_id()
    with session_scope() as session:
        repository.insert_run(
            session,
            run_id=run_id,
            source=source,
            owner_key=owner_key,
            request_preview=_safe_preview(request_preview),
            started_at=datetime.now(),
        )
        repository.insert_event(
            session,
            run_id=run_id,
            event_type="RUN",
            event_name="RUN_STARTED",
            status="success",
            iteration=None,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            duration_ms=0,
            metadata_json={"source": source},
            error_type=None,
        )
    return run_id


def complete_run(run_id: str, *, duration_ms: int) -> None:
    """Guarded UPDATE running -> completed (+ RUN_COMPLETED event)."""
    now = datetime.now()
    with session_scope() as session:
        changed = repository.complete_run(
            session, run_id, completed_at=now, duration_ms=duration_ms
        )
        if changed:
            repository.insert_event(
                session,
                run_id=run_id,
                event_type="RUN",
                event_name="RUN_COMPLETED",
                status="success",
                iteration=None,
                started_at=now,
                completed_at=now,
                duration_ms=0,
                metadata_json=None,
                error_type=None,
            )


def fail_run(run_id: str, *, error_type: str | None, duration_ms: int) -> None:
    """Guarded UPDATE running -> failed (+ RUN_FAILED event)."""
    now = datetime.now()
    with session_scope() as session:
        changed = repository.fail_run(
            session,
            run_id,
            completed_at=now,
            duration_ms=duration_ms,
            error_type=_bounded_error_type(error_type),
        )
        if changed:
            repository.insert_event(
                session,
                run_id=run_id,
                event_type="RUN",
                event_name="RUN_FAILED",
                status="failed",
                iteration=None,
                started_at=now,
                completed_at=now,
                duration_ms=0,
                metadata_json=None,
                error_type=(error_type or "")[:100] or None,
            )


def _bounded_error_type(error_type: str | None) -> str | None:
    """Exception class name only, bounded — never a message/traceback."""
    if not error_type:
        return None
    return error_type[:100] or None


def record_event(
    run_id: str,
    event_type: str,
    event_name: str,
    *,
    status: str = "success",
    iteration: int | None = None,
    duration_ms: int | None = None,
    metadata: dict[str, Any] | None = None,
    error_type: str | None = None,
) -> int:
    """Insert one already-measured event row; returns its id."""
    now = datetime.now()
    with session_scope() as session:
        record = repository.insert_event(
            session,
            run_id=run_id,
            event_type=event_type,
            event_name=event_name,
            status=status,
            iteration=iteration,
            started_at=now,
            completed_at=now,
            duration_ms=duration_ms,
            metadata_json=sanitize_metadata(metadata),
            error_type=_bounded_error_type(error_type),
        )
        return record.id


def start_event(
    run_id: str,
    event_type: str,
    event_name: str,
    *,
    iteration: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Insert a 'started' event row; complete it later by id."""
    with session_scope() as session:
        record = repository.insert_event(
            session,
            run_id=run_id,
            event_type=event_type,
            event_name=event_name,
            status="started",
            iteration=iteration,
            started_at=datetime.now(),
            completed_at=None,
            duration_ms=None,
            metadata_json=sanitize_metadata(metadata),
            error_type=None,
        )
        return record.id


def complete_event(
    event_id: int,
    *,
    duration_ms: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Close a started event as success."""
    with session_scope() as session:
        repository.update_event(
            session,
            event_id,
            status="success",
            completed_at=datetime.now(),
            duration_ms=duration_ms,
            metadata_json=sanitize_metadata(metadata),
        )


def fail_event(
    event_id: int, *, duration_ms: int, error_type: str | None = None
) -> None:
    """Close a started event as failed."""
    with session_scope() as session:
        repository.update_event(
            session,
            event_id,
            status="failed",
            completed_at=datetime.now(),
            duration_ms=duration_ms,
            error_type=_bounded_error_type(error_type),
        )


def get_run(run_id: str) -> AgentRun | None:
    """Return the run snapshot (any status), or None."""
    with session_scope() as session:
        record = repository.find_run(session, run_id)
        if record is None:
            return None
        count = len(repository.run_events(session, run_id, limit=1000))
        return _as_run(record, count)


def list_runs(
    *,
    status: str | None = None,
    source: str | None = None,
    owner_key: str | None = None,
    limit: int = 50,
) -> list[AgentRun]:
    """Recent runs, newest first, with event counts."""
    with session_scope() as session:
        pairs = repository.list_runs(
            session,
            status=status,
            source=source,
            owner_key=owner_key,
            limit=max(1, min(int(limit), 200)),
        )
        return [_as_run(record, count) for record, count in pairs]


def get_run_events(
    run_id: str, *, event_type: str | None = None, limit: int = 500
) -> list[AgentEvent] | None:
    """One run's events in order, or None when the run does not exist."""
    with session_scope() as session:
        if repository.find_run(session, run_id) is None:
            return None
        records = repository.run_events(
            session,
            run_id,
            event_type=event_type,
            limit=max(1, min(int(limit), 1000)),
        )
        return [_as_event(record) for record in records]


def trace_approval_decision(approval: Any, *, approved: bool) -> None:
    """Record an approval decision as its own micro run.

    Informational only: this observes the decision, it never executes
    or bypasses anything. Failure is swallowed (never breaks the
    approval API).
    """
    try:
        run_id = start_run(source="approval_api")
        record_event(
            run_id,
            "APPROVAL",
            "APPROVAL_APPROVED" if approved else "APPROVAL_REJECTED",
            duration_ms=0,
            metadata={
                "approval_id": getattr(approval, "approval_id", ""),
                "action": getattr(approval, "action", ""),
            },
        )
        complete_run(run_id, duration_ms=0)
    except Exception as exc:  # noqa: BLE001 - tracing must never break the API
        logger.warning(
            "observability: failed to trace approval decision (%s)",
            type(exc).__name__,
        )


def _as_run(record: Any, event_count: int = 0) -> AgentRun:
    return AgentRun(
        run_id=record.run_id,
        source=record.source,
        owner_key=record.owner_key,
        status=record.status,
        request_preview=record.request_preview,
        started_at=record.started_at.isoformat(timespec="seconds"),
        completed_at=(
            record.completed_at.isoformat(timespec="seconds")
            if record.completed_at is not None
            else None
        ),
        duration_ms=record.duration_ms,
        error_type=record.error_type,
        event_count=event_count,
    )


def _as_event(record: Any) -> AgentEvent:
    return AgentEvent(
        event_id=record.id,
        run_id=record.run_id,
        event_type=record.event_type,
        event_name=record.event_name,
        status=record.status,
        iteration=record.iteration,
        started_at=record.started_at.isoformat(timespec="seconds"),
        completed_at=(
            record.completed_at.isoformat(timespec="seconds")
            if record.completed_at is not None
            else None
        ),
        duration_ms=record.duration_ms,
        metadata=record.event_metadata,
        error_type=record.error_type,
    )


# ---------------------------------------------------------------------------
# TraceRecorder: the agent-facing interface
# ---------------------------------------------------------------------------


@dataclass
class TraceRecorder:
    """Per-run tracing handle used by the agent composition.

    Created before the agent runs; every method is best-effort and
    swallows its own failures with a warning log, so observability can
    never crash or replace a business error. ``model`` is carried so
    LLM events can name the model without the generic Agent knowing it.
    """

    source: str = "api"
    owner_key: str | None = None
    request_preview: str | None = None
    model: str | None = None
    run_id: str = field(default_factory=new_run_id)
    _ok: bool = field(default=True, repr=False)
    _started: float = field(default_factory=perf_counter, repr=False)

    def start(self) -> None:
        """Open the run row + RUN_STARTED event (best-effort)."""
        if not self._ok:
            return
        try:
            run_id = start_run(
                source=self.source,
                owner_key=self.owner_key,
                request_preview=self.request_preview,
            )
            self.run_id = run_id
        except Exception as exc:  # noqa: BLE001
            self._ok = False
            logger.warning(
                "observability: run tracing disabled after start failure (%s)",
                type(exc).__name__,
            )

    def record(
        self,
        event_type: str,
        event_name: str,
        *,
        status: str = "success",
        iteration: int | None = None,
        duration_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
        error_type: str | None = None,
    ) -> None:
        """Persist one completed event (best-effort)."""
        if not self._ok:
            return
        try:
            record_event(
                self.run_id,
                event_type,
                event_name,
                status=status,
                iteration=iteration,
                duration_ms=duration_ms,
                metadata=metadata,
                error_type=error_type,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "observability: event %s not recorded (%s)",
                event_name,
                type(exc).__name__,
            )

    def llm_call(
        self,
        *,
        iteration: int,
        duration_ms: int,
        usage: dict[str, int] | None,
        error_type: str | None = None,
    ) -> None:
        """Record one LLM call with safe metadata (never the payload)."""
        metadata: dict[str, Any] = {}
        if self.model:
            metadata["model"] = self.model
        if usage:
            # Only counts the provider actually returned — never invented.
            for key in ("input_tokens", "output_tokens"):
                if isinstance(usage.get(key), int):
                    metadata[key] = usage[key]
        self.record(
            "LLM",
            "LLM_CALL",
            iteration=iteration,
            duration_ms=duration_ms,
            metadata=metadata or None,
            error_type=error_type,
            status="failed" if error_type else "success",
        )

    def tool_call(
        self,
        tool_name: str,
        *,
        iteration: int | None,
        duration_ms: int,
        ok: bool,
        error_type: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record one tool execution with typed memory/RAG/approval names."""
        event_type, event_name = classify_tool(tool_name)
        metadata: dict[str, Any] = {"tool": tool_name}
        if extra_metadata:
            metadata.update(extra_metadata)
        self.record(
            event_type,
            event_name,
            iteration=iteration,
            duration_ms=duration_ms,
            metadata=metadata,
            error_type=error_type,
            status="success" if ok else "failed",
        )

    def complete(self) -> None:
        """Close the run as completed (best-effort)."""
        if not self._ok:
            return
        try:
            complete_run(
                self.run_id, duration_ms=int((perf_counter() - self._started) * 1000)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "observability: run completion not recorded (%s)",
                type(exc).__name__,
            )

    def fail(self, exc: BaseException | None) -> None:
        """Close the run as failed; the business error still propagates."""
        if not self._ok:
            return
        try:
            fail_run(
                self.run_id,
                error_type=type(exc).__name__ if exc is not None else None,
                duration_ms=int((perf_counter() - self._started) * 1000),
            )
        except Exception as trace_exc:  # noqa: BLE001
            logger.warning(
                "observability: run failure not recorded (%s)",
                type(trace_exc).__name__,
            )
