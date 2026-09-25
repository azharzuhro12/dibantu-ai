"""Agent observability package for DibantuAI (Step 15).

Lightweight, PostgreSQL-only tracing — no external observability
platform. ``agent_runs`` + ``agent_events`` rows are tied together by a
stable ``run_id`` returned to API callers. Every record is safe by
construction: compact metadata, no prompts, no raw responses, no
chain-of-thought, no secrets, no stack traces; and tracing failures are
swallowed so they can never break a business request.
"""

from .manager import (
    EVENT_NAMES,
    EVENT_TYPES,
    RUN_STATUSES,
    AgentEvent,
    AgentRun,
    TraceRecorder,
    classify_tool,
    complete_event,
    complete_run,
    fail_event,
    fail_run,
    get_run,
    get_run_events,
    list_runs,
    new_run_id,
    record_event,
    sanitize_metadata,
    start_event,
    start_run,
    trace_approval_decision,
)
from .repository import (
    complete_run as complete_run_row
)  # noqa: F401 - re-exported for advanced callers

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
    "complete_run_row",
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
