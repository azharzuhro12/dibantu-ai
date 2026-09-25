"""Deterministic evaluator for the DibantuAI agent (Step 6 → Step 12).

Replays each dataset case through the real ``Agent`` loop driven by a
``ScriptedGLMClient`` (no network, no API key) while the real registry
business tools run against the freshly seeded mock store. Per case it
checks five simple, explainable properties:

1. Tool selection   -- the executed tool set equals the expected set.
2. Tool call count  -- the number of executions is within the expected
   (min, max) range.
3. Task completion  -- the mock store reached the expected final state
   (product stocks and 30-day order count, read via public tools).
4. Tool error handling -- for error cases, the failed/unknown tool came
   back to the model as an ``{"error": ...}`` tool_result and the run
   still completed.
5. Response grounding -- every grounding fact appears in the final reply
   AND in a tool result the agent actually received, so replies cannot
   cite data the tools never returned.

Since Step 12, knowledge_* cases run the real ``search_knowledge_base``
tool against an isolated scratch vector store (deterministic hashing
embeddings, temporary directory) — mirroring the scratch-database
isolation, the developer's real ``data/rag`` store is never touched and
the run stays offline.

Metrics are plain rates over these checks. There is deliberately no
weighted overall score.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any

from app.agent.agent import Agent
from app.agent.glm_client import GLMResponse, ToolUse
from app.tools.business_tools import check_stock, get_sales_report, reset_mock_data
from .dataset import (
    APPROVAL_CATEGORIES,
    MEMORY_CATEGORIES,
    OBSERVABILITY_CATEGORIES,
    RAG_CATEGORIES,
    ApprovalFlow,
    EvalCase,
    GLMTurn,
    StateExpectation,
    load_cases,
)

__all__ = [
    "CaseResult",
    "EvaluationReport",
    "GLMScriptError",
    "ScriptedGLMClient",
    "format_report",
    "main",
    "run_case",
    "run_evaluation",
]


class GLMScriptError(RuntimeError):
    """Raised when a scripted trajectory runs out of GLM turns."""


class ScriptedGLMClient:
    """Deterministic GLM stand-in used by the evaluator.

    ``complete_with_tools`` returns the next scripted turn and, from the
    incoming conversation, records which tools the agent actually
    executed (echoed assistant ``tool_use`` blocks) plus every tool
    result fed back to the model. Executions are deduplicated by
    tool_use id because each request replays the whole conversation.
    Those recordings are the evidence behind the tool-selection, count,
    grounding, and error checks.
    """

    def __init__(self, turns: list[GLMTurn] | tuple[GLMTurn, ...]) -> None:
        self._turns = list(turns)
        self.executed_tools: list[str] = []
        self.tool_results: list[str] = []
        self.requests = 0
        self._id_counter = 0
        self._seen_tool_use_ids: set[str] = set()

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str,
        tools: list[dict[str, Any]],
    ) -> GLMResponse:
        self.requests += 1
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                kind = block.get("type")
                if kind == "tool_use":
                    block_id = str(block.get("id"))
                    if block_id not in self._seen_tool_use_ids:
                        self._seen_tool_use_ids.add(block_id)
                        self.executed_tools.append(block["name"])
                elif kind == "tool_result":
                    self.tool_results.append(str(block.get("content", "")))
        if not self._turns:
            raise GLMScriptError(
                "scripted GLM turns exhausted before the agent stopped "
                "calling tools"
            )
        turn = self._turns.pop(0)
        tool_uses = []
        for call in turn.tool_calls:
            tool_uses.append(
                ToolUse(
                    name=call.name,
                    input=dict(call.input),
                    id=f"eval_{call.name}_{self._id_counter}",
                )
            )
            self._id_counter += 1
        first = tool_uses[0] if tool_uses else None
        return GLMResponse(
            text=turn.text,
            tool_name=first.name if first else None,
            tool_input=first.input if first else None,
            stop_reason="tool_use" if tool_uses else "end_turn",
            tool_uses=tuple(tool_uses),
        )


@dataclass(frozen=True)
class CaseResult:
    """Outcome of one evaluation case with per-check booleans."""

    case_id: str
    category: str
    passed: bool
    tool_selection_ok: bool
    tool_count_ok: bool
    completion_ok: bool
    grounding_ok: bool
    error_handling_ok: bool | None  # None when the case expects no tool error
    tools_executed: tuple[str, ...]
    reply: str
    failures: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate metrics over a set of case results (no overall score)."""

    results: tuple[CaseResult, ...]
    total: int
    passed: int
    failed: int
    pass_rate: float
    tool_selection_accuracy: float
    tool_count_accuracy: float
    task_completion_rate: float
    tool_error_handling_rate: float | None  # None when no error cases
    response_grounding_rate: float


def run_case(case: EvalCase) -> CaseResult:
    """Replay one case through the real Agent loop and check it.

    The mock store, the memory store, and the approval store are
    reseeded before the case (isolation from whatever ran earlier; the
    case's ``memories`` rows are seeded into the scratch memory store)
    and reset after it (the evaluator must not leak mutations into
    other test modules sharing the process).
    """
    from app.approval import reset_approvals
    from app.memory import create_memory, reset_memories

    reset_mock_data()
    reset_memories()
    reset_approvals()
    for owner_key, memory_type, content in case.memories:
        create_memory(
            owner_key=owner_key, memory_type=memory_type, content=content
        )
    try:
        return _run_case(case)
    finally:
        reset_mock_data()
        reset_memories()
        reset_approvals()


def _run_case(case: EvalCase) -> CaseResult:
    """Execute one already-isolated case; see ``run_case``."""
    client = ScriptedGLMClient(case.glm_turns)
    agent = Agent(client)  # real registry tools and schemas
    failures: list[str] = []
    error_handling_ok: bool | None = None
    run_ok = True

    try:
        reply = asyncio.run(agent.run(case.user_message))
    except Exception as exc:  # noqa: BLE001 - any crash fails the case
        reply = ""
        run_ok = False
        failures.append(f"agent run raised {type(exc).__name__}: {exc}")
        if case.expects_tool_error:
            error_handling_ok = False
    else:
        if case.expects_tool_error:
            error_reported = any('"error"' in r for r in client.tool_results)
            error_handling_ok = error_reported and bool(reply)
            if not error_reported:
                failures.append(
                    "tool error was not reported back as a tool_result"
                )

    executed = tuple(client.executed_tools)

    # 1. Tool selection accuracy.
    tool_selection_ok = set(executed) == set(case.expected_tools)
    if not tool_selection_ok:
        failures.append(
            f"tool selection: executed {sorted(set(executed))}, "
            f"expected {sorted(set(case.expected_tools))}"
        )

    # 2. Tool call count accuracy.
    minimum, maximum = case.expected_tool_count
    tool_count_ok = minimum <= len(executed) <= maximum
    if not tool_count_ok:
        failures.append(
            f"tool count: executed {len(executed)} calls, "
            f"expected between {minimum} and {maximum}"
        )

    # 3. Task completion: the scripted approval lifecycle first (its
    #    executions change business data), THEN the expected final
    #    state via public tools.
    state_failures: list[str] = []
    if case.approval_flow is not None:
        state_failures.extend(_run_approval_flow(case.approval_flow))
    state_failures.extend(_check_state(case.expected_state))
    completion_ok = not state_failures
    failures.extend(state_failures)

    # 4. Response grounding.
    if not reply:
        grounding_ok = False
        if run_ok:
            failures.append("agent returned an empty reply")
    else:
        missing_in_reply = [f for f in case.grounding_facts if f not in reply]
        missing_in_results = [
            f
            for f in case.grounding_facts
            if not any(f in result for result in client.tool_results)
        ]
        grounding_ok = not missing_in_reply and not missing_in_results
        if missing_in_reply:
            failures.append(f"grounding: facts missing from reply: {missing_in_reply}")
        if missing_in_results:
            failures.append(
                f"grounding: facts absent from every tool result: "
                f"{missing_in_results}"
            )

    passed = (
        run_ok
        and tool_selection_ok
        and tool_count_ok
        and completion_ok
        and grounding_ok
        and error_handling_ok is not False
    )
    return CaseResult(
        case_id=case.id,
        category=case.category,
        passed=passed,
        tool_selection_ok=tool_selection_ok,
        tool_count_ok=tool_count_ok,
        completion_ok=completion_ok,
        grounding_ok=grounding_ok,
        error_handling_ok=error_handling_ok,
        tools_executed=executed,
        reply=reply,
        failures=tuple(failures),
    )


def _run_approval_flow(flow: ApprovalFlow) -> list[str]:
    """Drive one scripted approval lifecycle against the real stack.

    Runs after the agent turn: finds the approval the agent created (or
    proves none was created), applies the human steps through the real
    manager/executor (same code the API uses, against the scratch
    database), and checks the terminal status, the recorded result, and
    — when requested — the observability trace contract. Business
    effects are checked separately via the case's ``expected_state``.
    """
    from app.approval import (
        ApprovalConflictError,
        approve,
        get_approval,
        list_approvals,
        reject,
    )
    from app.approval.executor import execute_approval

    created = [a for a in list_approvals(["pending"]) if a.action == flow.action]
    if not flow.expect_created:
        if created or list_approvals():
            return ["approval flow: an approval was created, but none was expected"]
        return []
    if not created:
        return [f"approval flow: no pending '{flow.action}' approval was created"]
    approval = created[0]
    failures: list[str] = []

    for step in flow.steps:
        if step == "approve":
            approve(approval.approval_id)
        elif step == "reject":
            reject(approval.approval_id)
        elif step == "execute":
            finished = execute_approval(approval.approval_id)
            if (
                flow.expect_result_success is not None
                and finished.status == "executed"
            ):
                result_success = (finished.execution_result or {}).get("success")
                if result_success is not flow.expect_result_success:
                    failures.append(
                        "approval flow: execution result success="
                        f"{result_success}, expected {flow.expect_result_success}"
                    )
        elif step == "execute_conflict":
            try:
                execute_approval(approval.approval_id)
            except ApprovalConflictError:
                continue
            failures.append(
                "approval flow: duplicate execution did not raise a conflict"
            )
        elif step == "verify_persisted":
            from app.db.database import dispose_engines

            before = get_approval(approval.approval_id)
            dispose_engines()  # simulated restart: only PostgreSQL knows
            after = get_approval(approval.approval_id)
            if after is None or before is None or after.status != before.status:
                failures.append(
                    "approval flow: terminal state did not survive the "
                    "simulated restart"
                )
        else:  # pragma: no cover - dataset is hand-maintained
            failures.append(f"approval flow: unknown step '{step}'")

    final = get_approval(approval.approval_id)
    if final is None or final.status != flow.final_status:
        actual = final.status if final is not None else "missing"
        failures.append(
            f"approval flow: final status is '{actual}', "
            f"expected '{flow.final_status}'"
        )

    if flow.expect_trace_event is not None:
        failures.extend(_check_execution_trace(approval.approval_id, flow))

    return failures


def _check_execution_trace(
    approval_id: str, flow: ApprovalFlow
) -> list[str]:
    """Verify the executor observability contract for one approval.

    The trace run must carry APPROVAL_EXECUTING followed by the expected
    terminal APPROVAL event, with compact metadata (approval id, action,
    phase) and never the payload itself.
    """
    from app.observability import get_run_events, list_runs

    runs = list_runs(source="approval_executor", limit=50)
    events = []
    for run in runs:
        for event in get_run_events(run.run_id, event_type="APPROVAL") or []:
            if (event.metadata or {}).get("approval_id") == approval_id:
                events.append(event)
    names = [event.event_name for event in events]
    if names != ["APPROVAL_EXECUTING", flow.expect_trace_event]:
        return [
            f"observability: approval events are {names}, expected "
            f"['APPROVAL_EXECUTING', '{flow.expect_trace_event}']"
        ]
    for event in events:
        if not set(event.metadata or {}) <= {"approval_id", "action", "status"}:
            return [
                "observability: approval event metadata carries unexpected keys"
            ]
    return []


def _setup_scratch_rag_store() -> str:
    """Ingest the knowledge docs into a throwaway vector store.

    Deterministic hashing embeddings keep the run offline and free (no
    model download); the temporary directory is returned so the caller
    can remove it afterwards. Installed via ``set_rag_service`` so the
    real ``search_knowledge_base`` tool resolves to this store.
    """
    from app.config import get_settings
    from app.rag.service import RagService, set_rag_service

    settings = get_settings()
    scratch_dir = tempfile.mkdtemp(prefix="dibantu_rag_eval_")
    service = RagService(
        knowledge_dir=settings.knowledge_dir,
        store_dir=scratch_dir,
        embeddings_provider="hashing",
    )
    service.ingest()
    set_rag_service(service)
    return scratch_dir


def _teardown_scratch_rag_store(scratch_dir: str) -> None:
    """Restore settings-driven RAG construction and delete the scratch."""
    from app.rag.service import reset_rag_service

    reset_rag_service()
    shutil.rmtree(scratch_dir, ignore_errors=True)


def _check_state(expected: StateExpectation) -> list[str]:
    """Compare the mock store against the expected final state."""
    failures: list[str] = []
    for product, wanted in expected.stock.items():
        actual = check_stock(product)
        if not actual.get("success"):
            failures.append(f"state: product '{product}' no longer exists")
        elif actual["stock"] != wanted:
            failures.append(
                f"state: {product} stock is {actual['stock']}, expected {wanted}"
            )
    report = get_sales_report("monthly")
    if report["total_orders"] != expected.monthly_orders:
        failures.append(
            f"state: monthly orders is {report['total_orders']}, "
            f"expected {expected.monthly_orders}"
        )
    return failures


def run_evaluation(
    cases: list[EvalCase] | tuple[EvalCase, ...] | None = None,
) -> EvaluationReport:
    """Evaluate every case (default: the full dataset) and aggregate.

    Knowledge_* cases need the RAG stack: before running them, a scratch
    vector store (hashing embeddings, temporary directory) is ingested
    from the configured knowledge directory and installed as the shared
    service — the real ``data/rag`` store is never touched — and torn
    down afterwards.
    """
    evaluated = list(load_cases() if cases is None else cases)
    scratch_dir = (
        _setup_scratch_rag_store()
        if any(case.category in RAG_CATEGORIES for case in evaluated)
        else None
    )
    try:
        results = tuple(run_case(case) for case in evaluated)
    finally:
        if scratch_dir is not None:
            _teardown_scratch_rag_store(scratch_dir)
    total = len(results)

    def percent(count: int) -> float:
        return round(100 * count / total, 1) if total else 0.0

    error_results = [r for r in results if r.error_handling_ok is not None]
    if error_results:
        handled = sum(1 for r in error_results if r.error_handling_ok)
        error_rate: float | None = round(100 * handled / len(error_results), 1)
    else:
        error_rate = None
    return EvaluationReport(
        results=results,
        total=total,
        passed=sum(1 for r in results if r.passed),
        failed=sum(1 for r in results if not r.passed),
        pass_rate=percent(sum(1 for r in results if r.passed)),
        tool_selection_accuracy=percent(sum(1 for r in results if r.tool_selection_ok)),
        tool_count_accuracy=percent(sum(1 for r in results if r.tool_count_ok)),
        task_completion_rate=percent(sum(1 for r in results if r.completion_ok)),
        tool_error_handling_rate=error_rate,
        response_grounding_rate=percent(sum(1 for r in results if r.grounding_ok)),
    )


def format_report(report: EvaluationReport) -> str:
    """Render an EvaluationReport as the plain-text CLI report.

    Cases are grouped by concern — Business (Step 6), knowledge/RAG
    (Step 12), Memory (Step 14), Approval execution (Step 16), and
    Observability — so an addition in one area can never silently
    shift another group's picture. These are deterministic workflow
    pass rates, not model accuracy claims.
    """
    error_rate = (
        "n/a" if report.tool_error_handling_rate is None
        else f"{report.tool_error_handling_rate}%"
    )
    non_business = (
        set(RAG_CATEGORIES)
        | set(MEMORY_CATEGORIES)
        | set(APPROVAL_CATEGORIES)
        | set(OBSERVABILITY_CATEGORIES)
    )
    business = [r for r in report.results if r.category not in non_business]
    knowledge = [r for r in report.results if r.category in RAG_CATEGORIES]
    memory = [r for r in report.results if r.category in MEMORY_CATEGORIES]
    approval = [r for r in report.results if r.category in APPROVAL_CATEGORIES]
    observability = [
        r for r in report.results if r.category in OBSERVABILITY_CATEGORIES
    ]

    def group_lines(name: str, results: list[CaseResult]) -> list[str]:
        if not results:
            return [f"{name}: 0 cases"]
        passed = sum(1 for r in results if r.passed)
        rate = round(100 * passed / len(results), 1)
        return [f"{name}: {passed}/{len(results)} passed ({rate}%)"]

    lines = [
        "DibantuAI Agent Evaluation",
        "==========================",
        f"Cases: {report.total}",
        f"Passed: {report.passed}",
        f"Failed: {report.failed}",
        f"Pass Rate: {report.pass_rate}%",
        "",
        *group_lines("Business cases (Step 6)", business),
        *group_lines("Knowledge/RAG cases (Step 12)", knowledge),
        *group_lines("Memory cases (Step 14)", memory),
        *group_lines("Approval execution cases (Step 16)", approval),
        *group_lines("Observability cases (Step 15/16)", observability),
        "",
        f"Tool Selection Accuracy: {report.tool_selection_accuracy}%",
        f"Task Completion Rate: {report.task_completion_rate}%",
        f"Tool Error Handling Rate: {error_rate}",
        f"Tool Call Count Accuracy: {report.tool_count_accuracy}%",
        f"Response Grounding Rate: {report.response_grounding_rate}%",
    ]
    failed = [r for r in report.results if not r.passed]
    if failed:
        lines.append("")
        lines.append("Failed cases:")
        for result in failed:
            lines.append(
                f"- {result.case_id} ({result.category}): "
                + "; ".join(result.failures)
            )
    return "\n".join(lines)


def main() -> None:
    """CLI entry point: ``python -m evaluation.evaluator``.

    Every case resets the store (``reset_mock_data()``), so the run is
    pinned to the isolated scratch database first — the main business
    database behind DATABASE_URL/.env is never touched (see
    ``evaluation/db_isolation.py``). Knowledge cases likewise run
    against a throwaway vector store (hashing embeddings), never the
    developer's real ``data/rag``.
    """
    from .db_isolation import ScratchDatabaseUnavailable, database_name, ensure_scratch_database

    try:
        url = ensure_scratch_database()
    except ScratchDatabaseUnavailable as exc:
        raise SystemExit(f"Cannot evaluate: {exc}") from exc
    print(f"Scratch database: {database_name(url)} (main business DB untouched)")
    print("RAG scratch store: temporary (hashing embeddings, real data/rag untouched)")
    print(format_report(run_evaluation()))


if __name__ == "__main__":
    main()
