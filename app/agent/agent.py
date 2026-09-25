"""Core agent module for DibantuAI.

``DibantuAgent`` is the API-facing agent (Step 4): it owns the settings
and GLM client and delegates to ``Agent``, the simple tool-calling loop
(Step 4C). The loop sends the user message together with the registry
tool schemas to GLM via ``complete_with_tools``, executes the requested
tool(s), feeds the results back as Anthropic-style ``tool_use``/
``tool_result`` messages, and returns the model's final text answer.
Since Step 5 a single turn may execute several tool calls (parallel
tool_use blocks), which keeps multi-product workflows -- check every
product, then create one order -- within the iteration budget. Tool
executions are capped at ``MAX_TOOL_ITERATIONS`` to prevent infinite
loops.
"""

from __future__ import annotations

import inspect
import json
from time import perf_counter
from typing import Any, Callable

import httpx

from app.agent.glm_client import GLMClient, GLMResponse, SYSTEM_PROMPT, ToolUse
from app.approval import (
    approval_required_result,
    create_pending_approval,
    is_sensitive_action,
)
from app.config import Settings, get_settings
from app.models.schemas import AgentRequest, AgentResponse
from app.tools.registry import get_tool, get_tool_schemas

#: Maximum number of tool executions per ``Agent.run`` call.
MAX_TOOL_ITERATIONS = 5

#: Reply returned when the loop exhausts the tool iteration cap.
TOOL_LIMIT_REACHED_TEXT = (
    "Sorry, I hit the tool call limit before I could finish that request."
)

#: System prompt for the tool-calling agent (Steps 4-5).
AGENT_SYSTEM_PROMPT = (
    "You are DibantuAI, an AI operations assistant for small businesses "
    "such as warungs, cafes, and small shops. You manage real business "
    "data: product stock, orders, customers, and sales reports.\n\n"
    "Follow these rules strictly:\n"
    "1. Use the provided tools whenever real business data is needed or a "
    "business action must be performed (checking stock, creating orders, "
    "updating stock, finding customers, sales reports). Never invent or "
    "guess business data such as stock levels, prices, orders, or "
    "customer records.\n"
    "2. Always wait for the tool result and confirm it before responding. "
    "Base your answer only on what the tool actually returned.\n"
    "3. If required information is missing or ambiguous (for example an "
    "unclear product or customer name), ask a short clarification "
    "question instead of guessing.\n"
    "4. If a tool reports an error or fails, say so honestly and explain "
    "what went wrong. Never claim an action succeeded when the tool "
    "failed.\n"
    "5. Keep responses concise and actionable, and reply in the user's "
    "language (Indonesian or English).\n"
    "6. Some actions (refund_order, cancel_order, bulk_stock_update) are "
    "sensitive and need human approval. When a tool result says "
    "\"approval_required\": true, the action was NOT executed: tell the "
    "user it is waiting for human approval, share the approval_id, and "
    "never claim the action happened.\n"
    "7. Documented business knowledge (policies, procedures, rules — "
    "refunds, cancellations, stock rules, payments, operating rules) "
    "lives in a separate knowledge base: call search_knowledge_base for "
    "those questions. Answer ONLY from the returned passages, cite the "
    "source document inline (e.g. \"Menurut refund_policy.md, ...\" and "
    "the section when available), and never invent or guess policy "
    "details. If the tool returns no results, an error, or passages "
    "that do not actually cover the question, say plainly that the "
    "information is not available in the knowledge base.\n"
    "8. Keep the two data sources strictly separate: the database tools "
    "(check_stock, create_order, update_stock, search_customer, "
    "get_sales_report, get_low_stock) return live business data, while "
    "search_knowledge_base returns documented policies. Never present "
    "knowledge-base text as live stock/order numbers, and never present "
    "live data as a documented policy.\n"
    "9. Long-lived facts about the user and the business live in a "
    "memory store. Save a memory ONLY when the user explicitly asks to "
    "remember something (save_memory), recall with search_memory when "
    "they ask what you remember, and delete with delete_memory when they "
    "ask you to forget. Never save secrets (passwords, API keys, tokens, "
    "payment credentials) or unsolicited personal details, and never "
    "perform a business action through memory — it is context only. "
    "Saved facts may appear in the prompt as 'Known facts' context; "
    "treat them as helpful hints, never as live data.\n\n"
    "Workflow playbooks:\n"
    "- Stock check: call check_stock for the product and answer strictly "
    "from the returned stock, price, in_stock, and low_stock data.\n"
    "- New order: when the customer is not clearly known, call "
    "search_customer first to identify them. Check stock for every "
    "requested product before ordering, then create ONE create_order "
    "call containing every line item. create_order deducts stock "
    "automatically -- never call update_stock to compensate for an order; "
    "use update_stock only for explicit restocks or manual corrections "
    "requested by the user.\n"
    "- Insufficient stock: if check_stock or create_order reports "
    "insufficient stock, do not force or split the order to work around "
    "it. Report the actual available amount and the shortage, and "
    "suggest ordering a smaller quantity. Mention restocking as an "
    "option, but only perform it if the user asks.\n"
    "- Sales report: call get_sales_report with the requested period "
    "(default daily) and summarize total orders, items sold, revenue, "
    "and the top product using only the returned data.\n"
    "- Low stock: call get_low_stock with an explicit threshold (use 10 "
    "when the user does not give one) and list the returned products "
    "with their actual stock levels.\n"
    "- Policy/procedure question: call search_knowledge_base with the "
    "user's question, then answer strictly from the returned passages "
    "and cite each source document (and section) you used. Mixed "
    "requests (policy + live data) need both: search_knowledge_base "
    "for the policy part and the database tool for the live numbers, "
    "and keep the two clearly labelled in your answer.\n"
    "- Remember/recall: when the user asks you to remember something, "
    "save it with save_memory and confirm; when they ask what you "
    "remember, search_memory first and answer only from what it "
    "returns (say so plainly when nothing relevant is stored)."
)


class DibantuAgent:
    """Business operations agent backed by the GLM API.

    Composes the ``Agent`` tool-calling loop (Step 4) with the business
    tools registered in ``app.tools.registry``: by default the loop both
    advertises ``get_tool_schemas()`` and dispatches through
    ``get_tool``, so schemas and tools are never duplicated here. The
    ``tools``/``schemas``/``max_tool_iterations`` seams exist so API
    tests can inject fakes without network access.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: GLMClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        tools: dict[str, Callable[..., Any]] | None = None,
        schemas: list[dict[str, Any]] | None = None,
        max_tool_iterations: int = MAX_TOOL_ITERATIONS,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client or GLMClient(self.settings, transport=transport)
        self._agent = Agent(
            self._client,
            tools=tools,
            schemas=schemas,
            max_tool_iterations=max_tool_iterations,
            system=AGENT_SYSTEM_PROMPT,
        )

    def is_configured(self) -> bool:
        """Return True if a GLM API key is available."""
        return bool(self.settings.glm_api_key)

    async def run(self, request: AgentRequest) -> AgentResponse:
        """Process a single user request through the GLM tool-calling loop.

        Step 14: the request's owner key (WhatsApp sender, chat owner
        key, or "default") scopes the run — relevant saved memories for
        that owner are appended to the system prompt as a small context
        block, and the memory tools inside the run write to (and read
        from) that owner's scope only. With no stored memories the
        prompt and behavior are exactly as before.

        Step 15: the run is traced end-to-end (LLM + tool events with
        iteration numbers, tied to one stable ``run_id`` returned to the
        caller). Observability failures are swallowed by the recorder
        and can never break the business request; a business error still
        propagates after the run is marked failed.
        """
        from app.memory.manager import (
            DEFAULT_OWNER_KEY,
            memory_context_block,
            owner_scope,
        )
        from app.observability import TraceRecorder

        history = [message.model_dump() for message in request.history]
        owner_key = (request.owner_key or DEFAULT_OWNER_KEY).strip() or DEFAULT_OWNER_KEY
        extra_system = memory_context_block(owner_key, request.message)
        recorder = TraceRecorder(
            source=request.source,
            owner_key=owner_key,
            request_preview=request.message,
            model=self.settings.glm_model,
        )
        recorder.start()
        try:
            with owner_scope(owner_key):
                reply = await self._agent.run(
                    request.message,
                    history=history,
                    extra_system=extra_system,
                    trace=recorder,
                )
        except Exception as exc:
            recorder.fail(exc)
            raise
        recorder.complete()
        return AgentResponse(
            reply=reply, model=self.settings.glm_model, run_id=recorder.run_id
        )


class Agent:
    """Simple tool-calling agent loop (Step 4C).

    ``client`` is any object exposing ``complete_with_tools`` -- the real
    ``GLMClient`` or a test double. Tools are resolved through the
    registry's ``get_tool`` by default; passing ``tools`` replaces the
    lookup table and ``schemas`` overrides the advertised schemas. Both
    seams keep unit tests free of network calls.
    """

    def __init__(
        self,
        client: Any,
        *,
        tools: dict[str, Callable[..., Any]] | None = None,
        schemas: list[dict[str, Any]] | None = None,
        max_tool_iterations: int = MAX_TOOL_ITERATIONS,
        system: str = SYSTEM_PROMPT,
    ) -> None:
        self._client = client
        self._tools = tools
        self._schemas = schemas if schemas is not None else get_tool_schemas()
        self._max_tool_iterations = max_tool_iterations
        self._system = system

    async def run(
        self,
        user_message: str,
        *,
        history: list[dict[str, Any]] | None = None,
        extra_system: str = "",
        trace: Any = None,
    ) -> str:
        """Answer ``user_message``, running tools until GLM sends plain text.

        ``history`` holds earlier conversation turns in the Anthropic
        message shape and is sent before the user message; it may be
        omitted for single-turn requests. ``extra_system`` (Step 14) is
        appended to this agent's system prompt for the whole run — the
        composition layer uses it to inject a small, relevant memory
        context; the default keeps the prompt byte-identical to before.
        ``trace`` (Step 15) is an optional ``TraceRecorder``; when given,
        every LLM call and tool execution of the run is traced with its
        iteration number (1-based loop pass). The default keeps the loop
        untraced and behavior identical.
        Each iteration sends the conversation so far to GLM. A text-only
        reply is returned immediately. One turn may request several
        tools (Step 5, e.g. a stock check per product): every requested
        call is executed, echoed back as one assistant message of
        ``tool_use`` blocks plus one user message of matching
        ``tool_result`` blocks, and GLM is called again for the final
        answer. At most ``max_tool_iterations`` tools run per call, so a
        turn that would exceed the budget executes only its leading
        tools.
        """
        messages: list[dict[str, Any]] = list(history) if history else []
        messages.append({"role": "user", "content": user_message})
        system = self._system + extra_system
        executed = 0
        iteration = 0
        while executed < self._max_tool_iterations:
            iteration += 1
            response = await self._call_glm_traced(messages, system, trace, iteration)
            if response.tool_name is None:
                return response.text or ""
            batch = _tool_uses_of(response)[: self._max_tool_iterations - executed]
            assistant_blocks: list[dict[str, Any]] = []
            result_blocks: list[dict[str, Any]] = []
            if response.text:
                assistant_blocks.append({"type": "text", "text": response.text})
            for tool_use in batch:
                tool_use_id = tool_use.id or f"toolu_{tool_use.name}_{executed}"
                result = await self._run_tool(
                    tool_use.name,
                    tool_use.input,
                    trace=trace,
                    iteration=iteration,
                )
                assistant_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": tool_use.name,
                        "input": tool_use.input,
                    }
                )
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result,
                    }
                )
                executed += 1
            messages.append({"role": "assistant", "content": assistant_blocks})
            messages.append({"role": "user", "content": result_blocks})
        iteration += 1
        response = await self._call_glm_traced(messages, system, trace, iteration)
        return response.text or TOOL_LIMIT_REACHED_TEXT

    async def _call_glm(
        self, messages: list[dict[str, Any]], *, system: str | None = None
    ) -> GLMResponse:
        """Send one turn to GLM with the system prompt and schemas."""
        return await self._client.complete_with_tools(
            messages,
            system=self._system if system is None else system,
            tools=self._schemas,
        )

    async def _call_glm_traced(
        self,
        messages: list[dict[str, Any]],
        system: str,
        trace: Any,
        iteration: int,
    ) -> GLMResponse:
        """``_call_glm`` with an optional Step 15 LLM event per call.

        The event carries only safe metadata (model, duration, usage
        when the provider reported it) — never prompts or responses.
        Tracing failures are swallowed by the recorder.
        """
        if trace is None:
            return await self._call_glm(messages, system=system)
        started = perf_counter()
        try:
            response = await self._call_glm(messages, system=system)
        except Exception as exc:
            trace.llm_call(
                iteration=iteration,
                duration_ms=int((perf_counter() - started) * 1000),
                usage=None,
                error_type=type(exc).__name__,
            )
            raise
        trace.llm_call(
            iteration=iteration,
            duration_ms=int((perf_counter() - started) * 1000),
            usage=getattr(response, "usage", None),
        )
        return response

    async def _run_tool(
        self,
        name: str,
        tool_input: dict[str, Any],
        *,
        trace: Any = None,
        iteration: int | None = None,
    ) -> str:
        """Execute one tool call and serialize the result for the model.

        Sensitive actions (Step 8: refund_order, cancel_order,
        bulk_stock_update) are never executed: requesting one creates a
        pending approval and returns a structured approval-required
        result instead, so the model tells the user a human must
        approve. This is the central choke point every tool call from
        every route passes through. Unknown tools and tool exceptions
        become ``{"error": ...}`` JSON strings so the model can react
        instead of crashing the loop. Sync and async tool callables are
        both supported. With ``trace`` (Step 15) each execution is
        recorded as a typed tool/memory/RAG/approval event — metadata
        holds the tool name and safe extras only, never the payload.
        """
        started = perf_counter()
        if is_sensitive_action(name):
            approval = create_pending_approval(
                action=name, requested_by="agent", payload=tool_input
            )
            if trace is not None:
                trace.tool_call(
                    name,
                    iteration=iteration,
                    duration_ms=int((perf_counter() - started) * 1000),
                    ok=True,
                    extra_metadata={"approval_id": approval.approval_id},
                )
            return json.dumps(
                approval_required_result(approval), ensure_ascii=False
            )
        if self._tools is not None:
            tool = self._tools.get(name)
        else:
            tool = get_tool(name)
        if tool is None:
            if trace is not None:
                trace.tool_call(
                    name,
                    iteration=iteration,
                    duration_ms=int((perf_counter() - started) * 1000),
                    ok=False,
                    error_type="UnknownTool",
                )
            return json.dumps({"error": f"Unknown tool: {name}"})
        error_type: str | None = None
        try:
            result = tool(**tool_input)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            error_type = type(exc).__name__
            if trace is not None:
                trace.tool_call(
                    name,
                    iteration=iteration,
                    duration_ms=int((perf_counter() - started) * 1000),
                    ok=False,
                    error_type=error_type,
                )
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
        if trace is not None:
            trace.tool_call(
                name,
                iteration=iteration,
                duration_ms=int((perf_counter() - started) * 1000),
                ok=True,
                error_type=None,
            )
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)


def _tool_uses_of(response: Any) -> list[ToolUse]:
    """Collect every tool call requested by one GLM response.

    Prefers ``tool_uses`` (all blocks, Step 5) and falls back to the
    single ``tool_name``/``tool_input`` pair so response shapes without
    ``tool_uses`` (e.g. older test doubles) keep working unchanged.
    """
    tool_uses = list(getattr(response, "tool_uses", None) or ())
    if tool_uses:
        return tool_uses
    if response.tool_name is not None:
        return [ToolUse(name=response.tool_name, input=response.tool_input or {})]
    return []
