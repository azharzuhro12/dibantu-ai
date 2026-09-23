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
    "never claim the action happened.\n\n"
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
    "with their actual stock levels."
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
        """Process a single user request through the GLM tool-calling loop."""
        history = [message.model_dump() for message in request.history]
        reply = await self._agent.run(request.message, history=history)
        return AgentResponse(reply=reply, model=self.settings.glm_model)


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
    ) -> str:
        """Answer ``user_message``, running tools until GLM sends plain text.

        ``history`` holds earlier conversation turns in the Anthropic
        message shape and is sent before the user message; it may be
        omitted for single-turn requests. Each iteration sends the
        conversation so far to GLM. A text-only reply is returned
        immediately. One turn may request several tools (Step 5, e.g. a
        stock check per product): every requested call is executed,
        echoed back as one assistant message of ``tool_use`` blocks plus
        one user message of matching ``tool_result`` blocks, and GLM is
        called again for the final answer. At most
        ``max_tool_iterations`` tools run per call, so a turn that would
        exceed the budget executes only its leading tools.
        """
        messages: list[dict[str, Any]] = list(history) if history else []
        messages.append({"role": "user", "content": user_message})
        executed = 0
        while executed < self._max_tool_iterations:
            response = await self._call_glm(messages)
            if response.tool_name is None:
                return response.text or ""
            batch = _tool_uses_of(response)[: self._max_tool_iterations - executed]
            assistant_blocks: list[dict[str, Any]] = []
            result_blocks: list[dict[str, Any]] = []
            if response.text:
                assistant_blocks.append({"type": "text", "text": response.text})
            for tool_use in batch:
                tool_use_id = tool_use.id or f"toolu_{tool_use.name}_{executed}"
                result = await self._run_tool(tool_use.name, tool_use.input)
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
        response = await self._call_glm(messages)
        return response.text or TOOL_LIMIT_REACHED_TEXT

    async def _call_glm(self, messages: list[dict[str, Any]]) -> GLMResponse:
        """Send one turn to GLM with this agent's system prompt and schemas."""
        return await self._client.complete_with_tools(
            messages, system=self._system, tools=self._schemas
        )

    async def _run_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        """Execute one tool call and serialize the result for the model.

        Sensitive actions (Step 8: refund_order, cancel_order,
        bulk_stock_update) are never executed: requesting one creates a
        pending approval and returns a structured approval-required
        result instead, so the model tells the user a human must
        approve. This is the central choke point every tool call from
        every route passes through. Unknown tools and tool exceptions
        become ``{"error": ...}`` JSON strings so the model can react
        instead of crashing the loop. Sync and async tool callables are
        both supported.
        """
        if is_sensitive_action(name):
            approval = create_pending_approval(
                action=name, requested_by="agent", payload=tool_input
            )
            return json.dumps(
                approval_required_result(approval), ensure_ascii=False
            )
        if self._tools is not None:
            tool = self._tools.get(name)
        else:
            tool = get_tool(name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            result = tool(**tool_input)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
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
