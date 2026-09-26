"""Pydantic schemas used across the DibantuAI API and agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response payload for GET /health."""

    status: str = Field(default="ok", description="Service status.")
    service: str = Field(default="DibantuAI", description="Service name.")
    version: str = Field(..., description="Service version.")


class ChatMessage(BaseModel):
    """A single conversation turn."""

    role: str = Field(..., description="Message role, e.g. 'user' or 'assistant'.")
    content: str = Field(..., description="Message text.")


class AgentRequest(BaseModel):
    """Request payload for the agent endpoint (used from Step 2)."""

    message: str = Field(..., min_length=1, description="The user's request.")
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous conversation turns, oldest first.",
    )
    owner_key: str | None = Field(
        default=None,
        max_length=120,
        description=(
            "Application-level owner identity for memory scoping "
            "(Step 14); defaults to 'default'. Not authentication."
        ),
    )
    source: str = Field(
        default="api",
        max_length=30,
        description=(
            "Channel label for observability (Step 15), e.g. 'chat' or "
            "'webhook_whatsapp'."
        ),
    )


class AgentResponse(BaseModel):
    """Response payload for the agent endpoint (used from Step 2)."""

    reply: str = Field(..., description="The agent's reply text.")
    model: str = Field(..., description="GLM model that produced the reply.")
    run_id: str | None = Field(
        default=None,
        description="Observability run id (Step 15) tracing this request.",
    )


class ChatRequest(BaseModel):
    """Request payload for POST /api/chat."""

    message: str = Field(..., min_length=1, description="The user's chat message.")
    owner_key: str | None = Field(
        default=None,
        max_length=120,
        description=(
            "Optional application-level owner identity for agent memory "
            "(Step 14). Omitting it keeps the previous behavior ('default' "
            "scope). Not authentication."
        ),
    )


class ChatResponse(BaseModel):
    """Response payload for POST /api/chat."""

    response: str = Field(..., description="The assistant's reply text.")
    run_id: str | None = Field(
        default=None,
        description="Observability run id (Step 15) tracing this request.",
    )


class WhatsAppWebhookRequest(BaseModel):
    """Request payload for POST /webhook/whatsapp (local simulator, Step 7).

    Mirrors the shape of a WhatsApp text message: ``from`` is the
    sender's phone number (kept as ``sender`` because ``from`` is a
    Python keyword; the JSON field stays ``from`` via the alias).
    """

    sender: str = Field(
        ...,
        alias="from",
        min_length=1,
        description="Sender's WhatsApp number, e.g. '628123456789'.",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Incoming WhatsApp message text.",
    )


class WhatsAppWebhookResponse(BaseModel):
    """Response payload for POST /webhook/whatsapp (local simulator)."""

    to: str = Field(..., description="WhatsApp number the reply is addressed to.")
    response: str = Field(..., description="The agent's reply text.")
    run_id: str | None = Field(
        default=None,
        description="Observability run id (Step 15) tracing this request.",
    )


class ApprovalResponse(BaseModel):
    """One approval record (human-in-the-loop, Step 8; Step 13: persistent; Step 16: execution)."""

    approval_id: str = Field(..., description="Unique approval id (apr-...).")
    action: str = Field(..., description="Sensitive action requested.")
    requested_by: str = Field(..., description="Who requested the action.")
    payload: dict[str, Any] = Field(
        ...,
        description=(
            "Immutable execution snapshot: the exact JSON captured when the "
            "agent requested the action. Human review AND executor input — "
            "clients can never re-supply it."
        ),
    )
    status: str = Field(
        ...,
        description=(
            "pending, approved, rejected, executing, executed, or failed."
        ),
    )
    created_at: str = Field(..., description="ISO timestamp of the request.")
    decided_at: str | None = Field(
        default=None, description="ISO timestamp of the decision, if any."
    )
    decision_reason: str | None = Field(
        default=None,
        description="Optional human note recorded with the decision (Step 13).",
    )
    execution_started_at: str | None = Field(
        default=None,
        description="ISO timestamp the approved action started executing (Step 16).",
    )
    executed_at: str | None = Field(
        default=None,
        description="ISO timestamp the execution finished (executed or failed).",
    )
    execution_error: str | None = Field(
        default=None,
        description=(
            "Business-level reason the execution failed (bounded; no "
            "tracebacks, no secrets)."
        ),
    )
    execution_result: dict[str, Any] | None = Field(
        default=None,
        description="The sensitive tool's result dict on successful execution.",
    )


class ApprovalListResponse(BaseModel):
    """Response payload for GET /api/approvals."""

    approvals: list[ApprovalResponse] = Field(
        ..., description="Pending approvals, oldest first."
    )
    count: int = Field(..., description="Number of pending approvals.")


class MemoryCreateRequest(BaseModel):
    """Request payload for POST /api/memory (Step 14)."""

    owner_key: str | None = Field(
        default=None,
        max_length=120,
        description="Application-level owner scope; defaults to 'default'.",
    )
    memory_type: str = Field(
        ...,
        description="One of: preference, customer_context, business_context, instruction.",
    )
    content: str = Field(..., min_length=1, max_length=2000, description="The fact to remember.")


class MemoryResponse(BaseModel):
    """One memory record (Step 14)."""

    memory_id: int = Field(..., description="Stable id of the memory.")
    owner_key: str = Field(..., description="Application-level owner scope.")
    memory_type: str = Field(..., description="Kind of fact (preference, ...).")
    content: str = Field(..., description="The remembered fact.")
    created_at: str = Field(..., description="ISO timestamp of creation.")
    updated_at: str = Field(..., description="ISO timestamp of last update.")


class MemoryListResponse(BaseModel):
    """Response payload for GET /api/memory."""

    memories: list[MemoryResponse] = Field(
        ..., description="Memories, newest first."
    )
    count: int = Field(..., description="Number of memories returned.")


AgentRunStatus = Literal["running", "completed", "failed"]
AgentEventType = Literal["RUN", "LLM", "TOOL", "MEMORY", "RAG", "APPROVAL"]


class AgentRunResponse(BaseModel):
    """One traced agent run (Step 15)."""

    run_id: str = Field(..., description="Stable id tying the run's events together.")
    source: str = Field(..., description="Channel that started the run (chat, webhook_whatsapp, ...).")
    owner_key: str | None = Field(default=None, description="Application-level owner scope, if any.")
    status: str = Field(..., description="running, completed, or failed.")
    request_preview: str | None = Field(
        default=None, description="Short prefix of the user message (never the full prompt)."
    )
    started_at: str = Field(..., description="ISO timestamp the run started.")
    completed_at: str | None = Field(default=None, description="ISO timestamp the run finished.")
    duration_ms: int | None = Field(default=None, description="Total run duration in milliseconds.")
    error_type: str | None = Field(
        default=None, description="Exception class name when the run failed (never a traceback)."
    )
    event_count: int = Field(default=0, description="Events recorded for this run.")


class AgentRunListResponse(BaseModel):
    """Response payload for GET /api/observability/runs."""

    runs: list[AgentRunResponse] = Field(..., description="Runs, newest first.")
    count: int = Field(..., description="Number of runs returned.")


class AgentEventResponse(BaseModel):
    """One traced operation inside a run (Step 15)."""

    event_id: int = Field(..., description="Stable event id (insertion order).")
    run_id: str = Field(..., description="Owning run id.")
    event_type: str = Field(..., description="RUN, LLM, TOOL, MEMORY, RAG, or APPROVAL.")
    event_name: str = Field(..., description="e.g. LLM_CALL, TOOL_CALL, MEMORY_SEARCH.")
    status: str = Field(..., description="started, success, or failed.")
    iteration: int | None = Field(default=None, description="Agent loop pass (1-based), when known.")
    started_at: str = Field(..., description="ISO timestamp the operation started.")
    completed_at: str | None = Field(default=None, description="ISO timestamp it finished.")
    duration_ms: int | None = Field(default=None, description="Duration in milliseconds.")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Small, safe metadata (tool name, model, usage); never payloads."
    )
    error_type: str | None = Field(default=None, description="Exception class name on failure.")


class AgentEventListResponse(BaseModel):
    """Response payload for GET /api/observability/runs/{run_id}/events."""

    run_id: str = Field(..., description="The run the events belong to.")
    events: list[AgentEventResponse] = Field(..., description="Events in order.")
    count: int = Field(..., description="Number of events returned.")


class KnowledgeSearchResult(BaseModel):
    """One retrieved knowledge passage with its citation metadata."""

    content: str = Field(..., description="The passage text.")
    source: str = Field(..., description="Document file name, e.g. 'refund_policy.md'.")
    section: str = Field(..., description="Section title the passage was cut from.")
    chunk_id: str = Field(..., description="Stable id of the chunk.")
    similarity: float = Field(..., description="Cosine similarity of the passage to the query.")


class KnowledgeSearchResponse(BaseModel):
    """Response payload for GET /api/knowledge/search (RAG, Step 12)."""

    success: bool = Field(..., description="Whether the search ran.")
    query: str = Field(..., description="The query that was searched.")
    results: list[KnowledgeSearchResult] = Field(
        default_factory=list, description="Retrieved passages, best first."
    )
    sources: list[str] = Field(
        default_factory=list, description="Unique source documents, in result order."
    )
    message: str | None = Field(
        default=None, description="Hint when the store is empty, etc."
    )


class KnowledgeIngestResponse(BaseModel):
    """Response payload for POST /api/knowledge/ingest (RAG, Step 12)."""

    documents_found: int = Field(..., description="Markdown documents seen in the knowledge directory.")
    documents_created: int = Field(..., description="Documents ingested for the first time.")
    documents_updated: int = Field(..., description="Changed documents re-ingested.")
    documents_skipped: int = Field(..., description="Unchanged documents skipped (idempotency).")
    chunks_created: int = Field(..., description="Chunks embedded and stored this run.")
    total_chunks: int = Field(..., description="Chunks now in the vector store.")
    knowledge_dir: str = Field(..., description="Knowledge directory that was ingested.")
    store_dir: str = Field(..., description="Vector store directory.")


class InventoryItemResponse(BaseModel):
    """One product row for GET /api/inventory.

    Field semantics mirror the ``check_stock`` tool exactly (same
    source of truth, same low-stock definition), so the inventory
    table and the chat answer can never disagree.
    """

    name: str = Field(..., description="Product name.")
    price: int | float = Field(..., description="Unit price (IDR, whole number).")
    stock: int = Field(..., description="Current stock quantity.")
    low_stock_threshold: int = Field(
        ..., description="Stock level at/below which the product counts as low."
    )
    in_stock: bool = Field(..., description="Whether stock > 0.")
    low_stock: bool = Field(
        ..., description="Whether 0 < stock <= low_stock_threshold."
    )


class InventoryListResponse(BaseModel):
    """Response payload for GET /api/inventory (read-only)."""

    products: list[InventoryItemResponse] = Field(
        default_factory=list, description="Every product, in catalog order."
    )
    count: int = Field(..., description="Number of products returned.")


class OrderCustomerResponse(BaseModel):
    """Customer block embedded in an order row (GET /api/orders)."""

    name: str = Field(..., description="Customer display name.")
    phone: str | None = Field(None, description="Phone number, when known.")
    email: str | None = Field(None, description="Email address, when known.")


class OrderItemResponse(BaseModel):
    """One order line for GET /api/orders — the priced snapshot stored
    on the row at creation time (name and price never drift with later
    catalog edits)."""

    product_name: str = Field(..., description="Product name as ordered.")
    quantity: int = Field(..., description="Units ordered.")
    unit_price: int | float = Field(..., description="Unit price (IDR).")
    line_total: int | float = Field(
        ..., description="Subtotal for this line (unit_price * quantity)."
    )


class OrderResponse(BaseModel):
    """One order row for GET /api/orders.

    Field semantics mirror the ``create_order`` tool's payload (same
    tables, same money conversion), so the orders table and the chat
    answer can never disagree.
    """

    order_id: str = Field(..., description="Public order code (ORD-####).")
    customer: OrderCustomerResponse | None = Field(
        None, description="Customer info when available."
    )
    items: list[OrderItemResponse] = Field(
        default_factory=list, description="Order lines with priced snapshots."
    )
    total_price: int | float = Field(..., description="Order total (IDR).")
    status: str = Field(..., description="Order status (completed/refunded/cancelled).")
    created_at: str = Field(
        ..., description="Creation time (local, second precision, ISO 8601)."
    )


class OrderListResponse(BaseModel):
    """Response payload for GET /api/orders (read-only)."""

    orders: list[OrderResponse] = Field(
        default_factory=list, description="Every order, newest first."
    )
    count: int = Field(..., description="Number of orders returned.")


class ReportWindowResponse(BaseModel):
    """One rolling sales window for GET /api/reports.

    Field semantics mirror the ``get_sales_report`` tool exactly (same
    ``sales_window`` aggregation, same completed-orders-only rule), so
    the report cards and the chat answer can never disagree.
    """

    period: str = Field(..., description="Window name (daily/weekly/monthly).")
    window_days: int = Field(..., description="Rolling window length in days.")
    total_orders: int = Field(
        ..., description="Completed orders in the window (refunded/cancelled excluded)."
    )
    total_items_sold: int = Field(..., description="Units sold in the window.")
    total_revenue: int | float = Field(..., description="Revenue in the window (IDR).")
    products_sold: dict[str, int] = Field(
        default_factory=dict,
        description="Units sold per product name, in order of first sale.",
    )
    top_product: str | None = Field(
        None, description="Best-selling product in the window (first max), if any."
    )


class OrderStatusSummaryResponse(BaseModel):
    """Order counts by status for GET /api/reports, as stored in the
    database (no fixed status set is assumed)."""

    by_status: dict[str, int] = Field(
        default_factory=dict,
        description="Order count for every status present in the table.",
    )
    total: int = Field(..., description="Orders overall, every status combined.")


class ReportsResponse(BaseModel):
    """Response payload for GET /api/reports (read-only)."""

    generated_at: str = Field(
        ..., description="When the report was computed (local, ISO 8601)."
    )
    status_summary: OrderStatusSummaryResponse = Field(
        ..., description="Order counts by status, across all orders."
    )
    windows: list[ReportWindowResponse] = Field(
        default_factory=list,
        description="Daily, weekly, and monthly rolling sales windows.",
    )
