"""Additional API routes for DibantuAI (chat endpoint backed by the GLM
tool-calling agent loop, Step 4; local WhatsApp webhook simulator, Step 7;
human-in-the-loop approval endpoints, Step 8; knowledge-base endpoints,
Step 12; approvals persisted to PostgreSQL, Step 13; persistent agent
memory endpoints, Step 14; deferred approval execution endpoint, Step 16)."""

from dataclasses import asdict
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.exc import SQLAlchemyError

from app.agent.agent import DibantuAgent
from app.agent.glm_client import GLMConfigError, GLMError
from app.approval import ApprovalNotPendingError, ApprovalNotFoundError
from app.approval import manager as approval_manager
from app.approval.executor import (
    ApprovalConflictError,
    ApprovalNotExecutableError,
    execute_approval,
)
from app.db import database, repository
from app.db.models import money
from app.memory import (
    DEFAULT_OWNER_KEY,
    MemoryNotFoundError,
    MemoryValidationError,
    manager as memory_manager,
)
from app.models.schemas import (
    AgentEventListResponse,
    AgentEventResponse,
    AgentRequest,
    AgentRunListResponse,
    AgentRunResponse,
    AgentRunStatus,
    AgentEventType,
    ApprovalListResponse,
    ApprovalResponse,
    ChatRequest,
    ChatResponse,
    InventoryItemResponse,
    InventoryListResponse,
    KnowledgeIngestResponse,
    KnowledgeSearchResponse,
    MemoryCreateRequest,
    MemoryListResponse,
    MemoryResponse,
    OrderCustomerResponse,
    OrderItemResponse,
    OrderListResponse,
    OrderResponse,
    OrderStatusSummaryResponse,
    ReportWindowResponse,
    ReportsResponse,
    WhatsAppWebhookRequest,
    WhatsAppWebhookResponse,
)
from app.rag.errors import RagError
from app.rag.service import get_rag_service
from app.observability import (
    get_run,
    get_run_events,
    list_runs as list_observability_runs,
    trace_approval_decision,
)

router = APIRouter()


def get_agent() -> DibantuAgent:
    """Build the agent used by API routes (overridable in tests).

    The agent runs the GLM tool-calling loop over the registered
    business tools (stock, orders, customers, reports).
    """
    return DibantuAgent()


@router.post("/api/chat", response_model=ChatResponse, tags=["chat"])
async def chat(
    request: ChatRequest, agent: DibantuAgent = Depends(get_agent)
) -> ChatResponse:
    """Reply to a user message via the GLM tool-calling agent loop.

    Flow: message -> Agent -> GLM -> tool_use -> business tool ->
    tool_result -> GLM -> final reply, returned as ``ChatResponse``.
    """
    if not agent.is_configured():
        raise HTTPException(
            status_code=503, detail="GLM_API_KEY is not configured."
        )
    try:
        result = await agent.run(
            AgentRequest(
                message=request.message,
                owner_key=request.owner_key,
                source="chat",
            )
        )
    except GLMConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GLMError as exc:
        raise HTTPException(status_code=502, detail=f"GLM API error: {exc}") from exc
    return ChatResponse(response=result.reply, run_id=result.run_id)


@router.post(
    "/webhook/whatsapp",
    response_model=WhatsAppWebhookResponse,
    tags=["webhook"],
)
async def whatsapp_webhook(
    request: WhatsAppWebhookRequest,
    agent: DibantuAgent = Depends(get_agent),
) -> WhatsAppWebhookResponse:
    """Simulated WhatsApp webhook (Step 7): answer a WhatsApp-style message.

    Simulator only -- no Meta verification or signature check yet. The
    message goes through the same tool-calling agent dependency as
    /api/chat (tool failures stay visible to the agent instead of
    crashing the webhook), and the reply is addressed back to the
    sender's number.
    """
    if not agent.is_configured():
        raise HTTPException(
            status_code=503, detail="GLM_API_KEY is not configured."
        )
    try:
        result = await agent.run(
            # The WhatsApp sender number is the stable owner identity
            # for memory scoping (Step 14).
            AgentRequest(
                message=request.message,
                owner_key=request.sender,
                source="webhook_whatsapp",
            )
        )
    except GLMConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GLMError as exc:
        raise HTTPException(status_code=502, detail=f"GLM API error: {exc}") from exc
    return WhatsAppWebhookResponse(
        to=request.sender, response=result.reply, run_id=result.run_id
    )


def _raise_approval_store_error(exc: Exception) -> None:
    """Map an approval-store failure to a 503 (DB unreachable, etc.).

    The original exception is chained for server logs but never reaches
    the client — the response body stays a plain detail string.
    """
    raise HTTPException(
        status_code=503,
        detail="Approval store is unavailable (database error).",
    ) from exc


@router.get(
    "/api/approvals",
    response_model=ApprovalListResponse,
    tags=["approval"],
)
async def list_approvals(
    status: str | None = Query(
        None,
        pattern="^(pending|approved|rejected|executing|executed|failed|all)$",
        description=(
            "Filter by status; omit for pending only (the original "
            "behavior), 'all' for every approval."
        ),
    ),
) -> ApprovalListResponse:
    """List approvals, oldest first (default: pending only, Step 13).

    Step 16 adds the ``status`` query parameter so the lifecycle after
    the decision (approved/executing/executed/failed) is visible too —
    the frontend uses it to offer execution of approved actions.
    """
    try:
        if status is None:
            approvals = approval_manager.list_pending()
        elif status == "all":
            approvals = approval_manager.list_approvals()
        else:
            approvals = approval_manager.list_approvals([status])
    except SQLAlchemyError as exc:
        _raise_approval_store_error(exc)
    return ApprovalListResponse(
        approvals=[ApprovalResponse(**asdict(a)) for a in approvals],
        count=len(approvals),
    )


@router.post(
    "/api/approvals/{approval_id}/approve",
    response_model=ApprovalResponse,
    tags=["approval"],
)
async def approve_approval(
    approval_id: str,
    reason: str | None = Query(
        None, max_length=500, description="Optional note recorded with the decision."
    ),
) -> ApprovalResponse:
    """Approve a pending approval — the decision only, never execution.

    Step 16: approving still executes nothing. The approved action runs
    only through the explicit execute endpoint, driven by the immutable
    payload stored at creation. The decision and timestamp are
    persisted (Step 13); a second decision on the same approval fails
    with 409 because the change is a guarded, atomic database update.
    """
    try:
        approval = approval_manager.approve(approval_id, reason=reason)
    except ApprovalNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"Approval '{approval_id}' not found."
        ) from exc
    except ApprovalNotPendingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except approval_manager.ApprovalError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        _raise_approval_store_error(exc)
    # Informational tracing only — never executes or bypasses anything.
    trace_approval_decision(approval, approved=True)
    return ApprovalResponse(**asdict(approval))


@router.post(
    "/api/approvals/{approval_id}/reject",
    response_model=ApprovalResponse,
    tags=["approval"],
)
async def reject_approval(
    approval_id: str,
    reason: str | None = Query(
        None, max_length=500, description="Optional note recorded with the decision."
    ),
) -> ApprovalResponse:
    """Reject a pending approval, optionally recording a reason.

    A rejected approval can never be executed (Step 16): the executor's
    atomic claim only accepts the ``approved`` status.
    """
    try:
        approval = approval_manager.reject(approval_id, reason=reason)
    except ApprovalNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"Approval '{approval_id}' not found."
        ) from exc
    except ApprovalNotPendingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except approval_manager.ApprovalError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        _raise_approval_store_error(exc)
    # Informational tracing only — never executes or bypasses anything.
    trace_approval_decision(approval, approved=False)
    return ApprovalResponse(**asdict(approval))


@router.post(
    "/api/approvals/{approval_id}/execute",
    response_model=ApprovalResponse,
    tags=["approval"],
)
async def execute_approved_action(approval_id: str) -> ApprovalResponse:
    """Execute an APPROVED sensitive action exactly once (Step 16).

    Runs the action from the immutable payload snapshot stored at
    creation — this endpoint accepts no arguments of any kind. The
    ``approved -> executing`` claim is one atomic guarded UPDATE, so
    two concurrent execute attempts can never both run the action, and
    every terminal state (executed/failed) or pre-decision state
    (pending/rejected) answers 409 without side effects. The response
    carries the finished approval: ``status`` is ``executed`` with
    ``execution_result``, or ``failed`` with a bounded, business-level
    ``execution_error`` (never a traceback).
    """
    try:
        approval = await run_in_threadpool(execute_approval, approval_id)
    except ApprovalNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"Approval '{approval_id}' not found."
        ) from exc
    except ApprovalConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ApprovalNotExecutableError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except approval_manager.ApprovalError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        _raise_approval_store_error(exc)
    return ApprovalResponse(**asdict(approval))


@router.get(
    "/api/observability/runs",
    response_model=AgentRunListResponse,
    tags=["observability"],
)
async def list_agent_runs(
    status: AgentRunStatus | None = Query(
        None, description="Filter by run status."
    ),
    source: str | None = Query(
        None, max_length=30, description="Filter by channel (chat, webhook_whatsapp, ...)."
    ),
    owner_key: str | None = Query(
        None,
        max_length=120,
        description="Filter by application-level owner scope (NOT authentication).",
    ),
    limit: int = Query(50, ge=1, le=200),
) -> AgentRunListResponse:
    """List recent traced agent runs, newest first (Step 15).

    Read-only. ``owner_key`` follows the same application-level
    ownership model as memory: the caller states the scope — this is
    not authentication-based isolation.
    """
    try:
        runs = list_observability_runs(
            status=status, source=source, owner_key=owner_key, limit=limit
        )
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Observability store is unavailable (database error).",
        ) from exc
    return AgentRunListResponse(
        runs=[AgentRunResponse(**r.as_dict()) for r in runs],
        count=len(runs),
    )


@router.get(
    "/api/observability/runs/{run_id}",
    response_model=AgentRunResponse,
    tags=["observability"],
)
async def get_agent_run(run_id: str) -> AgentRunResponse:
    """Retrieve one traced run by its stable run_id (Step 15)."""
    try:
        run = get_run(run_id)
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Observability store is unavailable (database error).",
        ) from exc
    if run is None:
        raise HTTPException(
            status_code=404, detail=f"Run '{run_id}' not found."
        )
    return AgentRunResponse(**run.as_dict())


@router.get(
    "/api/observability/runs/{run_id}/events",
    response_model=AgentEventListResponse,
    tags=["observability"],
)
async def get_agent_run_events(
    run_id: str,
    event_type: AgentEventType | None = Query(
        None, description="Filter events by type."
    ),
    limit: int = Query(500, ge=1, le=1000),
) -> AgentEventListResponse:
    """One run's event timeline, in occurrence order (Step 15)."""
    try:
        events = get_run_events(run_id, event_type=event_type, limit=limit)
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Observability store is unavailable (database error).",
        ) from exc
    if events is None:
        raise HTTPException(
            status_code=404, detail=f"Run '{run_id}' not found."
        )
    return AgentEventListResponse(
        run_id=run_id,
        events=[AgentEventResponse(**e.as_dict()) for e in events],
        count=len(events),
    )


@router.get(
    "/api/memory",
    response_model=MemoryListResponse,
    tags=["memory"],
)
async def list_memories(
    owner_key: str = Query(
        DEFAULT_OWNER_KEY,
        max_length=120,
        description="Application-level owner scope (NOT authentication).",
    ),
    memory_type: str | None = Query(
        None, description="Optionally filter by memory type."
    ),
    q: str | None = Query(
        None, min_length=1, description="Optional keyword filter."
    ),
    limit: int = Query(100, ge=1, le=500),
) -> MemoryListResponse:
    """List one owner's memories, newest first (Step 14).

    Ownership here is application-level only: the caller states the
    owner key, exactly like the agent tools do per request. There is
    deliberately no authentication in this project yet.
    """
    try:
        if q:
            memories = memory_manager.search_memories(
                owner_key=owner_key, query=q, memory_type=memory_type,
                limit=limit,
            )
        else:
            memories = memory_manager.list_memories(
                owner_key=owner_key, memory_type=memory_type, limit=limit
            )
    except MemoryValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        _raise_memory_store_error(exc)
    return MemoryListResponse(
        memories=[MemoryResponse(**m.as_dict()) for m in memories],
        count=len(memories),
    )


@router.post(
    "/api/memory",
    response_model=MemoryResponse,
    tags=["memory"],
)
async def create_memory(request: MemoryCreateRequest) -> MemoryResponse:
    """Create one memory for an owner scope (Step 14).

    Validation lives in the manager: closed set of memory types,
    bounded content, and a secret screen that refuses password/API
    key/token/card-like content.
    """
    owner_key = (request.owner_key or DEFAULT_OWNER_KEY).strip() or DEFAULT_OWNER_KEY
    try:
        memory = memory_manager.create_memory(
            owner_key=owner_key,
            memory_type=request.memory_type,
            content=request.content,
        )
    except MemoryValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        _raise_memory_store_error(exc)
    return MemoryResponse(**memory.as_dict())


@router.delete(
    "/api/memory/{memory_id}",
    response_model=MemoryResponse,
    tags=["memory"],
)
async def delete_memory(
    memory_id: int,
    owner_key: str = Query(
        DEFAULT_OWNER_KEY,
        max_length=120,
        description="Owner scope the memory must belong to.",
    ),
) -> MemoryResponse:
    """Delete one of an owner's memories.

    Unknown ids and other owners' ids are both a plain 404 (an owner
    cannot probe which ids exist outside their scope).
    """
    try:
        memory = memory_manager.delete_memory(memory_id, owner_key=owner_key)
    except MemoryNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"Memory '{memory_id}' not found."
        ) from exc
    except MemoryValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        _raise_memory_store_error(exc)
    return MemoryResponse(**memory.as_dict())


def _raise_memory_store_error(exc: Exception) -> None:
    """Map a memory-store failure to a 503 without leaking internals."""
    raise HTTPException(
        status_code=503,
        detail="Memory store is unavailable (database error).",
    ) from exc


@router.post(
    "/api/knowledge/ingest",
    response_model=KnowledgeIngestResponse,
    tags=["knowledge"],
)
async def ingest_knowledge() -> KnowledgeIngestResponse:
    """(Re)ingest the configured knowledge directory into ChromaDB.

    Deliberately takes no request body: ingestion is restricted to the
    server-configured ``KNOWLEDGE_DIR`` — callers can never point it at
    an arbitrary filesystem path. Idempotent; heavy work (embedding)
    runs in a thread so the event loop is not blocked.
    """
    try:
        report = await run_in_threadpool(get_rag_service().ingest)
    except RagError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return KnowledgeIngestResponse(
        documents_found=report.documents_found,
        documents_created=report.documents_created,
        documents_updated=report.documents_updated,
        documents_skipped=report.documents_skipped,
        chunks_created=report.chunks_created,
        total_chunks=report.total_chunks,
        knowledge_dir=get_rag_service().knowledge_dir,
        store_dir=report.store_dir,
    )


@router.get(
    "/api/knowledge/search",
    response_model=KnowledgeSearchResponse,
    tags=["knowledge"],
)
async def search_knowledge(
    q: str = Query(..., min_length=1, description="Question or keywords to look up."),
    top_k: int | None = Query(
        None, ge=1, le=10, description="Passages to return (default 3)."
    ),
) -> KnowledgeSearchResponse:
    """Search the knowledge base (retrieval only; no LLM involved).

    The primary user-facing path stays POST /api/chat — this endpoint
    exists for inspecting what the RAG tool would retrieve.
    """
    try:
        result = await run_in_threadpool(
            get_rag_service().search, q, top_k=top_k
        )
    except RagError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return KnowledgeSearchResponse(**result)


@router.get(
    "/api/inventory",
    response_model=InventoryListResponse,
    tags=["inventory"],
)
async def get_inventory() -> InventoryListResponse:
    """Read-only product inventory for the dashboard table.

    Lists every product straight from PostgreSQL through the shared
    repository — the same source of truth the ``check_stock`` tool
    reads — with the same per-product fields ``check_stock`` computes
    (price, in-stock, low-stock against the product's own threshold).
    No LLM, no tool calling; read-only.
    """
    try:
        with database.session_scope() as session:
            products = repository.list_products(session)
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Inventory is unavailable (database error).",
        ) from exc
    return InventoryListResponse(
        products=[
            InventoryItemResponse(
                name=product.name,
                price=money(product.price),
                stock=product.stock_quantity,
                low_stock_threshold=product.low_stock_threshold,
                in_stock=product.stock_quantity > 0,
                low_stock=0 < product.stock_quantity <= product.low_stock_threshold,
            )
            for product in products
        ],
        count=len(products),
    )


@router.get(
    "/api/orders",
    response_model=OrderListResponse,
    tags=["orders"],
)
async def get_orders() -> OrderListResponse:
    """Read-only order history for the dashboard table.

    Lists every order straight from PostgreSQL through the shared
    repository — the same rows the ``create_order`` tool writes and the
    sales report reads — newest first, with the customer info and each
    line's priced snapshot (product name, quantity, unit price,
    subtotal). No LLM, no tool calling; read-only.
    """
    try:
        with database.session_scope() as session:
            orders = repository.list_orders(session)
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Orders are unavailable (database error).",
        ) from exc
    return OrderListResponse(
        orders=[
            OrderResponse(
                order_id=order.id,
                customer=(
                    OrderCustomerResponse(
                        name=order.customer.name,
                        phone=order.customer.phone,
                        email=order.customer.email,
                    )
                    if order.customer is not None
                    else None
                ),
                items=[
                    OrderItemResponse(
                        product_name=item.product_name,
                        quantity=item.quantity,
                        unit_price=money(item.unit_price),
                        line_total=money(item.line_total),
                    )
                    for item in order.items
                ],
                total_price=money(order.total_price),
                status=order.status,
                created_at=order.created_at.isoformat(timespec="seconds"),
            )
            for order in orders
        ],
        count=len(orders),
    )


#: Rolling windows reported by GET /api/reports — the same periods (and
#: day lengths) the ``get_sales_report`` tool serves, so the report
#: cards and a chat answer can never disagree.
_REPORT_WINDOWS: tuple[tuple[str, int], ...] = (
    ("daily", 1),
    ("weekly", 7),
    ("monthly", 30),
)


@router.get(
    "/api/reports",
    response_model=ReportsResponse,
    tags=["reports"],
)
async def get_reports() -> ReportsResponse:
    """Read-only sales report for the dashboard.

    Aggregates straight from PostgreSQL through the shared repository —
    the same ``sales_window`` computation the ``get_sales_report`` tool
    runs (completed orders only; refunded/cancelled excluded because
    their stock was restored and revenue undone) — for the daily,
    weekly, and monthly rolling windows, plus a count of orders per
    status across the whole table. No LLM, no tool calling; read-only.
    """
    now = datetime.now()
    try:
        with database.session_scope() as session:
            status_counts = dict(repository.count_orders_by_status(session))
            windows = [
                ReportWindowResponse(
                    period=period,
                    window_days=window_days,
                    **repository.sales_window(
                        session, now=now, window_days=window_days
                    ),
                )
                for period, window_days in _REPORT_WINDOWS
            ]
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Reports are unavailable (database error).",
        ) from exc
    return ReportsResponse(
        generated_at=now.isoformat(timespec="seconds"),
        status_summary=OrderStatusSummaryResponse(
            by_status=status_counts, total=sum(status_counts.values())
        ),
        windows=windows,
    )
