"""Additional API routes for DibantuAI (chat endpoint backed by the GLM
tool-calling agent loop, Step 4; local WhatsApp webhook simulator, Step 7;
human-in-the-loop approval endpoints, Step 8; knowledge-base endpoints,
Step 12)."""

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from app.agent.agent import DibantuAgent
from app.agent.glm_client import GLMConfigError, GLMError
from app.approval import ApprovalNotPendingError, ApprovalNotFoundError
from app.approval import manager as approval_manager
from app.models.schemas import (
    AgentRequest,
    ApprovalListResponse,
    ApprovalResponse,
    ChatRequest,
    ChatResponse,
    KnowledgeIngestResponse,
    KnowledgeSearchResponse,
    WhatsAppWebhookRequest,
    WhatsAppWebhookResponse,
)
from app.rag.errors import RagError
from app.rag.service import get_rag_service

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
        result = await agent.run(AgentRequest(message=request.message))
    except GLMConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GLMError as exc:
        raise HTTPException(status_code=502, detail=f"GLM API error: {exc}") from exc
    return ChatResponse(response=result.reply)


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
        result = await agent.run(AgentRequest(message=request.message))
    except GLMConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GLMError as exc:
        raise HTTPException(status_code=502, detail=f"GLM API error: {exc}") from exc
    return WhatsAppWebhookResponse(to=request.sender, response=result.reply)


@router.get(
    "/api/approvals",
    response_model=ApprovalListResponse,
    tags=["approval"],
)
async def list_approvals() -> ApprovalListResponse:
    """List pending sensitive-action approvals, oldest first."""
    approvals = approval_manager.list_pending()
    return ApprovalListResponse(
        approvals=[ApprovalResponse(**asdict(a)) for a in approvals],
        count=len(approvals),
    )


@router.post(
    "/api/approvals/{approval_id}/approve",
    response_model=ApprovalResponse,
    tags=["approval"],
)
async def approve_approval(approval_id: str) -> ApprovalResponse:
    """Approve a pending approval and return the approved payload.

    MVP (Step 8): approving records the human decision only -- no real
    refund/payment is executed anywhere.
    """
    try:
        approval = approval_manager.approve(approval_id)
    except ApprovalNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"Approval '{approval_id}' not found."
        ) from exc
    except ApprovalNotPendingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ApprovalResponse(**asdict(approval))


@router.post(
    "/api/approvals/{approval_id}/reject",
    response_model=ApprovalResponse,
    tags=["approval"],
)
async def reject_approval(approval_id: str) -> ApprovalResponse:
    """Reject a pending approval."""
    try:
        approval = approval_manager.reject(approval_id)
    except ApprovalNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"Approval '{approval_id}' not found."
        ) from exc
    except ApprovalNotPendingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ApprovalResponse(**asdict(approval))


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
