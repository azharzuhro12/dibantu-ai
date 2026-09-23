"""Pydantic schemas used across the DibantuAI API and agent."""

from __future__ import annotations

from typing import Any

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


class AgentResponse(BaseModel):
    """Response payload for the agent endpoint (used from Step 2)."""

    reply: str = Field(..., description="The agent's reply text.")
    model: str = Field(..., description="GLM model that produced the reply.")


class ChatRequest(BaseModel):
    """Request payload for POST /api/chat."""

    message: str = Field(..., min_length=1, description="The user's chat message.")


class ChatResponse(BaseModel):
    """Response payload for POST /api/chat."""

    response: str = Field(..., description="The assistant's reply text.")


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


class ApprovalResponse(BaseModel):
    """One approval record (human-in-the-loop, Step 8)."""

    approval_id: str = Field(..., description="Unique approval id (apr-...).")
    action: str = Field(..., description="Sensitive action requested.")
    requested_by: str = Field(..., description="Who requested the action.")
    payload: dict[str, Any] = Field(..., description="Action payload awaiting approval.")
    status: str = Field(..., description="pending, approved, or rejected.")
    created_at: str = Field(..., description="ISO timestamp of the request.")
    decided_at: str | None = Field(
        default=None, description="ISO timestamp of the decision, if any."
    )


class ApprovalListResponse(BaseModel):
    """Response payload for GET /api/approvals."""

    approvals: list[ApprovalResponse] = Field(
        ..., description="Pending approvals, oldest first."
    )
    count: int = Field(..., description="Number of pending approvals.")
