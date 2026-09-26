"""Orchestrator for the real WhatsApp Cloud API flow (Step 17).

    Meta webhook POST
      → parse (text messages only; statuses/unsupported ignored)
      → claim each wamid in PostgreSQL (duplicate suppression)
      → respond 200 immediately (Meta must not wait for the LLM)
      → background: DibantuAgent (owner = Meta sender wa_id)
                  → WhatsApp-format + chunk the reply
                  → send via the Meta Graph API

The agent, its tools, RAG, memory scoping, approvals, and observability
are the exact same machinery as /api/chat and the local simulator —
only the transport differs. Sensitive actions still go through the
human-in-the-loop approval flow; nothing is auto-executed from
WhatsApp.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi.concurrency import run_in_threadpool
from sqlalchemy.exc import SQLAlchemyError

from app.agent.agent import DibantuAgent
from app.db import database
from app.models.schemas import AgentRequest
from app.whatsapp import repository
from app.whatsapp.client import (
    WhatsAppCloudClient,
    WhatsAppConfigError,
    WhatsAppSendError,
)
from app.whatsapp.config import WhatsAppConfig
from app.whatsapp.formatting import chunk_text, format_reply_for_whatsapp
from app.whatsapp.schemas import IncomingMessage, MetaWebhookEvent
from app.whatsapp.webhook import extract_text_messages

__all__ = [
    "WebhookOutcome",
    "handle_meta_webhook",
    "process_message",
    "run_agent_reply",
]

logger = logging.getLogger("dibantu.whatsapp")

#: Observability source label for runs that entered via the real
#: Meta webhook (the simulator keeps its own "webhook_whatsapp").
SOURCE_WHATSAPP_CLOUD = "webhook_whatsapp_cloud"


@dataclass(frozen=True)
class WebhookOutcome:
    """What the POST handler decided — returned to Meta as the ack."""

    status: str  # accepted | ignored | disabled
    new_messages: int
    duplicates: int


async def handle_meta_webhook(
    event: MetaWebhookEvent,
    *,
    config: WhatsAppConfig,
    agent: DibantuAgent,
    client: WhatsAppCloudClient | None,
    background_tasks,
) -> WebhookOutcome:
    """Parse, claim idempotently, and schedule background processing.

    Duplicate detection happens HERE — synchronously, before any agent
    run — so a Meta redelivery can never execute the agent twice. The
    HTTP response is fast (no LLM call in it); the actual agent run and
    the WhatsApp send happen in the background task.
    """
    if not config.enabled:
        return WebhookOutcome(status="disabled", new_messages=0, duplicates=0)

    messages = extract_text_messages(event)
    if not messages:
        return WebhookOutcome(status="ignored", new_messages=0, duplicates=0)

    claimed: list[IncomingMessage] = []
    duplicates = 0
    try:
        for message in messages:
            # Claiming through the threadpool keeps the async handler
            # off blocking database calls (same pattern as the API).
            is_new = await run_in_threadpool(
                _claim, message.message_id, message.sender
            )
            if is_new:
                claimed.append(message)
            else:
                duplicates += 1
    except SQLAlchemyError:
        # Without the idempotency ledger we cannot guarantee
        # exactly-once processing; asking Meta to retry (503) is safer
        # than possibly double-executing an order.
        raise

    for message in claimed:
        background_tasks.add_task(
            process_message, message, agent=agent, client=client
        )
    return WebhookOutcome(
        status="accepted", new_messages=len(claimed), duplicates=duplicates
    )


def _claim(message_id: str, sender_id: str) -> bool:
    """Claim a wamid in its own committed transaction."""
    with database.session_scope() as session:
        return repository.claim_message(
            session, message_id=message_id, sender_id=sender_id
        )


async def process_message(
    message: IncomingMessage,
    *,
    agent: DibantuAgent,
    client: WhatsAppCloudClient | None,
) -> None:
    """Run the agent for one claimed message and send the reply.

    Failures are contained: the agent run keeps its own observability
    status, the ledger row is marked failed, and a safe warning is
    logged (ids only — never message content in bulk, never tokens).
    """
    try:
        result = await run_agent_reply(
            agent,
            message=message.text,
            owner_key=message.sender,
            source=SOURCE_WHATSAPP_CLOUD,
        )
    except Exception as exc:  # noqa: BLE001 - the webhook must not crash
        logger.warning(
            "WhatsApp message %s: agent run failed (%s).",
            message.message_id,
            exc.__class__.__name__,
        )
        _mark(message.message_id, repository.mark_failed)
        return

    reply = format_reply_for_whatsapp(result.reply)
    try:
        if client is None:
            raise WhatsAppConfigError(
                "WhatsApp sending is not configured "
                "(WHATSAPP_ACCESS_TOKEN / WHATSAPP_PHONE_NUMBER_ID)."
            )
        for chunk in chunk_text(reply):
            await client.send_text(message.sender, chunk)
    except (WhatsAppSendError, WhatsAppConfigError) as exc:
        logger.warning(
            "WhatsApp message %s: reply could not be sent (%s).",
            message.message_id,
            exc.__class__.__name__,
        )
        _mark(message.message_id, repository.mark_failed)
        return

    _mark(message.message_id, repository.mark_processed)


async def run_agent_reply(
    agent: DibantuAgent, *, message: str, owner_key: str, source: str
):
    """Run the shared DibantuAgent for a WhatsApp-sourced message.

    Thin shared seam used by BOTH inbound WhatsApp flows (simulator and
    Meta Cloud API) so owner scoping, memory, tools, approvals, and
    observability behave identically.
    """
    return await agent.run(
        AgentRequest(message=message, owner_key=owner_key, source=source)
    )


def _mark(message_id: str, marker) -> None:
    """Best-effort ledger update; a failure here is only logged."""
    try:
        with database.session_scope() as session:
            marker(session, message_id)
    except SQLAlchemyError:  # pragma: no cover - ledger is best-effort
        logger.warning(
            "WhatsApp ledger update failed for %s.", message_id
        )
