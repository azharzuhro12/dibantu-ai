"""Parser for real Meta WhatsApp Cloud API webhook payloads.

Extracts the actionable events — incoming TEXT messages — and safely
ignores everything else: delivery ``statuses``, non-text message types
(images, audio, buttons, ...), and fields Meta may add later. The
parser never raises on unexpected-but-valid envelopes; structural
garbage is rejected earlier by the Pydantic envelope models.
"""

from __future__ import annotations

from app.whatsapp.schemas import (
    IncomingMessage,
    MetaWebhookEvent,
    WebhookContact,
)

__all__ = ["extract_text_messages"]


def extract_text_messages(event: MetaWebhookEvent) -> list[IncomingMessage]:
    """All incoming text messages in a Meta webhook payload, in order.

    Non-text messages, status updates, and entries without messages are
    dropped. A message without an id, sender, or body cannot be made
    idempotent or answered, so it is dropped too.
    """
    incoming: list[IncomingMessage] = []
    for entry in event.entry:
        for change in entry.changes:
            value = change.value
            contacts = value.contacts or []
            for message in value.messages:
                if message.type != "text":
                    continue
                text = (message.text.body if message.text else "") or ""
                message_id = message.id.strip()
                sender = message.from_.strip()
                if not message_id or not sender or not text.strip():
                    continue
                incoming.append(
                    IncomingMessage(
                        message_id=message_id,
                        sender=sender,
                        text=text,
                        profile_name=_profile_name(contacts, sender),
                    )
                )
    return incoming


def _profile_name(
    contacts: list[WebhookContact], sender: str
) -> str | None:
    """The sender's profile name when Meta included a contact card."""
    for contact in contacts:
        if contact.wa_id and contact.wa_id == sender:
            return (contact.profile.name or None) if contact.profile else None
    return None
