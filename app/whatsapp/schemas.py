"""Pydantic models for the Meta WhatsApp Cloud API wire format.

Shapes follow Meta's official webhook components documentation: a
webhook is ``{"object": "whatsapp_business_account", "entry": [...]}``
where each entry carries changes whose ``value`` holds either
``messages`` (an incoming message — the only event DibantuAI treats as
a user message), ``statuses`` (delivery receipts for messages we sent
— ignored), or other fields. Unknown fields are ignored on purpose so
Meta adding fields never breaks parsing; unsupported message types are
dropped safely by the parser rather than by validation errors.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "IncomingMessage",
    "MetaWebhookEvent",
    "WebhookChange",
    "WebhookContact",
    "WebhookEntry",
    "WebhookMessage",
    "WebhookMetadata",
    "WebhookProfile",
    "WebhookStatus",
    "WebhookText",
    "WebhookValue",
]


class _MetaModel(BaseModel):
    """Base for Meta payload models: tolerate fields Meta may add."""

    model_config = ConfigDict(extra="ignore")


class WebhookText(_MetaModel):
    """The ``text`` object of a text message: just the body."""

    body: str = Field(default="", description="Message text.")


class WebhookMessage(_MetaModel):
    """One incoming message inside ``value.messages``."""

    id: str = Field(default="", description="WhatsApp message id (wamid.*).")
    from_: str = Field(
        default="",
        alias="from",
        description="Sender's WhatsApp id (phone number digits).",
    )
    timestamp: str = Field(default="", description="Unix seconds.")
    type: str = Field(default="", description="Message type (text, image, ...).")
    text: WebhookText | None = Field(
        default=None, description="Present only for type=text."
    )


class WebhookProfile(_MetaModel):
    """The ``profile`` block of a contact (display name)."""

    name: str = Field(default="", description="WhatsApp display name.")


class WebhookContact(_MetaModel):
    """Contact information Meta includes alongside messages."""

    profile: WebhookProfile | None = Field(default=None)
    wa_id: str = Field(default="", description="WhatsApp id (phone digits).")


class WebhookMetadata(_MetaModel):
    """Business-side metadata (our number / phone number id)."""

    display_phone_number: str = Field(default="")
    phone_number_id: str = Field(default="")


class WebhookStatus(_MetaModel):
    """Delivery status for a message WE sent (sent/delivered/read)."""

    id: str = Field(default="", description="wamid of the outgoing message.")
    status: str = Field(default="", description="sent | delivered | read | failed.")


class WebhookValue(_MetaModel):
    """The ``value`` object: messages in, statuses out, never both."""

    messaging_product: str = Field(default="")
    metadata: WebhookMetadata | None = Field(default=None)
    contacts: list[WebhookContact] = Field(default_factory=list)
    messages: list[WebhookMessage] = Field(default_factory=list)
    statuses: list[WebhookStatus] = Field(default_factory=list)


class WebhookChange(_MetaModel):
    """One change inside an entry."""

    field: str = Field(default="")
    value: WebhookValue = Field(default_factory=WebhookValue)


class WebhookEntry(_MetaModel):
    """One webhook entry (per WhatsApp business account)."""

    id: str = Field(default="")
    changes: list[WebhookChange] = Field(default_factory=list)


class MetaWebhookEvent(_MetaModel):
    """The full Meta webhook POST payload.

    ``object`` and ``entry`` are required so a Meta envelope can never
    be mistaken for the local simulator payload (which has neither) —
    the two share one endpoint (POST /webhook/whatsapp).
    """

    object: str = Field(
        ..., description="Always 'whatsapp_business_account'."
    )
    entry: list[WebhookEntry] = Field(min_length=1)


@dataclass(frozen=True)
class IncomingMessage:
    """A parsed, actionable incoming text message.

    ``sender`` (the Meta ``from`` wa_id) is the stable owner identity
    used for memory scoping; ``message_id`` is the wamid used for
    idempotency.
    """

    message_id: str
    sender: str
    text: str
    profile_name: str | None = None
