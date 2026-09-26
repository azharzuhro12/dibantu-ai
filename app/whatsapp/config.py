"""Environment configuration for the WhatsApp Cloud API integration.

All values come from environment variables (see ``.env.example``); the
integration is DISABLED by default so the application behaves exactly
as before until WhatsApp is deliberately switched on. Secrets (access
token, verify token, app secret) are read here and never logged,
returned by the API, or sent to the frontend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

__all__ = [
    "DEFAULT_API_VERSION",
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_TEXT_LENGTH",
    "WhatsAppConfig",
    "get_whatsapp_config",
]

#: Graph API version used for the Cloud API calls. v25.0 is the version
#: Meta's own send-message documentation example uses and has the
#: longest support runway (until mid-2028); override with
#: WHATSAPP_API_VERSION when Meta deprecates it.
DEFAULT_API_VERSION = "v25.0"

#: Explicit timeout for every call to the Graph API — no call may hang.
DEFAULT_TIMEOUT_SECONDS = 15.0

#: WhatsApp text messages accept at most 4096 characters per message
#: body; longer agent replies are chunked deterministically (see
#: app/whatsapp/formatting.py).
MAX_TEXT_LENGTH = 4096


@dataclass(frozen=True)
class WhatsAppConfig:
    """Runtime configuration for the WhatsApp Cloud API integration."""

    enabled: bool = False
    verify_token: str = ""
    access_token: str = ""
    phone_number_id: str = ""
    app_secret: str = ""
    api_version: str = DEFAULT_API_VERSION
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def can_send(self) -> bool:
        """Whether replies can be sent (needs token + phone number id)."""
        return bool(self.access_token.strip() and self.phone_number_id.strip())

    def can_verify(self) -> bool:
        """Whether webhook verification can succeed (needs verify token)."""
        return bool(self.verify_token.strip())

    def can_validate_signature(self) -> bool:
        """Whether webhook signatures must be validated (needs app secret).

        Empty by default: signature enforcement is deliberately opt-in so
        deployments without an app secret keep working exactly as before.
        """
        return bool(self.app_secret.strip())


def get_whatsapp_config() -> WhatsAppConfig:
    """Build the WhatsApp configuration from the environment.

    Read on every call (cheap) so tests and runtime overrides with
    monkeypatched environment variables take effect immediately; no
    value is cached and nothing is logged here.
    """
    raw_enabled = os.getenv("WHATSAPP_ENABLED", "false").strip().lower()
    try:
        timeout = float(os.getenv("WHATSAPP_TIMEOUT_SECONDS", "").strip() or DEFAULT_TIMEOUT_SECONDS)
    except ValueError:
        timeout = DEFAULT_TIMEOUT_SECONDS
    return WhatsAppConfig(
        enabled=raw_enabled in ("1", "true", "yes", "on"),
        verify_token=os.getenv("WHATSAPP_VERIFY_TOKEN", ""),
        access_token=os.getenv("WHATSAPP_ACCESS_TOKEN", ""),
        phone_number_id=os.getenv("WHATSAPP_PHONE_NUMBER_ID", ""),
        app_secret=os.getenv("WHATSAPP_APP_SECRET", ""),
        api_version=os.getenv("WHATSAPP_API_VERSION", "").strip() or DEFAULT_API_VERSION,
        timeout_seconds=timeout,
    )
