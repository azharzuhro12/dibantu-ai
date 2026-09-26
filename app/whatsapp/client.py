"""Client for the Meta WhatsApp Cloud API send-message endpoint.

Sends plain text replies via
``POST https://graph.facebook.com/{VERSION}/{PHONE_NUMBER_ID}/messages``
exactly as documented by Meta (``messaging_product``/``recipient_type``/
``to``/``type``/``text`` body, ``Authorization: Bearer`` header). The
access token lives only in the Authorization header, is never logged,
never included in raised errors, and never returned to callers.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.whatsapp.config import WhatsAppConfig

__all__ = ["WhatsAppCloudClient", "WhatsAppConfigError", "WhatsAppSendError"]

logger = logging.getLogger("dibantu.whatsapp")

_GRAPH_BASE_URL = "https://graph.facebook.com"


class WhatsAppConfigError(RuntimeError):
    """Raised when sending is attempted without complete configuration."""


class WhatsAppSendError(RuntimeError):
    """A safe, token-free send failure (HTTP or network level).

    The message carries the HTTP status and Meta's business-level error
    text at most — never the request headers, never the access token.
    """


class WhatsAppCloudClient:
    """Thin async client for the Cloud API messages endpoint.

    ``transport`` is an httpx seam exactly like the GLM client's, so
    tests can answer Graph API calls with ``httpx.MockTransport``
    without any network or real credentials.
    """

    def __init__(
        self,
        config: WhatsAppConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=_GRAPH_BASE_URL,
            timeout=config.timeout_seconds,
            transport=transport,
        )

    async def send_text(self, to: str, body: str) -> str | None:
        """Send one text message; return Meta's message id if any.

        Raises WhatsAppConfigError when the access token or phone
        number id is missing, and WhatsAppSendError on HTTP/network
        failures (safe text only — no headers, no token).
        """
        if not self._config.can_send():
            raise WhatsAppConfigError(
                "WhatsApp sending is not configured "
                "(WHATSAPP_ACCESS_TOKEN / WHATSAPP_PHONE_NUMBER_ID)."
            )
        path = (
            f"/{self._config.api_version}/{self._config.phone_number_id}"
            "/messages"
        )
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"body": body},
        }
        try:
            response = await self._client.post(
                path,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._config.access_token}",
                    "Content-Type": "application/json",
                },
            )
        except httpx.TimeoutException as exc:
            logger.warning(
                "WhatsApp send timed out after %ss (recipient=%s).",
                self._config.timeout_seconds,
                to,
            )
            raise WhatsAppSendError(
                f"WhatsApp API timed out after {self._config.timeout_seconds}s."
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning(
                "WhatsApp send failed at network level (%s).",
                exc.__class__.__name__,
            )
            raise WhatsAppSendError(
                "WhatsApp API could not be reached "
                f"({exc.__class__.__name__})."
            ) from exc

        if response.status_code >= 400:
            detail = _safe_error_detail(response)
            logger.warning(
                "WhatsApp send rejected with HTTP %s (recipient=%s): %s",
                response.status_code,
                to,
                detail,
            )
            raise WhatsAppSendError(
                f"WhatsApp API returned HTTP {response.status_code}"
                f"{f': {detail}' if detail else ''}."
            )

        try:
            data: dict[str, Any] = response.json()
        except ValueError:
            return None
        messages = data.get("messages") or []
        if messages and isinstance(messages[0], dict):
            return str(messages[0].get("id") or "") or None
        return None

    async def aclose(self) -> None:
        """Release the underlying HTTP resources."""
        await self._client.aclose()


def _safe_error_detail(response: httpx.Response) -> str:
    """Meta's business-level error text, bounded — never raw bodies."""
    try:
        error = (response.json() or {}).get("error") or {}
    except ValueError:
        return ""
    message = error.get("message")
    if isinstance(message, str) and message:
        return message[:300]
    return ""
