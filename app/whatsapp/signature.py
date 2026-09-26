"""X-Hub-Signature-256 validation for Meta webhooks (Step 17).

Meta signs every webhook delivery: the ``X-Hub-Signature-256`` header
carries ``sha256=<hexdigest>``, the hex HMAC-SHA256 of the **raw request
body** (byte-for-byte as delivered — never re-serialized JSON, which
would differ in key order/whitespace) keyed with the Meta App Secret
(App Dashboard → App Settings → Basic). Validation is enforced only
when an app secret is configured (``WHATSAPP_APP_SECRET``); the secret
is never logged, never returned, and never used for anything except
this HMAC.
"""

from __future__ import annotations

import hashlib
import hmac

__all__ = ["SIGNATURE_HEADER", "compute_signature", "is_valid_signature"]

#: The HTTP header Meta sets on every webhook POST (lowercase: header
#: names are case-insensitive, and Starlette lookups are lowercase).
SIGNATURE_HEADER = "x-hub-signature-256"

#: Prefix Meta puts in front of the hex digest ("sha256=<hex>").
_SIGNATURE_PREFIX = "sha256="


def compute_signature(raw_body: bytes, app_secret: str) -> str:
    """The expected header value for ``raw_body``: ``sha256=<hexdigest>``.

    Exactly Meta's documented construction: HMAC-SHA256 over the raw
    request bytes with the App Secret as the key.
    """
    digest = hmac.new(
        app_secret.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()
    return f"{_SIGNATURE_PREFIX}{digest}"


def is_valid_signature(
    raw_body: bytes, header_value: str | None, app_secret: str
) -> bool:
    """Whether the delivered signature authenticates the raw body.

    Constant-time comparison (``hmac.compare_digest`` on the encoded
    digests, so arbitrary header bytes never raise). Missing header,
    missing secret, a missing ``sha256=`` prefix, non-hex garbage, or a
    digest over any other bytes all return False — callers reject with
    403 before the payload is parsed or processed any further.
    """
    if not app_secret or not header_value:
        return False
    provided = header_value.strip()
    if not provided.startswith(_SIGNATURE_PREFIX):
        return False
    provided_hex = provided[len(_SIGNATURE_PREFIX):].strip().lower()
    expected_hex = compute_signature(raw_body, app_secret)[len(_SIGNATURE_PREFIX):]
    return hmac.compare_digest(
        expected_hex.encode("ascii"), provided_hex.encode("utf-8")
    )
