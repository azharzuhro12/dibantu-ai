"""HTTP client for the GLM API (Anthropic-compatible Messages endpoint).

Configuration comes from the dotenv-backed app settings; the API key is
never hardcoded or logged. All errors surface as GLMError subclasses so
callers get a single, predictable exception hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings, get_settings

#: Temporary system prompt for Step 2.
SYSTEM_PROMPT = (
    "You are DibantuAI, an AI operations assistant for small businesses. "
    "Keep responses concise, helpful, and actionable."
)

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_TOKENS = 1024
_ANTHROPIC_VERSION = "2023-06-01"


class GLMError(Exception):
    """Base class for all GLM client errors."""


class GLMConfigError(GLMError):
    """Raised when the GLM settings are incomplete (e.g. missing API key)."""


class GLMTimeoutError(GLMError):
    """Raised when the GLM API does not respond in time."""


class GLMAuthError(GLMError):
    """Raised when the GLM API rejects the credentials."""


class GLMAPIError(GLMError):
    """Raised when the GLM API returns an error or an unexpected payload."""


@dataclass(frozen=True)
class ToolUse:
    """One ``tool_use`` block from a GLM response (Step 5).

    ``id`` is the API-provided tool_use id when present; the agent
    synthesizes one when the response shape omits it.
    """

    name: str
    input: dict[str, Any]
    id: str | None = None


@dataclass(frozen=True)
class GLMResponse:
    """Structured result of one GLM chat turn (Step 4B).

    ``text`` is the concatenation of all text blocks, or ``None`` when
    the model only requested a tool. ``tool_name``/``tool_input``
    describe the first ``tool_use`` block, if any, while ``tool_uses``
    (Step 5) lists every requested tool call in order so multi-product
    requests can be answered in a single turn. ``stop_reason`` mirrors
    the API field (typically ``"end_turn"`` for plain text and
    ``"tool_use"`` for tool calls). ``usage`` (Step 15) carries the
    provider-reported token counts when present — ``None`` otherwise
    (counts are never invented); it feeds observability only.
    """

    text: str | None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    stop_reason: str | None = None
    tool_uses: tuple[ToolUse, ...] = ()
    usage: dict[str, int] | None = None


class GLMClient:
    """Thin async client for GLM's Anthropic-compatible /v1/messages endpoint."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._timeout = timeout
        # Injectable transport lets tests mock HTTP without network access.
        self._transport = transport

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        system: str = SYSTEM_PROMPT,
    ) -> str:
        """Send a chat turn to GLM and return the assistant's text reply.

        ``messages`` follows the Anthropic shape: [{"role": ..., "content": ...}].
        Raises a GLMError subclass on configuration, network, or API problems.
        """
        payload = {
            "model": self.settings.glm_model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "system": system,
            "messages": messages,
        }
        data = await self._post_messages(payload)
        return _extract_text(data)

    async def complete_with_tools(
        self,
        messages: list[dict[str, str]],
        *,
        system: str = SYSTEM_PROMPT,
        tools: list[dict[str, Any]],
    ) -> GLMResponse:
        """Send a chat turn with tool schemas and return a GLMResponse.

        Behaves like ``complete`` but adds the payload's ``tools`` field
        so the model may request a tool call. ``tools`` holds the
        Anthropic-style schemas from ``get_tool_schemas()``. The reply is
        parsed with ``_parse_response`` so callers can distinguish plain
        text (``text`` set) from a tool request (``tool_name`` set).
        Raises a GLMError subclass on configuration, network, or API
        problems.
        """
        payload = {
            "model": self.settings.glm_model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "system": system,
            "messages": messages,
            "tools": tools,
        }
        data = await self._post_messages(payload)
        return _parse_response(data)

    async def _post_messages(self, payload: dict[str, Any]) -> Any:
        """POST one /v1/messages payload and return the parsed JSON body.

        Shared by ``complete`` and ``complete_with_tools`` so both use
        the same URL, authentication, and error handling.
        """
        if not self.settings.glm_api_key:
            raise GLMConfigError("GLM_API_KEY is not configured.")

        url = f"{self.settings.glm_base_url.rstrip('/')}/v1/messages"
        # Z.ai's Anthropic-compatible endpoint authenticates with a Bearer
        # token (the ANTHROPIC_AUTH_TOKEN style from Z.ai's own Claude Code
        # setup), not the Anthropic-native x-api-key header.
        headers = {
            "Authorization": f"Bearer {self.settings.glm_api_key}",
            "anthropic-version": _ANTHROPIC_VERSION,
        }

        async with httpx.AsyncClient(
            timeout=self._timeout, transport=self._transport
        ) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
            except httpx.TimeoutException as exc:
                raise GLMTimeoutError(
                    f"GLM API timed out after {self._timeout}s."
                ) from exc
            except httpx.HTTPError as exc:
                raise GLMAPIError(f"Could not reach the GLM API: {exc}") from exc

        if response.status_code in (401, 403):
            raise GLMAuthError(
                f"GLM API rejected the credentials (HTTP {response.status_code})."
            )
        if response.status_code >= 400:
            raise GLMAPIError(
                f"GLM API returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise GLMAPIError("GLM API returned a non-JSON response.") from exc


def _extract_text(data: Any) -> str:
    """Pull the concatenated text blocks out of an Anthropic-style response."""
    try:
        blocks = data["content"]
        text = "".join(
            block["text"] for block in blocks if block.get("type") == "text"
        )
    except (KeyError, TypeError) as exc:
        raise GLMAPIError(
            f"Unexpected GLM response payload: {str(data)[:200]}"
        ) from exc
    if not text.strip():
        raise GLMAPIError("GLM response contained no text content.")
    return text.strip()


def _parse_response(data: Any) -> GLMResponse:
    """Parse an Anthropic-compatible response into a GLMResponse.

    Extracts the concatenated ``text`` blocks, every ``tool_use`` block
    (``tool_uses``, in order — the first also fills ``tool_name``/
    ``tool_input`` for single-call callers), the top-level
    ``stop_reason``, and the provider-reported token ``usage`` when
    present. Raises GLMAPIError on malformed payloads or when the
    response contains neither text nor tool_use content.
    """
    try:
        blocks = data["content"]
        text = "".join(
            block["text"] for block in blocks if block.get("type") == "text"
        ).strip()
        tool_uses = tuple(
            ToolUse(
                name=block["name"],
                input=block.get("input") or {},
                id=block.get("id"),
            )
            for block in blocks
            if block.get("type") == "tool_use"
        )
        tool_name = tool_uses[0].name if tool_uses else None
        tool_input = tool_uses[0].input if tool_uses else None
        stop_reason = data.get("stop_reason")
    except (KeyError, TypeError) as exc:
        raise GLMAPIError(
            f"Unexpected GLM response payload: {str(data)[:200]}"
        ) from exc
    if not text and tool_name is None:
        raise GLMAPIError("GLM response contained no text or tool_use content.")
    return GLMResponse(
        text=text or None,
        tool_name=tool_name,
        tool_input=tool_input,
        stop_reason=stop_reason,
        tool_uses=tool_uses,
        usage=_parse_usage(data.get("usage")),
    )


def _parse_usage(usage: Any) -> dict[str, int] | None:
    """Keep only integer token counts the provider actually returned."""
    if not isinstance(usage, dict):
        return None
    parsed = {
        key: usage[key]
        for key in ("input_tokens", "output_tokens")
        if isinstance(usage.get(key), int)
    }
    return parsed or None
