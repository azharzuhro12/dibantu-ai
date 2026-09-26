"""Offline tests for the real WhatsApp Cloud API integration (Step 17).

Everything runs without network, Meta credentials, or a real GLM key:
the Graph API send client is answered by an ``httpx.MockTransport``
(injected through the route's ``_build_cloud_client`` seam), the agent
dependency is overridden with a scripted-GLM ``DibantuAgent`` (the same
seam as the /api/chat tests), and the idempotency ledger runs on the
PostgreSQL scratch database. The local simulator payload keeps its own
tests in ``tests/test_webhook_whatsapp.py``; here the shared endpoint
must tell the two payload shapes apart.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agent.agent import DibantuAgent
from app.api import routes as api_routes
from app.api.routes import get_agent
from app.config import Settings
from app.db import database
from app.db.models import WhatsAppEventRecord
from app.main import app
from app.whatsapp import repository as whatsapp_repository
from app.whatsapp.client import (
    WhatsAppCloudClient,
    WhatsAppConfigError,
    WhatsAppSendError,
)
from app.whatsapp.config import WhatsAppConfig
from app.whatsapp.formatting import chunk_text, format_reply_for_whatsapp
from app.whatsapp.schemas import (
    IncomingMessage,
    WebhookContact,
    WebhookProfile,
)
from app.whatsapp.webhook import extract_text_messages


@pytest.fixture(autouse=True)
def _business_database(test_database):
    """Ledger writes run against the PostgreSQL scratch database."""
    yield


SENDER = "628123456789"
MESSAGE_ID = "wamid.TESTabc123"


# ---------------------------------------------------------------------------
# Payload builders (shapes follow Meta's webhook components documentation)
# ---------------------------------------------------------------------------


def meta_envelope(*messages: dict[str, Any], statuses: list[dict] | None = None) -> dict:
    """A Meta webhook payload with the given raw ``value.messages``."""
    value: dict[str, Any] = {
        "messaging_product": "whatsapp",
        "metadata": {"display_phone_number": "628111000222", "phone_number_id": "PNID1"},
        "contacts": [
            {"profile": {"name": "Sari Dewi"}, "wa_id": SENDER},
        ],
    }
    if messages:
        value["messages"] = list(messages)
    if statuses:
        value["statuses"] = statuses
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WBAID1",
                "changes": [{"field": "messages", "value": value}],
            }
        ],
    }


def text_message(body: str, *, message_id: str = MESSAGE_ID, sender: str = SENDER) -> dict:
    return {
        "from": sender,
        "id": message_id,
        "timestamp": "1770000000",
        "type": "text",
        "text": {"body": body},
    }


def image_message() -> dict:
    return {
        "from": SENDER,
        "id": "wamid.IMG1",
        "timestamp": "1770000000",
        "type": "image",
        "image": {"id": "MEDIA1", "mime_type": "image/jpeg"},
    }


def status_update() -> dict:
    return {"id": "wamid.OUT1", "status": "delivered", "timestamp": "1770000001"}


# ---------------------------------------------------------------------------
# Scripted doubles
# ---------------------------------------------------------------------------


def make_settings(api_key: str = "test-key") -> Settings:
    return Settings(
        glm_api_key=api_key,
        glm_base_url="https://glm.test/api/anthropic",
        glm_model="glm-test",
    )


def scripted_agent(bodies: list[dict[str, Any]]) -> DibantuAgent:
    """A DibantuAgent whose GLM exchanges replay queued response bodies."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=bodies.pop(0))

    return DibantuAgent(
        make_settings(), transport=httpx.MockTransport(handler)
    )


def glm_text_body(text: str) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
    }


class RecordingGraph:
    """MockTransport handler that records sends and answers success."""

    def __init__(self, replies: list[httpx.Response] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._replies = list(replies or [])

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._replies:
            return self._replies.pop(0)
        return httpx.Response(
            200, json={"messages": [{"id": f"wamid.out.{len(self.requests)}"}]}
        )

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


def enable_whatsapp(
    monkeypatch: pytest.MonkeyPatch,
    *,
    send: bool = True,
    app_secret: str | None = None,
) -> None:
    """Switch the integration on (and optionally fully configured to send)."""
    monkeypatch.setenv("WHATSAPP_ENABLED", "true")
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "verify-secret")
    if send:
        monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "graph-secret-token")
        monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "123456789012345")
    if app_secret is not None:
        monkeypatch.setenv("WHATSAPP_APP_SECRET", app_secret)


def send_config() -> WhatsAppConfig:
    return WhatsAppConfig(
        enabled=True,
        verify_token="verify-secret",
        access_token="graph-secret-token",
        phone_number_id="123456789012345",
    )


def post_meta(
    payload: dict[str, Any] | bytes,
    *,
    agent: DibantuAgent,
    graph: RecordingGraph | None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """POST a Meta envelope (dict, or exact raw bytes) with everything overridden.

    Passing ``bytes`` sends those exact bytes as the body — required for
    signature tests, which bind to the raw payload byte-for-byte.
    """
    app.dependency_overrides[get_agent] = lambda: agent

    def build_client(config: WhatsAppConfig) -> WhatsAppCloudClient | None:
        if graph is None:
            return None
        return WhatsAppCloudClient(
            config, transport=httpx.MockTransport(graph.handler)
        )

    original = api_routes._build_cloud_client
    api_routes._build_cloud_client = build_client
    try:
        with TestClient(app) as client:
            if isinstance(payload, bytes):
                merged = {"Content-Type": "application/json", **(headers or {})}
                return client.post("/webhook/whatsapp", content=payload, headers=merged)
            return client.post("/webhook/whatsapp", json=payload, headers=headers)
    finally:
        api_routes._build_cloud_client = original
        app.dependency_overrides.pop(get_agent, None)


def ledger_row(message_id: str) -> WhatsAppEventRecord | None:
    with database.session_scope() as session:
        return (
            session.query(WhatsAppEventRecord)
            .filter(WhatsAppEventRecord.message_id == message_id)
            .one_or_none()
        )


# ---------------------------------------------------------------------------
# GET verification handshake
# ---------------------------------------------------------------------------


def verify_webhook(**params: str) -> httpx.Response:
    with TestClient(app) as client:
        return client.get("/webhook/whatsapp", params=params)


def test_verification_echoes_challenge_when_token_matches(monkeypatch) -> None:
    enable_whatsapp(monkeypatch)

    response = verify_webhook(
        **{
            "hub.mode": "subscribe",
            "hub.verify_token": "verify-secret",
            "hub.challenge": "challenge-abc-123",
        }
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "challenge-abc-123"


def test_verification_rejects_wrong_token(monkeypatch) -> None:
    enable_whatsapp(monkeypatch)

    response = verify_webhook(
        **{
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong",
            "hub.challenge": "challenge-abc-123",
        }
    )

    assert response.status_code == 403
    # The configured token is never echoed back, only a plain failure.
    assert "verify-secret" not in response.text


def test_verification_rejects_wrong_mode(monkeypatch) -> None:
    enable_whatsapp(monkeypatch)

    response = verify_webhook(
        **{
            "hub.mode": "other",
            "hub.verify_token": "verify-secret",
            "hub.challenge": "challenge-abc-123",
        }
    )

    assert response.status_code == 403


def test_verification_rejects_when_integration_disabled(monkeypatch) -> None:
    monkeypatch.setenv("WHATSAPP_ENABLED", "false")
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "verify-secret")

    response = verify_webhook(
        **{
            "hub.mode": "subscribe",
            "hub.verify_token": "verify-secret",
            "hub.challenge": "challenge-abc-123",
        }
    )

    assert response.status_code == 403


def test_verification_rejects_empty_token_when_unconfigured(monkeypatch) -> None:
    monkeypatch.setenv("WHATSAPP_ENABLED", "true")
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "")

    response = verify_webhook(
        **{
            "hub.mode": "subscribe",
            "hub.verify_token": "",
            "hub.challenge": "challenge-abc-123",
        }
    )

    assert response.status_code == 403


# ---------------------------------------------------------------------------
# POST webhook: envelope routing, acknowledgement, idempotency
# ---------------------------------------------------------------------------


def test_disabled_integration_acks_without_processing(monkeypatch) -> None:
    monkeypatch.setenv("WHATSAPP_ENABLED", "false")
    agent = scripted_agent([glm_text_body("x")])

    response = post_meta(
        meta_envelope(text_message("Halo", message_id="wamid.T01")),
        agent=agent,
        graph=None,
    )

    assert response.status_code == 200
    assert response.json() == {"status": "disabled", "new_messages": 0, "duplicates": 0}
    assert ledger_row("wamid.T01") is None


def test_text_message_processed_and_replied(monkeypatch) -> None:
    enable_whatsapp(monkeypatch)
    agent = scripted_agent([glm_text_body("**Stok** Kopi Susu: 24")])
    graph = RecordingGraph()

    response = post_meta(
        meta_envelope(text_message("Cek stok kopi susu", message_id="wamid.T02")),
        agent=agent,
        graph=graph,
    )

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "new_messages": 1, "duplicates": 0}
    # Background task ran before TestClient returned: reply formatted for
    # WhatsApp (** → *) and sent through the Graph API to the sender.
    assert len(graph.bodies) == 1
    assert graph.bodies[0]["to"] == SENDER
    assert graph.bodies[0]["text"]["body"] == "*Stok* Kopi Susu: 24"
    row = ledger_row("wamid.T02")
    assert row is not None and row.status == "processed"
    assert row.sender_id == SENDER and row.processed_at is not None


def test_duplicate_redelivery_never_replays_agent_or_send(monkeypatch) -> None:
    enable_whatsapp(monkeypatch)
    agent = scripted_agent([glm_text_body("Oke")])
    graph = RecordingGraph()

    envelope = meta_envelope(text_message("Halo", message_id="wamid.T03"))

    first = post_meta(envelope, agent=agent, graph=graph)
    second = post_meta(envelope, agent=agent, graph=graph)

    assert first.json()["new_messages"] == 1
    assert second.status_code == 200
    assert second.json() == {
        "status": "accepted",
        "new_messages": 0,
        "duplicates": 1,
    }
    # Exactly one send, one ledger row — the duplicate claimed nothing.
    assert len(graph.bodies) == 1
    row = ledger_row("wamid.T03")
    assert row is not None and row.status == "processed"


def test_statuses_only_envelope_is_ignored(monkeypatch) -> None:
    enable_whatsapp(monkeypatch)
    agent = scripted_agent([glm_text_body("x")])
    graph = RecordingGraph()

    response = post_meta(
        meta_envelope(statuses=[status_update()]), agent=agent, graph=graph
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ignored", "new_messages": 0, "duplicates": 0}
    assert graph.bodies == []


def test_non_text_message_is_ignored(monkeypatch) -> None:
    enable_whatsapp(monkeypatch)
    agent = scripted_agent([glm_text_body("x")])
    graph = RecordingGraph()

    response = post_meta(meta_envelope(image_message()), agent=agent, graph=graph)

    assert response.status_code == 200
    assert response.json() == {"status": "ignored", "new_messages": 0, "duplicates": 0}


def test_multiple_messages_in_one_envelope(monkeypatch) -> None:
    enable_whatsapp(monkeypatch)
    agent = scripted_agent(
        [glm_text_body("Pertama"), glm_text_body("Kedua")]
    )
    graph = RecordingGraph()

    response = post_meta(
        meta_envelope(
            text_message("satu", message_id="wamid.M1"),
            text_message("dua", message_id="wamid.M2"),
        ),
        agent=agent,
        graph=graph,
    )

    assert response.status_code == 200
    assert response.json()["new_messages"] == 2
    assert [b["text"]["body"] for b in graph.bodies] == ["Pertama", "Kedua"]
    assert ledger_row("wamid.M1").status == "processed"
    assert ledger_row("wamid.M2").status == "processed"


def test_send_unconfigured_marks_ledger_failed(monkeypatch) -> None:
    enable_whatsapp(monkeypatch, send=False)
    agent = scripted_agent([glm_text_body("Oke")])

    response = post_meta(
        meta_envelope(text_message("Halo", message_id="wamid.T04")),
        agent=agent,
        graph=None,
    )

    assert response.status_code == 200
    assert response.json()["new_messages"] == 1
    assert ledger_row("wamid.T04").status == "failed"


def test_send_http_error_marks_ledger_failed(monkeypatch) -> None:
    enable_whatsapp(monkeypatch)
    agent = scripted_agent([glm_text_body("Oke")])
    graph = RecordingGraph(replies=[httpx.Response(500, json={"error": {"message": "boom"}})])

    response = post_meta(
        meta_envelope(text_message("Halo", message_id="wamid.T05")),
        agent=agent,
        graph=graph,
    )

    assert response.status_code == 200  # Meta still gets its ack
    assert ledger_row("wamid.T05").status == "failed"


def test_agent_failure_marks_ledger_failed(monkeypatch) -> None:
    enable_whatsapp(monkeypatch)
    agent = DibantuAgent(
        make_settings(),
        transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")),
    )
    graph = RecordingGraph()

    response = post_meta(
        meta_envelope(text_message("Halo", message_id="wamid.T06")),
        agent=agent,
        graph=graph,
    )

    assert response.status_code == 200
    assert ledger_row("wamid.T06").status == "failed"
    assert graph.bodies == []  # nothing to send when the run failed


def test_structurally_invalid_meta_envelope_is_422(monkeypatch) -> None:
    enable_whatsapp(monkeypatch)
    agent = scripted_agent([glm_text_body("x")])

    response = post_meta({"object": "x", "entry": []}, agent=agent, graph=None)

    assert response.status_code == 422
    assert ledger_row("wamid.NEVER") is None


def test_simulator_payload_still_answered_synchronously(monkeypatch) -> None:
    """The shared endpoint keeps telling the two payload shapes apart."""
    enable_whatsapp(monkeypatch)
    agent = scripted_agent([glm_text_body("Halo! Saya DibantuAI.")])
    app.dependency_overrides[get_agent] = lambda: agent
    try:
        with TestClient(app) as client:
            response = client.post(
                "/webhook/whatsapp", json={"from": SENDER, "message": "Halo"}
            )
    finally:
        app.dependency_overrides.pop(get_agent, None)

    assert response.status_code == 200
    assert response.json()["to"] == SENDER
    assert response.json()["response"] == "Halo! Saya DibantuAI."
    assert ledger_row("wamid.NOSIM") is None  # simulator bypasses the ledger


def test_non_json_body_is_422(monkeypatch) -> None:
    enable_whatsapp(monkeypatch)
    app.dependency_overrides[get_agent] = lambda: scripted_agent(
        [glm_text_body("x")]
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/webhook/whatsapp",
                content=b"not-json{",
                headers={"Content-Type": "application/json"},
            )
    finally:
        app.dependency_overrides.pop(get_agent, None)

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# X-Hub-Signature-256 validation (Meta's documented webhook security)
# ---------------------------------------------------------------------------

APP_SECRET = "meta-app-secret-for-tests"


def meta_signature(raw: bytes, secret: str = APP_SECRET) -> str:
    """Compute X-Hub-Signature-256 independently from the implementation.

    Mirrors Meta's documented construction with the stdlib directly, so
    the tests authenticate against the documented algorithm rather than
    against the code under test.
    """
    import hashlib
    import hmac as hmac_std

    digest = hmac_std.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def raw_envelope(message_id: str) -> bytes:
    """The exact bytes a Meta delivery would carry (compact JSON)."""
    return json.dumps(
        meta_envelope(text_message("Cek stok kopi susu", message_id=message_id))
    ).encode("utf-8")


def test_valid_signature_is_accepted_and_processed(monkeypatch) -> None:
    enable_whatsapp(monkeypatch, app_secret=APP_SECRET)
    agent = scripted_agent([glm_text_body("Stok Kopi Susu: 23.")])
    graph = RecordingGraph()
    raw = raw_envelope("wamid.S01")

    response = post_meta(
        raw,
        agent=agent,
        graph=graph,
        headers={"X-Hub-Signature-256": meta_signature(raw)},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "new_messages": 1, "duplicates": 0}
    assert ledger_row("wamid.S01").status == "processed"
    assert len(graph.bodies) == 1  # the reply was sent


def test_invalid_signature_is_rejected_before_processing(monkeypatch) -> None:
    enable_whatsapp(monkeypatch, app_secret=APP_SECRET)
    agent = scripted_agent([glm_text_body("x")])
    graph = RecordingGraph()
    raw = raw_envelope("wamid.S02")

    response = post_meta(
        raw,
        agent=agent,
        graph=graph,
        headers={"X-Hub-Signature-256": meta_signature(b"different bytes")},
    )

    assert response.status_code == 403
    assert "signature" in response.json()["detail"]
    assert ledger_row("wamid.S02") is None  # never claimed
    assert graph.bodies == []  # no agent run, no send


def test_missing_signature_header_rejected_when_required(monkeypatch) -> None:
    enable_whatsapp(monkeypatch, app_secret=APP_SECRET)
    agent = scripted_agent([glm_text_body("x")])
    graph = RecordingGraph()

    response = post_meta(raw_envelope("wamid.S03"), agent=agent, graph=graph)

    assert response.status_code == 403
    assert ledger_row("wamid.S03") is None
    assert graph.bodies == []


def test_malformed_signature_values_are_rejected(monkeypatch) -> None:
    enable_whatsapp(monkeypatch, app_secret=APP_SECRET)
    agent = scripted_agent([glm_text_body("x")])
    graph = RecordingGraph()
    malformed = [
        "",  # empty header
        "sha256=",  # prefix without digest
        "sha256=not-hex-at-all!!",  # non-hex digest
        "deadbeef",  # missing sha256= prefix
        "sha256=abc123",  # valid hex, wrong length
        "sha1=abcdef0123456789",  # wrong algorithm prefix
    ]

    for i, header in enumerate(malformed):
        response = post_meta(
            raw_envelope(f"wamid.S04{i}"),
            agent=agent,
            graph=graph,
            headers={"X-Hub-Signature-256": header},
        )
        assert response.status_code == 403, header

    assert graph.bodies == []
    assert ledger_row("wamid.S040") is None


def test_signature_binds_to_raw_bytes_not_reparsed_json(monkeypatch) -> None:
    """A digest over a re-serialization of the same JSON must be rejected.

    Meta signs the exact delivered bytes; validating against a
    re-parsed/re-serialized payload (different whitespace/order) would
    accept forged bodies.
    """
    enable_whatsapp(monkeypatch, app_secret=APP_SECRET)
    agent = scripted_agent([glm_text_body("x")])
    graph = RecordingGraph()
    envelope = meta_envelope(text_message("Cek stok", message_id="wamid.S05"))

    # Same JSON, different bytes (pretty-printed) → different signature.
    reparsed = json.dumps(envelope, indent=2).encode("utf-8")

    response = post_meta(
        json.dumps(envelope).encode("utf-8"),  # compact bytes delivered
        agent=agent,
        graph=graph,
        headers={"X-Hub-Signature-256": meta_signature(reparsed)},
    )

    assert response.status_code == 403
    assert ledger_row("wamid.S05") is None
    assert graph.bodies == []


def test_simulator_still_works_without_signature(monkeypatch) -> None:
    """The local simulator ignores signature enforcement entirely."""
    enable_whatsapp(monkeypatch, app_secret=APP_SECRET)
    agent = scripted_agent([glm_text_body("Halo! Saya DibantuAI.")])
    app.dependency_overrides[get_agent] = lambda: agent
    try:
        with TestClient(app) as client:
            response = client.post(
                "/webhook/whatsapp", json={"from": SENDER, "message": "Halo"}
            )
    finally:
        app.dependency_overrides.pop(get_agent, None)

    assert response.status_code == 200
    assert response.json()["response"] == "Halo! Saya DibantuAI."


def test_signature_not_required_without_app_secret(monkeypatch) -> None:
    """No app secret configured → validation stays off (back-compat)."""
    enable_whatsapp(monkeypatch)  # no app_secret
    agent = scripted_agent([glm_text_body("Oke.")])
    graph = RecordingGraph()

    response = post_meta(
        meta_envelope(text_message("Halo", message_id="wamid.S07")),
        agent=agent,
        graph=graph,
    )

    assert response.status_code == 200
    assert response.json()["new_messages"] == 1


def test_disabled_integration_does_not_demand_signature(monkeypatch) -> None:
    """Disabled integration acks 'disabled' instead of 403 (no Meta retry loop)."""
    monkeypatch.setenv("WHATSAPP_ENABLED", "false")
    monkeypatch.setenv("WHATSAPP_APP_SECRET", APP_SECRET)
    agent = scripted_agent([glm_text_body("x")])
    graph = RecordingGraph()

    response = post_meta(
        meta_envelope(text_message("Halo", message_id="wamid.S08")),
        agent=agent,
        graph=graph,
    )

    assert response.status_code == 200
    assert response.json()["status"] == "disabled"
    assert ledger_row("wamid.S08") is None


def test_signature_helper_unit_contract() -> None:
    from app.whatsapp.signature import compute_signature, is_valid_signature

    raw = b'{"object":"whatsapp_business_account"}'
    expected = meta_signature(raw)  # independent stdlib computation

    assert compute_signature(raw, APP_SECRET) == expected
    assert is_valid_signature(raw, expected, APP_SECRET) is True
    # Missing secret / missing header / wrong secret all fail.
    assert is_valid_signature(raw, expected, "") is False
    assert is_valid_signature(raw, None, APP_SECRET) is False
    assert is_valid_signature(raw, meta_signature(raw, "other-secret"), APP_SECRET) is False


# ---------------------------------------------------------------------------
# Parser (extract_text_messages)
# ---------------------------------------------------------------------------


def parsed(payload: dict[str, Any]) -> list[IncomingMessage]:
    from app.whatsapp.schemas import MetaWebhookEvent

    return extract_text_messages(MetaWebhookEvent.model_validate(payload))


def test_parser_extracts_text_message_with_profile_name() -> None:
    messages = parsed(meta_envelope(text_message("Halo")))

    assert messages == [
        IncomingMessage(
            message_id=MESSAGE_ID,
            sender=SENDER,
            text="Halo",
            profile_name="Sari Dewi",
        )
    ]


def test_parser_drops_non_text_statuses_and_empty() -> None:
    assert parsed(meta_envelope(image_message())) == []
    assert parsed(meta_envelope(statuses=[status_update()])) == []
    assert parsed(meta_envelope()) == []
    assert parsed(meta_envelope({"from": "", "id": "wamid.X", "type": "text", "text": {"body": "halo"}})) == []
    assert (
        parsed(meta_envelope(text_message("   "))) == []
    )  # whitespace-only body


def test_parser_ignores_extra_and_unknown_fields() -> None:
    payload = meta_envelope(text_message("Halo"))
    payload["entry"][0]["changes"][0]["value"]["unknown_future_field"] = {"x": 1}
    payload["entry"][0]["random_new_block"] = True

    assert [m.text for m in parsed(payload)] == ["Halo"]


def test_parser_profile_name_defaults_without_contacts() -> None:
    payload = meta_envelope(text_message("Halo"))
    payload["entry"][0]["changes"][0]["value"]["contacts"] = []

    assert parsed(payload)[0].profile_name is None


# ---------------------------------------------------------------------------
# Formatting (format_reply_for_whatsapp / chunk_text)
# ---------------------------------------------------------------------------


def test_formatting_bold_headings_bullets_links() -> None:
    text = "## Menu Hari Ini\n- **Kopi Susu** (Rp18.000)\nLihat [menu](https://dibantu.id/menu)"

    assert format_reply_for_whatsapp(text) == (
        "Menu Hari Ini\n• *Kopi Susu* (Rp18.000)\nLihat menu (https://dibantu.id/menu)"
    )


def test_formatting_table_becomes_header_value_pairs() -> None:
    text = "| Produk | Stok |\n|---|---|\n| Kopi Susu | 24 |\n| Matcha | 3 |"

    assert format_reply_for_whatsapp(text) == (
        "Produk: Kopi Susu · Stok: 24\nProduk: Matcha · Stok: 3"
    )


def test_formatting_leaves_plain_text_unchanged() -> None:
    assert format_reply_for_whatsapp("Stok aman.") == "Stok aman."


def test_chunk_text_short_is_single_chunk() -> None:
    assert chunk_text("halo", limit=100) == ["halo"]


def test_chunk_text_is_lossless_and_bounded() -> None:
    text = "\n\n".join(f"paragraph {i} " + "x" * 50 for i in range(50))

    chunks = chunk_text(text, limit=200)

    assert all(len(c) <= 200 for c in chunks)
    assert "".join(chunks) == text  # lossless
    assert len(chunks) > 1


def test_chunk_text_hard_slices_unbreakable_text() -> None:
    text = "y" * 500

    chunks = chunk_text(text, limit=200)

    assert chunks == ["y" * 200, "y" * 200, "y" * 100]


# ---------------------------------------------------------------------------
# Idempotency repository
# ---------------------------------------------------------------------------


def test_claim_message_once_then_duplicate() -> None:
    with database.session_scope() as session:
        assert whatsapp_repository.claim_message(
            session, message_id="wamid.R1", sender_id=SENDER
        )
    with database.session_scope() as session:
        assert not whatsapp_repository.claim_message(
            session, message_id="wamid.R1", sender_id=SENDER
        )


def test_mark_processed_and_failed_update_status() -> None:
    with database.session_scope() as session:
        assert whatsapp_repository.claim_message(
            session, message_id="wamid.R2", sender_id=SENDER
        )
        whatsapp_repository.mark_processed(session, "wamid.R2")
    row = ledger_row("wamid.R2")
    assert row.status == "processed" and row.processed_at is not None

    with database.session_scope() as session:
        assert whatsapp_repository.claim_message(
            session, message_id="wamid.R3", sender_id=SENDER
        )
        whatsapp_repository.mark_failed(session, "wamid.R3")
    assert ledger_row("wamid.R3").status == "failed"


# ---------------------------------------------------------------------------
# Graph API send client
# ---------------------------------------------------------------------------


def test_client_sends_documented_shape() -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"messages": [{"id": "wamid.OUT9"}]})

    client = WhatsAppCloudClient(
        send_config(), transport=httpx.MockTransport(handler)
    )

    import asyncio

    message_id = asyncio.run(client.send_text(SENDER, "Halo"))
    asyncio.run(client.aclose())

    assert message_id == "wamid.OUT9"
    (request,) = sent
    assert request.url.path == "/v25.0/123456789012345/messages"
    assert request.headers["Authorization"] == "Bearer graph-secret-token"
    body = json.loads(request.content)
    assert body == {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": SENDER,
        "type": "text",
        "text": {"body": "Halo"},
    }


def test_client_requires_configuration() -> None:
    client = WhatsAppCloudClient(WhatsAppConfig(enabled=True))

    import asyncio

    with pytest.raises(WhatsAppConfigError):
        asyncio.run(client.send_text(SENDER, "Halo"))
    asyncio.run(client.aclose())


def test_client_http_error_is_safe_and_token_free() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, json={"error": {"message": "Temporary send failure"}}
        )

    client = WhatsAppCloudClient(
        send_config(), transport=httpx.MockTransport(handler)
    )

    import asyncio

    with pytest.raises(WhatsAppSendError) as excinfo:
        asyncio.run(client.send_text(SENDER, "Halo"))
    asyncio.run(client.aclose())

    message = str(excinfo.value)
    assert "403" in message and "Temporary send failure" in message
    assert "graph-secret-token" not in message  # never the token


def test_client_timeout_is_safe() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    client = WhatsAppCloudClient(
        WhatsAppConfig(
            enabled=True,
            access_token="graph-secret-token",
            phone_number_id="123456789012345",
            timeout_seconds=0.01,
        ),
        transport=httpx.MockTransport(handler),
    )

    import asyncio

    with pytest.raises(WhatsAppSendError, match="timed out"):
        asyncio.run(client.send_text(SENDER, "Halo"))
    asyncio.run(client.aclose())
