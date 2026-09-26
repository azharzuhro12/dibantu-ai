"""WhatsApp Cloud API integration (Step 17).

Two inbound flows converge on the same DibantuAgent:

- the original local simulator (``POST /webhook/whatsapp`` with
  ``{"from": ..., "message": ...}``, kept unchanged for deterministic
  tests), and
- the real Meta WhatsApp Cloud API webhook (the Meta envelope with
  ``object``/``entry``, plus the ``GET`` verification handshake).

Outbound replies for the real flow are sent through the Meta Graph API
(``POST /v{VERSION}/{PHONE_NUMBER_ID}/messages``). Everything is
disabled by default (``WHATSAPP_ENABLED=false``) and no secret is ever
logged, returned, or exposed to the frontend.
"""
