# Layer 19 — Razorpay Test Mode Integration

Status: COMPLETE / FROZEN

Added:
- Test-mode-only Razorpay configuration with live-key fail-closed guard
- HTTPX Basic Auth client for Razorpay v1 REST API
- Payment fetch/list APIs
- Subscription fetch API
- Invoice fetch API
- Payment Link create/fetch/cancel APIs
- INR major-unit -> paise conversion for Payment Links
- Typed auth/not-found/API/transport/config errors
- Provider error field preservation for later diagnosis/webhook layers
- Secret redaction in config repr
- Safe entity-id validation
- Credential/API smoke-test CLI
- Optional one-link Test Mode creation in smoke test

Validation:
- Layer 19 tests: 14 passed
- Full backend suite: 211 passed

Important boundary:
- This layer is transport/integration only.
- Razorpay payload normalization/webhook processing remains the next layer.
- Money-action selection and policy guardrails remain deterministic upstream contracts.

Next layer: Layer 20 — Razorpay webhook verification, normalization, and processing.
