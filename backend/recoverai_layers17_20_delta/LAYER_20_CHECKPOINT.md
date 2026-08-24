# RecoverAI Layer 20 Checkpoint

## Layer 20 — Razorpay Webhook Verification, Normalization & Processing

Status: COMPLETE / TESTED

### Added
- Raw-body HMAC-SHA256 webhook signature verification using `RAZORPAY_WEBHOOK_SECRET`.
- `x-razorpay-event-id` idempotency with deterministic body-hash fallback.
- Normalization of Razorpay payment, subscription, and invoice payloads into RecoverAI domain models.
- Webhook receipt persistence in `webhook_events`.
- Automatic recovery-case creation for payment failures, subscription failures, and overdue invoices.
- Recovery/partial-recovery updates for successful payments and invoice events.
- Audit persistence for case creation, diagnosis, payment receipt, and recovery.
- FastAPI `POST /webhooks/razorpay` endpoint.

### Tests
- Layer 20: 16 passed.
- Full backend regression: 227 passed.

### Environment
Add a webhook-specific secret. It is different from the Razorpay API key secret:

```bash
export RAZORPAY_WEBHOOK_SECRET="replace_with_your_webhook_secret"
```

### Database
Layer 20 adds the `webhook_events` table. With the RecoverAI PostgreSQL container on port 5433:

```bash
export DATABASE_URL="postgresql+psycopg://recoverai:recoverai@localhost:5433/recoverai"
python -m app.db.init_db
```

### Test commands

```bash
pytest tests/test_razorpay_webhooks.py -v
pytest -q
```

### Local API smoke test

```bash
export DATABASE_URL="postgresql+psycopg://recoverai:recoverai@localhost:5433/recoverai"
export RAZORPAY_WEBHOOK_SECRET="dev_webhook_secret"
uvicorn app.main:app --reload --port 8000
```

Health endpoint:

```bash
curl http://127.0.0.1:8000/health
```
