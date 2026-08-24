from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect

from app.api.webhooks import get_db_session
from app.db import (
    AuditEventRepository,
    InvoiceRepository,
    PaymentRepository,
    RecoveryCaseRepository,
    SubscriptionRepository,
    WebhookEventRepository,
    build_engine,
    build_session_factory,
    create_schema,
    session_scope,
)
from app.domain.enums import (
    CaseType,
    FailureClass,
    InvoiceStatus,
    PaymentStatus,
    RecoveryCaseStatus,
    SubscriptionStatus,
)
from app.domain.recovery_case import RecoveryCase
from app.integrations.razorpay_webhooks import (
    RazorpayWebhookPayloadError,
    RazorpayWebhookSettings,
    RazorpayWebhookSignatureError,
    derive_event_id,
    normalize_razorpay_webhook,
    parse_razorpay_webhook,
    verify_razorpay_webhook_signature,
)
from app.main import app
from app.services.webhook_processor import RazorpayWebhookProcessor


SECRET = "whsec_recoverai_test_only"
CREATED_AT = 1787495400


@pytest.fixture()
def db(tmp_path):
    engine = build_engine(f"sqlite+pysqlite:///{tmp_path / 'webhooks.db'}")
    create_schema(engine)
    factory = build_session_factory(engine)
    try:
        yield engine, factory
    finally:
        engine.dispose()


def raw_event(event_type: str, payload: dict, *, account_id: str = "acc_test") -> bytes:
    return json.dumps(
        {
            "entity": "event",
            "account_id": account_id,
            "event": event_type,
            "contains": list(payload),
            "payload": payload,
            "created_at": CREATED_AT,
        },
        separators=(",", ":"),
    ).encode()


def sign(body: bytes) -> str:
    return hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


def payment_entity(
    *,
    payment_id: str = "pay_failed_1",
    status: str = "failed",
    amount: int = 125050,
    reason: str | None = "insufficient_funds",
    notes: dict | None = None,
) -> dict:
    entity = {
        "id": payment_id,
        "entity": "payment",
        "amount": amount,
        "currency": "INR",
        "status": status,
        "order_id": "order_1",
        "customer_id": "cust_1",
        "method": "card",
        "bank": "HDFC",
        "error_code": "BAD_REQUEST_ERROR" if status == "failed" else None,
        "error_source": "customer" if status == "failed" else None,
        "error_step": "payment_authorization" if status == "failed" else None,
        "error_reason": reason if status == "failed" else None,
        "error_description": "Payment failed" if status == "failed" else None,
        "created_at": CREATED_AT - 10,
    }
    if notes is not None:
        entity["notes"] = notes
    return {"payment": {"entity": entity}}


def subscription_entity(
    *,
    status: str,
    subscription_id: str = "sub_1",
    include_payment: bool = True,
) -> dict:
    payload = {
        "subscription": {
            "entity": {
                "id": subscription_id,
                "entity": "subscription",
                "customer_id": "cust_1",
                "status": status,
                "auth_attempts": 3,
                "current_start": CREATED_AT - 86400,
                "current_end": CREATED_AT + 2592000,
                "charge_at": CREATED_AT + 86400,
                "created_at": CREATED_AT - 2592000,
            }
        }
    }
    if include_payment:
        payload.update(
            payment_entity(
                payment_id="pay_sub_1",
                status="failed" if status in {"pending", "halted"} else "captured",
                amount=99900,
                reason="insufficient_funds",
            )
        )
    return payload


def invoice_entity(
    *,
    status: str,
    invoice_id: str = "inv_1",
    amount_due: int = 500000,
    amount_paid: int = 0,
) -> dict:
    return {
        "invoice": {
            "entity": {
                "id": invoice_id,
                "entity": "invoice",
                "customer_id": "cust_1",
                "status": status,
                "amount": 500000,
                "amount_due": amount_due,
                "amount_paid": amount_paid,
                "currency": "INR",
                "issued_at": CREATED_AT - 10 * 86400,
                "expire_by": CREATED_AT - 2 * 86400,
                "created_at": CREATED_AT - 10 * 86400,
            }
        }
    }


def process(factory, body: bytes, event_id: str):
    with session_scope(factory) as session:
        return RazorpayWebhookProcessor(
            session,
            settings=RazorpayWebhookSettings(SECRET),
        ).process(body, signature=sign(body), razorpay_event_id=event_id)


def test_layer20_schema_adds_webhook_event_table(db):
    engine, _ = db
    assert "webhook_events" in set(inspect(engine).get_table_names())


def test_signature_verification_uses_exact_raw_body():
    body = raw_event("payment.failed", payment_entity())
    verify_razorpay_webhook_signature(body, sign(body), SECRET)

    reparsed = json.dumps(json.loads(body), indent=2).encode()
    with pytest.raises(RazorpayWebhookSignatureError):
        verify_razorpay_webhook_signature(reparsed, sign(body), SECRET)


def test_missing_or_invalid_signature_is_rejected():
    body = raw_event("payment.failed", payment_entity())
    with pytest.raises(RazorpayWebhookSignatureError):
        verify_razorpay_webhook_signature(body, None, SECRET)
    with pytest.raises(RazorpayWebhookSignatureError):
        verify_razorpay_webhook_signature(body, "wrong", SECRET)


def test_event_id_falls_back_to_stable_body_hash():
    body = b'{"event":"payment.failed"}'
    event_id = derive_event_id(body, None)
    assert event_id == f"body_{hashlib.sha256(body).hexdigest()}"


def test_signed_malformed_json_fails_after_signature_verification():
    body = b"not-json"
    verify_razorpay_webhook_signature(body, sign(body), SECRET)
    with pytest.raises(RazorpayWebhookPayloadError):
        parse_razorpay_webhook(body, event_id="evt_bad")


def test_payment_failed_normalizes_and_creates_diagnosed_case(db):
    _, factory = db
    body = raw_event("payment.failed", payment_entity())
    result = process(factory, body, "evt_payment_failed_1")

    assert result.status == "processed"
    assert result.case_id is not None

    with session_scope(factory) as session:
        payment = PaymentRepository(session).get("pay_failed_1")
        case = RecoveryCaseRepository(session).get(result.case_id)
        audits = AuditEventRepository(session).list_for_case(result.case_id)
        receipt = WebhookEventRepository(session).get("evt_payment_failed_1")

        assert payment is not None and payment.status == PaymentStatus.FAILED
        assert payment.amount == Decimal("1250.50")
        assert case is not None and case.case_type == CaseType.PAYMENT_FAILURE
        assert case.failure_class == FailureClass.INSUFFICIENT_FUNDS
        assert case.status == RecoveryCaseStatus.DIAGNOSED
        assert len(audits) == 2
        assert receipt is not None and receipt.signature_verified is True
        assert receipt.case_id == case.id


def test_duplicate_event_id_is_idempotent(db):
    _, factory = db
    body = raw_event("payment.failed", payment_entity())
    first = process(factory, body, "evt_duplicate")
    second = process(factory, body, "evt_duplicate")

    assert second.duplicate is True
    assert second.case_id == first.case_id

    with session_scope(factory) as session:
        assert len(WebhookEventRepository(session).list_recent()) == 1
        assert len(AuditEventRepository(session).list_for_case(first.case_id)) == 2


def test_payment_captured_recovers_case_via_recoverai_note(db):
    _, factory = db
    case = RecoveryCase(
        id="rc_recover_me",
        merchant_id="acc_test",
        customer_id="cust_1",
        case_type=CaseType.PAYMENT_FAILURE,
        status=RecoveryCaseStatus.WAITING_CUSTOMER,
        amount_at_risk=Decimal("1250.50"),
        payment_id="pay_original_failed",
    )
    with session_scope(factory) as session:
        RecoveryCaseRepository(session).save(case)

    body = raw_event(
        "payment.captured",
        payment_entity(
            payment_id="pay_recovery_success",
            status="captured",
            amount=125050,
            reason=None,
            notes={"recoverai_case_id": case.id},
        ),
    )
    result = process(factory, body, "evt_captured_1")

    assert result.case_id == case.id
    with session_scope(factory) as session:
        loaded = RecoveryCaseRepository(session).get(case.id)
        assert loaded.status == RecoveryCaseStatus.RECOVERED
        assert loaded.recovered_amount == Decimal("1250.50")
        assert loaded.closed_at is not None
        assert len(AuditEventRepository(session).list_for_case(case.id)) == 2


def test_unmatched_success_event_is_safely_ignored(db):
    _, factory = db
    body = raw_event(
        "payment.captured",
        payment_entity(payment_id="pay_unmatched", status="captured", reason=None),
    )
    result = process(factory, body, "evt_unmatched")
    assert result.status == "ignored"
    assert result.case_id is None
    with session_scope(factory) as session:
        assert PaymentRepository(session).get("pay_unmatched") is not None


def test_subscription_pending_creates_subscription_failure_case(db):
    _, factory = db
    body = raw_event("subscription.pending", subscription_entity(status="pending"))
    result = process(factory, body, "evt_sub_pending")

    with session_scope(factory) as session:
        sub = SubscriptionRepository(session).get("sub_1")
        case = RecoveryCaseRepository(session).get(result.case_id)
        assert sub.status == SubscriptionStatus.PENDING
        assert sub.amount == Decimal("999.00")
        assert case.case_type == CaseType.SUBSCRIPTION_FAILURE
        assert case.subscription_id == "sub_1"
        assert case.amount_at_risk == Decimal("999.00")


def test_subscription_charged_recovers_open_subscription_case(db):
    _, factory = db
    pending = raw_event("subscription.pending", subscription_entity(status="pending"))
    pending_result = process(factory, pending, "evt_sub_pending_2")

    charged = raw_event("subscription.charged", subscription_entity(status="active"))
    result = process(factory, charged, "evt_sub_charged")
    assert result.case_id == pending_result.case_id

    with session_scope(factory) as session:
        case = RecoveryCaseRepository(session).get(result.case_id)
        assert case.status == RecoveryCaseStatus.RECOVERED
        assert case.recovered_amount == Decimal("999.00")


def test_invoice_expired_creates_overdue_invoice_case(db):
    _, factory = db
    body = raw_event("invoice.expired", invoice_entity(status="expired"))
    result = process(factory, body, "evt_invoice_expired")

    with session_scope(factory) as session:
        invoice = InvoiceRepository(session).get("inv_1")
        case = RecoveryCaseRepository(session).get(result.case_id)
        assert invoice.status == InvoiceStatus.OVERDUE
        assert invoice.days_overdue == 2
        assert case.case_type == CaseType.OVERDUE_INVOICE
        assert case.failure_class == FailureClass.OVERDUE_RECEIVABLE
        assert case.amount_at_risk == Decimal("5000.00")


def test_invoice_partial_payment_updates_amount_at_risk(db):
    _, factory = db
    expired = raw_event("invoice.expired", invoice_entity(status="expired"))
    first = process(factory, expired, "evt_invoice_expired_2")

    partial = raw_event(
        "invoice.partially_paid",
        invoice_entity(status="partially_paid", amount_due=300000, amount_paid=200000),
    )
    second = process(factory, partial, "evt_invoice_partial")
    assert second.case_id == first.case_id

    with session_scope(factory) as session:
        case = RecoveryCaseRepository(session).get(first.case_id)
        assert case.amount_at_risk == Decimal("5000.00")
        assert case.recovered_amount == Decimal("2000.00")
        assert case.status != RecoveryCaseStatus.RECOVERED


def test_invoice_paid_recovers_existing_case(db):
    _, factory = db
    expired = raw_event("invoice.expired", invoice_entity(status="expired"))
    first = process(factory, expired, "evt_invoice_expired_3")

    paid = raw_event(
        "invoice.paid",
        invoice_entity(status="paid", amount_due=0, amount_paid=500000),
    )
    result = process(factory, paid, "evt_invoice_paid")
    assert result.case_id == first.case_id

    with session_scope(factory) as session:
        case = RecoveryCaseRepository(session).get(first.case_id)
        assert case.status == RecoveryCaseStatus.RECOVERED
        assert case.recovered_amount == Decimal("5000.00")


def test_unsupported_signed_event_is_persisted_as_ignored(db):
    _, factory = db
    body = raw_event("refund.processed", {"refund": {"entity": {"id": "rfnd_1"}}})
    result = process(factory, body, "evt_unsupported")
    assert result.status == "ignored"
    with session_scope(factory) as session:
        receipt = WebhookEventRepository(session).get("evt_unsupported")
        assert receipt is not None
        assert receipt.event_type == "refund.processed"
        assert receipt.processing_status == "ignored"


def test_fastapi_webhook_endpoint_verifies_processes_and_dedupes(db, monkeypatch):
    _, factory = db
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", SECRET)

    def override_db():
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app.dependency_overrides[get_db_session] = override_db
    try:
        client = TestClient(app)
        body = raw_event("payment.failed", payment_entity(payment_id="pay_http_1"))
        headers = {
            "X-Razorpay-Signature": sign(body),
            "X-Razorpay-Event-Id": "evt_http_1",
            "Content-Type": "application/json",
        }
        first = client.post("/webhooks/razorpay", content=body, headers=headers)
        second = client.post("/webhooks/razorpay", content=body, headers=headers)
        bad = client.post(
            "/webhooks/razorpay",
            content=body,
            headers={**headers, "X-Razorpay-Signature": "bad"},
        )

        assert first.status_code == 200
        assert first.json()["ok"] is True
        assert first.json()["duplicate"] is False
        assert second.status_code == 200
        assert second.json()["duplicate"] is True
        assert bad.status_code == 400
    finally:
        app.dependency_overrides.clear()
