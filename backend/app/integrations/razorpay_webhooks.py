from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping

from app.config import ApplicationSettings
from app.domain.enums import InvoiceStatus, PaymentMethod, PaymentStatus, SubscriptionStatus
from app.domain.invoice import Invoice
from app.domain.payment import Payment
from app.domain.subscription import Subscription


class RazorpayWebhookError(RuntimeError):
    """Base error for Razorpay webhook handling."""


class RazorpayWebhookConfigurationError(RazorpayWebhookError):
    """Raised when webhook configuration is missing or unsafe."""


class RazorpayWebhookSignatureError(RazorpayWebhookError):
    """Raised when X-Razorpay-Signature validation fails."""


class RazorpayWebhookPayloadError(RazorpayWebhookError):
    """Raised when a signed webhook body is not a valid Razorpay event."""


@dataclass(frozen=True)
class RazorpayWebhookSettings:
    webhook_secret: str

    def __post_init__(self) -> None:
        if not self.webhook_secret.strip():
            raise RazorpayWebhookConfigurationError(
                "RAZORPAY_WEBHOOK_SECRET is required for webhook verification"
            )

    @classmethod
    def from_env(cls) -> "RazorpayWebhookSettings":
        return cls(
            webhook_secret=ApplicationSettings.from_env().razorpay_webhook_secret
        )

    def __repr__(self) -> str:
        return "RazorpayWebhookSettings(webhook_secret='***REDACTED***')"


@dataclass(frozen=True)
class RazorpayWebhookEnvelope:
    event_id: str
    event_type: str
    account_id: str
    created_at: datetime
    payload: dict[str, Any]
    raw_event: dict[str, Any]


@dataclass(frozen=True)
class NormalizedRazorpayWebhook:
    envelope: RazorpayWebhookEnvelope
    payment: Payment | None = None
    subscription: Subscription | None = None
    invoice: Invoice | None = None


def verify_razorpay_webhook_signature(
    raw_body: bytes,
    received_signature: str | None,
    webhook_secret: str,
) -> None:
    if not received_signature:
        raise RazorpayWebhookSignatureError("Missing X-Razorpay-Signature header")
    if not webhook_secret:
        raise RazorpayWebhookConfigurationError("Webhook secret cannot be empty")

    expected = hmac.new(
        webhook_secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, received_signature.strip()):
        raise RazorpayWebhookSignatureError("Invalid Razorpay webhook signature")


def derive_event_id(raw_body: bytes, razorpay_event_id: str | None) -> str:
    if razorpay_event_id and razorpay_event_id.strip():
        return razorpay_event_id.strip()
    return f"body_{hashlib.sha256(raw_body).hexdigest()}"


def _unix_time(value: Any, *, fallback: datetime | None = None) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    return fallback or datetime.now(timezone.utc)


def _entity(payload: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    wrapped = payload.get(name)
    if not isinstance(wrapped, Mapping):
        return None
    entity = wrapped.get("entity")
    return dict(entity) if isinstance(entity, Mapping) else None


def _money_from_subunits(value: Any) -> Decimal:
    if value is None:
        return Decimal("0")
    return (Decimal(str(value)) / Decimal("100")).quantize(Decimal("0.01"))


def _payment_method(value: Any) -> PaymentMethod:
    try:
        return PaymentMethod(str(value))
    except (TypeError, ValueError):
        return PaymentMethod.UNKNOWN


def _payment_status(value: Any) -> PaymentStatus:
    try:
        return PaymentStatus(str(value))
    except (TypeError, ValueError):
        return PaymentStatus.CREATED


def _subscription_status(value: Any) -> SubscriptionStatus:
    try:
        return SubscriptionStatus(str(value))
    except (TypeError, ValueError):
        return SubscriptionStatus.CREATED


def _invoice_status(value: Any, event_type: str) -> InvoiceStatus:
    if event_type == "invoice.expired":
        return InvoiceStatus.OVERDUE
    try:
        return InvoiceStatus(str(value))
    except (TypeError, ValueError):
        return InvoiceStatus.ISSUED


def parse_razorpay_webhook(
    raw_body: bytes,
    *,
    event_id: str,
) -> RazorpayWebhookEnvelope:
    try:
        decoded = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RazorpayWebhookPayloadError("Webhook body must be valid UTF-8 JSON") from exc

    if not isinstance(decoded, dict):
        raise RazorpayWebhookPayloadError("Webhook body must be a JSON object")

    event_type = decoded.get("event")
    payload = decoded.get("payload")
    if not isinstance(event_type, str) or not event_type.strip():
        raise RazorpayWebhookPayloadError("Webhook payload is missing event type")
    if not isinstance(payload, dict):
        raise RazorpayWebhookPayloadError("Webhook payload is missing payload object")

    return RazorpayWebhookEnvelope(
        event_id=event_id,
        event_type=event_type.strip(),
        account_id=str(decoded.get("account_id") or "razorpay_test_account"),
        created_at=_unix_time(decoded.get("created_at")),
        payload=payload,
        raw_event=decoded,
    )


def normalize_razorpay_webhook(envelope: RazorpayWebhookEnvelope) -> NormalizedRazorpayWebhook:
    payment_entity = _entity(envelope.payload, "payment")
    subscription_entity = _entity(envelope.payload, "subscription")
    invoice_entity = _entity(envelope.payload, "invoice")

    payment: Payment | None = None
    if payment_entity and payment_entity.get("id"):
        payment = Payment(
            id=str(payment_entity["id"]),
            merchant_id=envelope.account_id,
            customer_id=(
                str(payment_entity["customer_id"])
                if payment_entity.get("customer_id")
                else None
            ),
            order_id=(
                str(payment_entity["order_id"])
                if payment_entity.get("order_id")
                else None
            ),
            amount=_money_from_subunits(payment_entity.get("amount")),
            currency=str(payment_entity.get("currency") or "INR"),
            status=_payment_status(payment_entity.get("status")),
            method=_payment_method(payment_entity.get("method")),
            bank=str(payment_entity["bank"]) if payment_entity.get("bank") else None,
            attempt_number=max(1, int(payment_entity.get("attempts") or 1)),
            error_code=(
                str(payment_entity["error_code"])
                if payment_entity.get("error_code")
                else None
            ),
            error_source=(
                str(payment_entity["error_source"])
                if payment_entity.get("error_source")
                else None
            ),
            error_step=(
                str(payment_entity["error_step"])
                if payment_entity.get("error_step")
                else None
            ),
            error_reason=(
                str(payment_entity["error_reason"])
                if payment_entity.get("error_reason")
                else None
            ),
            error_description=(
                str(payment_entity["error_description"])
                if payment_entity.get("error_description")
                else None
            ),
            created_at=_unix_time(payment_entity.get("created_at"), fallback=envelope.created_at),
            raw_payload=payment_entity,
        )

    subscription: Subscription | None = None
    if subscription_entity and subscription_entity.get("id"):
        payment_amount = payment.amount if payment is not None else Decimal("0")
        subscription = Subscription(
            id=str(subscription_entity["id"]),
            merchant_id=envelope.account_id,
            customer_id=str(subscription_entity.get("customer_id") or "unknown_customer"),
            amount=payment_amount,
            currency=payment.currency if payment is not None else "INR",
            status=_subscription_status(subscription_entity.get("status")),
            retry_count=max(0, int(subscription_entity.get("auth_attempts") or 0)),
            mandate_active=str(subscription_entity.get("status") or "") not in {"cancelled", "expired"},
            current_period_start=_unix_time(subscription_entity.get("current_start"))
            if subscription_entity.get("current_start")
            else None,
            current_period_end=_unix_time(subscription_entity.get("current_end"))
            if subscription_entity.get("current_end")
            else None,
            next_charge_at=_unix_time(subscription_entity.get("charge_at"))
            if subscription_entity.get("charge_at")
            else None,
            created_at=_unix_time(subscription_entity.get("created_at"), fallback=envelope.created_at),
            raw_payload=subscription_entity,
        )

    invoice: Invoice | None = None
    if invoice_entity and invoice_entity.get("id"):
        issued_at = _unix_time(
            invoice_entity.get("issued_at") or invoice_entity.get("date"),
            fallback=envelope.created_at,
        )
        due_at = _unix_time(invoice_entity.get("expire_by"), fallback=issued_at)
        days_overdue = max(0, (envelope.created_at.date() - due_at.date()).days)
        invoice = Invoice(
            id=str(invoice_entity["id"]),
            merchant_id=envelope.account_id,
            customer_id=str(invoice_entity.get("customer_id") or "unknown_customer"),
            amount_due=_money_from_subunits(invoice_entity.get("amount_due")),
            amount_paid=_money_from_subunits(invoice_entity.get("amount_paid")),
            currency=str(invoice_entity.get("currency") or "INR"),
            status=_invoice_status(invoice_entity.get("status"), envelope.event_type),
            issued_at=issued_at,
            due_at=due_at,
            days_overdue=days_overdue,
            created_at=_unix_time(invoice_entity.get("created_at"), fallback=issued_at),
            raw_payload=invoice_entity,
        )

    return NormalizedRazorpayWebhook(
        envelope=envelope,
        payment=payment,
        subscription=subscription,
        invoice=invoice,
    )


def payment_notes(payment: Payment | None) -> dict[str, Any]:
    if payment is None:
        return {}
    notes = payment.raw_payload.get("notes")
    return dict(notes) if isinstance(notes, Mapping) else {}
