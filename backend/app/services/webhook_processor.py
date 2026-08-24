from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from backend.app.db.repositories import (
    AuditEventRepository,
    InvoiceRepository,
    PaymentRepository,
    RecoveryCaseRepository,
    SubscriptionRepository,
    WebhookEventRepository,
)
from app.domain.audit import AuditEvent
from app.domain.enums import (
    AuditActor,
    AuditEventType,
    CaseType,
    FailureClass,
    InvoiceStatus,
    PaymentStatus,
    RecoveryCaseStatus,
    SubscriptionStatus,
)
from app.domain.recovery_case import RecoveryCase
from backend.app.integrations.razorpay_webhooks import (
    NormalizedRazorpayWebhook,
    RazorpayWebhookSettings,
    derive_event_id,
    normalize_razorpay_webhook,
    parse_razorpay_webhook,
    payment_notes,
    verify_razorpay_webhook_signature,
)
from app.services.diagnosis import apply_diagnosis


@dataclass(frozen=True)
class WebhookProcessingResult:
    event_id: str
    event_type: str
    status: str
    duplicate: bool = False
    case_id: str | None = None


class RazorpayWebhookProcessor:
    def __init__(
        self,
        session: Session,
        *,
        settings: RazorpayWebhookSettings | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or RazorpayWebhookSettings.from_env()
        self.webhooks = WebhookEventRepository(session)
        self.payments = PaymentRepository(session)
        self.subscriptions = SubscriptionRepository(session)
        self.invoices = InvoiceRepository(session)
        self.cases = RecoveryCaseRepository(session)
        self.audit = AuditEventRepository(session)

    def process(
        self,
        raw_body: bytes,
        *,
        signature: str | None,
        razorpay_event_id: str | None,
    ) -> WebhookProcessingResult:
        verify_razorpay_webhook_signature(
            raw_body,
            signature,
            self.settings.webhook_secret,
        )

        event_id = derive_event_id(raw_body, razorpay_event_id)
        existing = self.webhooks.get(event_id)
        if existing is not None:
            return WebhookProcessingResult(
                event_id=event_id,
                event_type=existing.event_type,
                status=existing.processing_status,
                duplicate=True,
                case_id=existing.case_id,
            )

        envelope = parse_razorpay_webhook(raw_body, event_id=event_id)
        normalized = normalize_razorpay_webhook(envelope)

        receipt = self.webhooks.create_received(
            event_id=event_id,
            event_type=envelope.event_type,
            account_id=envelope.account_id,
            payload=envelope.raw_event,
        )

        case_id = self._process_normalized(normalized)
        receipt.processing_status = "processed" if case_id else "ignored"
        receipt.case_id = case_id
        receipt.processed_at = datetime.now(timezone.utc)
        self.session.flush()

        return WebhookProcessingResult(
            event_id=event_id,
            event_type=envelope.event_type,
            status=receipt.processing_status,
            duplicate=False,
            case_id=case_id,
        )

    def _process_normalized(self, event: NormalizedRazorpayWebhook) -> str | None:
        event_type = event.envelope.event_type

        if event.payment is not None:
            self.payments.save(event.payment)
        if event.subscription is not None:
            self.subscriptions.save(event.subscription)
        if event.invoice is not None:
            self.invoices.save(event.invoice)

        if event_type == "payment.failed" and event.payment is not None:
            return self._handle_payment_failed(event)

        if event_type in {"payment.captured", "order.paid"} and event.payment is not None:
            case = self._resolve_case(event)
            if case is None:
                return None
            self._recover_case(case, event.payment.amount, event_type)
            return case.id

        if event_type in {"subscription.pending", "subscription.halted"} and event.subscription is not None:
            return self._handle_subscription_failure(event)

        if event_type in {"subscription.charged", "subscription.activated"} and event.subscription is not None:
            case = self.cases.find_open_by_subscription_id(event.subscription.id)
            if case is None:
                return None
            amount = event.payment.amount if event.payment is not None else case.amount_at_risk
            self._recover_case(case, amount, event_type)
            return case.id

        if event_type == "invoice.expired" and event.invoice is not None:
            return self._handle_invoice_overdue(event)

        if event_type == "invoice.partially_paid" and event.invoice is not None:
            case = self.cases.find_open_by_invoice_id(event.invoice.id)
            if case is None and event.invoice.amount_due > 0:
                case = self._create_invoice_case(event)
            if case is None:
                return None
            baseline_paid = Decimal(str(case.metadata.get("provider_amount_paid_at_case_creation", "0")))
            provider_recovered = max(Decimal("0"), event.invoice.amount_paid - baseline_paid)
            case.recovered_amount = min(case.amount_at_risk, provider_recovered)
            case.updated_at = datetime.now(timezone.utc)
            self.cases.save(case)
            self._audit(
                case.id,
                AuditEventType.PAYMENT_RECEIVED,
                "Razorpay reported a partial invoice payment.",
                {"invoice_id": event.invoice.id, "amount_paid": str(event.invoice.amount_paid)},
            )
            return case.id

        if event_type == "invoice.paid" and event.invoice is not None:
            case = self.cases.find_open_by_invoice_id(event.invoice.id)
            if case is None:
                return None
            self._recover_case(case, case.amount_at_risk, event_type)
            return case.id

        return None

    def _handle_payment_failed(self, event: NormalizedRazorpayWebhook) -> str:
        payment = event.payment
        assert payment is not None

        existing = self.cases.find_open_by_payment_id(payment.id)
        if existing is not None:
            return existing.id

        case = RecoveryCase(
            merchant_id=payment.merchant_id,
            customer_id=payment.customer_id,
            case_type=CaseType.PAYMENT_FAILURE,
            amount_at_risk=payment.amount,
            currency=payment.currency,
            payment_id=payment.id,
            payment_method=payment.method,
            error_code=payment.error_code,
            error_source=payment.error_source,
            error_step=payment.error_step,
            error_reason=payment.error_reason,
            error_description=payment.error_description,
            attempt_count=payment.attempt_number,
            metadata={"source": "razorpay_webhook", "event_id": event.envelope.event_id},
        )
        diagnosis = apply_diagnosis(case)
        self.cases.save(case)
        self._audit(
            case.id,
            AuditEventType.CASE_CREATED,
            "Recovery case created from Razorpay payment.failed webhook.",
            {"payment_id": payment.id, "event_id": event.envelope.event_id},
        )
        self._audit(
            case.id,
            AuditEventType.CASE_DIAGNOSED,
            "Payment failure deterministically diagnosed from Razorpay provider facts.",
            {"failure_class": case.failure_class.value},
        )
        return case.id

    def _handle_subscription_failure(self, event: NormalizedRazorpayWebhook) -> str:
        subscription = event.subscription
        assert subscription is not None

        existing = self.cases.find_open_by_subscription_id(subscription.id)
        if existing is not None:
            existing.recovery_retry_count = max(existing.recovery_retry_count, subscription.retry_count)
            existing.updated_at = datetime.now(timezone.utc)
            self.cases.save(existing)
            return existing.id

        amount = event.payment.amount if event.payment is not None else subscription.amount
        case = RecoveryCase(
            merchant_id=subscription.merchant_id,
            customer_id=subscription.customer_id,
            case_type=CaseType.SUBSCRIPTION_FAILURE,
            amount_at_risk=amount,
            currency=subscription.currency,
            payment_id=event.payment.id if event.payment is not None else None,
            subscription_id=subscription.id,
            payment_method=event.payment.method if event.payment is not None else None,
            failure_class=FailureClass.MANDATE_FAILURE,
            error_code=event.payment.error_code if event.payment is not None else None,
            error_source=event.payment.error_source if event.payment is not None else None,
            error_step=event.payment.error_step if event.payment is not None else None,
            error_reason=event.payment.error_reason if event.payment is not None else None,
            error_description=event.payment.error_description if event.payment is not None else None,
            attempt_count=event.payment.attempt_number if event.payment is not None else 0,
            recovery_retry_count=subscription.retry_count,
            metadata={"source": "razorpay_webhook", "event_id": event.envelope.event_id},
        )
        if event.payment is not None:
            apply_diagnosis(case)
        self.cases.save(case)
        self._audit(
            case.id,
            AuditEventType.CASE_CREATED,
            f"Recovery case created from Razorpay {event.envelope.event_type} webhook.",
            {"subscription_id": subscription.id, "event_id": event.envelope.event_id},
        )
        return case.id

    def _handle_invoice_overdue(self, event: NormalizedRazorpayWebhook) -> str:
        invoice = event.invoice
        assert invoice is not None
        existing = self.cases.find_open_by_invoice_id(invoice.id)
        if existing is not None:
            existing.amount_at_risk = invoice.amount_due
            existing.updated_at = datetime.now(timezone.utc)
            self.cases.save(existing)
            return existing.id
        return self._create_invoice_case(event).id

    def _create_invoice_case(self, event: NormalizedRazorpayWebhook) -> RecoveryCase:
        invoice = event.invoice
        assert invoice is not None
        case = RecoveryCase(
            merchant_id=invoice.merchant_id,
            customer_id=invoice.customer_id,
            case_type=CaseType.OVERDUE_INVOICE,
            amount_at_risk=invoice.amount_due,
            currency=invoice.currency,
            invoice_id=invoice.id,
            failure_class=FailureClass.OVERDUE_RECEIVABLE,
            recovered_amount=Decimal("0"),
            metadata={
                "source": "razorpay_webhook",
                "event_id": event.envelope.event_id,
                "provider_amount_paid_at_case_creation": str(invoice.amount_paid),
            },
        )
        self.cases.save(case)
        self._audit(
            case.id,
            AuditEventType.CASE_CREATED,
            f"Recovery case created from Razorpay {event.envelope.event_type} webhook.",
            {"invoice_id": invoice.id, "event_id": event.envelope.event_id},
        )
        return case

    def _resolve_case(self, event: NormalizedRazorpayWebhook) -> RecoveryCase | None:
        notes = payment_notes(event.payment)
        explicit_case_id = notes.get("recoverai_case_id")
        if explicit_case_id:
            case = self.cases.get(str(explicit_case_id))
            if case is not None and case.status not in {
                RecoveryCaseStatus.RECOVERED,
                RecoveryCaseStatus.STOPPED,
            }:
                return case

        if event.payment is not None:
            case = self.cases.find_open_by_payment_id(event.payment.id)
            if case is not None:
                return case
        if event.subscription is not None:
            case = self.cases.find_open_by_subscription_id(event.subscription.id)
            if case is not None:
                return case
        if event.invoice is not None:
            return self.cases.find_open_by_invoice_id(event.invoice.id)
        return None

    def _recover_case(self, case: RecoveryCase, amount: Decimal, event_type: str) -> None:
        recovered = min(max(amount, Decimal("0")), case.amount_at_risk)
        case.recovered_amount = max(case.recovered_amount, recovered)
        if case.recovered_amount >= case.amount_at_risk:
            case.status = RecoveryCaseStatus.RECOVERED
            case.closed_at = datetime.now(timezone.utc)
        case.updated_at = datetime.now(timezone.utc)
        self.cases.save(case)
        self._audit(
            case.id,
            AuditEventType.PAYMENT_RECEIVED,
            f"Razorpay reported successful recovery via {event_type}.",
            {"recovered_amount": str(case.recovered_amount)},
        )
        if case.status == RecoveryCaseStatus.RECOVERED:
            self._audit(
                case.id,
                AuditEventType.CASE_RECOVERED,
                "Revenue-at-risk case marked recovered.",
                {"recovered_amount": str(case.recovered_amount)},
            )

    def _audit(
        self,
        case_id: str,
        event_type: AuditEventType,
        message: str,
        data: dict[str, Any],
    ) -> None:
        self.audit.append(
            AuditEvent(
                id=f"audit_{uuid4().hex}",
                case_id=case_id,
                event_type=event_type,
                actor=AuditActor.RAZORPAY,
                message=message,
                data=data,
            )
        )
