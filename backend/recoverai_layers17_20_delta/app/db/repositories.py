from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    AuditEventRecord,
    CustomerRecord,
    InvoiceRecord,
    MerchantPolicyRecord,
    PaymentRecord,
    RecoveryActionRecord,
    RecoveryCaseRecord,
    SubscriptionRecord,
    WebhookEventRecord,
)
from app.domain.actions import RecoveryAction
from app.domain.audit import AuditEvent
from app.domain.customer import Customer
from app.domain.enums import (
    ActionStatus,
    AuditActor,
    AuditEventType,
    CaseType,
    CommunicationChannel,
    FailureClass,
    InvoiceStatus,
    PaymentMethod,
    PaymentStatus,
    RecoveryActionType,
    RecoveryCaseStatus,
    SubscriptionStatus,
)
from app.domain.invoice import Invoice
from app.domain.payment import Payment
from app.domain.policies import MerchantPolicy
from app.domain.recovery_case import RecoveryCase
from app.domain.subscription import Subscription


def _enum_value(value):
    return None if value is None else value.value


def _probability(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


class CustomerRepository:
    def __init__(self, session: Session):
        self.session = session

    def save(self, customer: Customer) -> Customer:
        row = CustomerRecord(
            id=customer.id,
            merchant_id=customer.merchant_id,
            email=customer.email,
            phone=customer.phone,
            created_at=customer.created_at,
            lifetime_value=customer.lifetime_value,
            successful_payments=customer.successful_payments,
            failed_payments=customer.failed_payments,
            historical_payment_success_rate=Decimal(str(customer.historical_payment_success_rate)),
            previous_recovery_attempts=customer.previous_recovery_attempts,
            previous_recovery_successes=customer.previous_recovery_successes,
            preferred_payment_method=_enum_value(customer.preferred_payment_method),
            preferred_channel=_enum_value(customer.preferred_channel),
            language_preference=customer.language_preference,
            do_not_contact=customer.do_not_contact,
            timezone=customer.timezone,
        )
        self.session.merge(row)
        self.session.flush()
        return customer

    def get(self, customer_id: str) -> Customer | None:
        row = self.session.get(CustomerRecord, customer_id)
        if row is None:
            return None
        return Customer(
            id=row.id,
            merchant_id=row.merchant_id,
            email=row.email,
            phone=row.phone,
            created_at=row.created_at,
            lifetime_value=row.lifetime_value,
            successful_payments=row.successful_payments,
            failed_payments=row.failed_payments,
            historical_payment_success_rate=float(row.historical_payment_success_rate),
            previous_recovery_attempts=row.previous_recovery_attempts,
            previous_recovery_successes=row.previous_recovery_successes,
            preferred_payment_method=PaymentMethod(row.preferred_payment_method) if row.preferred_payment_method else None,
            preferred_channel=CommunicationChannel(row.preferred_channel) if row.preferred_channel else None,
            language_preference=row.language_preference,
            do_not_contact=row.do_not_contact,
            timezone=row.timezone,
        )


class PaymentRepository:
    def __init__(self, session: Session):
        self.session = session

    def save(self, payment: Payment) -> Payment:
        self.session.merge(PaymentRecord(
            id=payment.id, merchant_id=payment.merchant_id, customer_id=payment.customer_id,
            order_id=payment.order_id, amount=payment.amount, currency=payment.currency,
            status=payment.status.value, method=payment.method.value, bank=payment.bank,
            attempt_number=payment.attempt_number, error_code=payment.error_code,
            error_source=payment.error_source, error_step=payment.error_step,
            error_reason=payment.error_reason, error_description=payment.error_description,
            created_at=payment.created_at, raw_payload=payment.raw_payload,
        ))
        self.session.flush()
        return payment

    def get(self, payment_id: str) -> Payment | None:
        row = self.session.get(PaymentRecord, payment_id)
        if row is None:
            return None
        return Payment(
            id=row.id, merchant_id=row.merchant_id, customer_id=row.customer_id,
            order_id=row.order_id, amount=row.amount, currency=row.currency,
            status=PaymentStatus(row.status), method=PaymentMethod(row.method), bank=row.bank,
            attempt_number=row.attempt_number, error_code=row.error_code,
            error_source=row.error_source, error_step=row.error_step,
            error_reason=row.error_reason, error_description=row.error_description,
            created_at=row.created_at, raw_payload=row.raw_payload or {},
        )


class SubscriptionRepository:
    def __init__(self, session: Session): self.session = session

    def save(self, subscription: Subscription) -> Subscription:
        self.session.merge(SubscriptionRecord(
            id=subscription.id, merchant_id=subscription.merchant_id,
            customer_id=subscription.customer_id, amount=subscription.amount,
            currency=subscription.currency, status=subscription.status.value,
            preferred_payment_method=_enum_value(subscription.preferred_payment_method),
            retry_count=subscription.retry_count, mandate_active=subscription.mandate_active,
            current_period_start=subscription.current_period_start,
            current_period_end=subscription.current_period_end,
            next_charge_at=subscription.next_charge_at, created_at=subscription.created_at,
            raw_payload=subscription.raw_payload,
        ))
        self.session.flush()
        return subscription

    def get(self, subscription_id: str) -> Subscription | None:
        row = self.session.get(SubscriptionRecord, subscription_id)
        if row is None: return None
        return Subscription(
            id=row.id, merchant_id=row.merchant_id, customer_id=row.customer_id,
            amount=row.amount, currency=row.currency, status=SubscriptionStatus(row.status),
            preferred_payment_method=PaymentMethod(row.preferred_payment_method) if row.preferred_payment_method else None,
            retry_count=row.retry_count, mandate_active=row.mandate_active,
            current_period_start=row.current_period_start, current_period_end=row.current_period_end,
            next_charge_at=row.next_charge_at, created_at=row.created_at, raw_payload=row.raw_payload or {},
        )


class InvoiceRepository:
    def __init__(self, session: Session): self.session = session

    def save(self, invoice: Invoice) -> Invoice:
        self.session.merge(InvoiceRecord(
            id=invoice.id, merchant_id=invoice.merchant_id, customer_id=invoice.customer_id,
            amount_due=invoice.amount_due, amount_paid=invoice.amount_paid,
            currency=invoice.currency, status=invoice.status.value, issued_at=invoice.issued_at,
            due_at=invoice.due_at, days_overdue=invoice.days_overdue,
            created_at=invoice.created_at, raw_payload=invoice.raw_payload,
        ))
        self.session.flush()
        return invoice

    def get(self, invoice_id: str) -> Invoice | None:
        row = self.session.get(InvoiceRecord, invoice_id)
        if row is None: return None
        return Invoice(
            id=row.id, merchant_id=row.merchant_id, customer_id=row.customer_id,
            amount_due=row.amount_due, amount_paid=row.amount_paid, currency=row.currency,
            status=InvoiceStatus(row.status), issued_at=row.issued_at, due_at=row.due_at,
            days_overdue=row.days_overdue, created_at=row.created_at, raw_payload=row.raw_payload or {},
        )


class RecoveryCaseRepository:
    def __init__(self, session: Session): self.session = session

    def save(self, case: RecoveryCase) -> RecoveryCase:
        self.session.merge(RecoveryCaseRecord(
            id=case.id, merchant_id=case.merchant_id, customer_id=case.customer_id,
            case_type=case.case_type.value, status=case.status.value,
            amount_at_risk=case.amount_at_risk, currency=case.currency,
            payment_id=case.payment_id, subscription_id=case.subscription_id,
            invoice_id=case.invoice_id, checkout_id=case.checkout_id,
            payment_method=_enum_value(case.payment_method), failure_class=case.failure_class.value,
            error_code=case.error_code, error_source=case.error_source, error_step=case.error_step,
            error_reason=case.error_reason, error_description=case.error_description,
            attempt_count=case.attempt_count, recovery_retry_count=case.recovery_retry_count,
            previous_contacts=case.previous_contacts,
            predicted_recovery_probability=Decimal(str(case.predicted_recovery_probability)) if case.predicted_recovery_probability is not None else None,
            expected_recovery_value=case.expected_recovery_value, recovered_amount=case.recovered_amount,
            selected_action_id=case.selected_action_id, metadata_json=case.metadata,
            created_at=case.created_at, updated_at=case.updated_at, closed_at=case.closed_at,
        ))
        self.session.flush()
        return case

    def get(self, case_id: str) -> RecoveryCase | None:
        row = self.session.get(RecoveryCaseRecord, case_id)
        return None if row is None else self._to_domain(row)

    def list_for_merchant(self, merchant_id: str, *, status: RecoveryCaseStatus | None = None) -> list[RecoveryCase]:
        stmt = select(RecoveryCaseRecord).where(RecoveryCaseRecord.merchant_id == merchant_id)
        if status is not None:
            stmt = stmt.where(RecoveryCaseRecord.status == status.value)
        stmt = stmt.order_by(RecoveryCaseRecord.created_at.desc())
        return [self._to_domain(row) for row in self.session.scalars(stmt).all()]

    def find_open_by_payment_id(self, payment_id: str) -> RecoveryCase | None:
        stmt = (
            select(RecoveryCaseRecord)
            .where(
                RecoveryCaseRecord.payment_id == payment_id,
                RecoveryCaseRecord.status.notin_([
                    RecoveryCaseStatus.RECOVERED.value,
                    RecoveryCaseStatus.STOPPED.value,
                ]),
            )
            .order_by(RecoveryCaseRecord.created_at.desc())
            .limit(1)
        )
        row = self.session.scalar(stmt)
        return None if row is None else self._to_domain(row)

    def find_open_by_subscription_id(self, subscription_id: str) -> RecoveryCase | None:
        stmt = (
            select(RecoveryCaseRecord)
            .where(
                RecoveryCaseRecord.subscription_id == subscription_id,
                RecoveryCaseRecord.status.notin_([
                    RecoveryCaseStatus.RECOVERED.value,
                    RecoveryCaseStatus.STOPPED.value,
                ]),
            )
            .order_by(RecoveryCaseRecord.created_at.desc())
            .limit(1)
        )
        row = self.session.scalar(stmt)
        return None if row is None else self._to_domain(row)

    def find_open_by_invoice_id(self, invoice_id: str) -> RecoveryCase | None:
        stmt = (
            select(RecoveryCaseRecord)
            .where(
                RecoveryCaseRecord.invoice_id == invoice_id,
                RecoveryCaseRecord.status.notin_([
                    RecoveryCaseStatus.RECOVERED.value,
                    RecoveryCaseStatus.STOPPED.value,
                ]),
            )
            .order_by(RecoveryCaseRecord.created_at.desc())
            .limit(1)
        )
        row = self.session.scalar(stmt)
        return None if row is None else self._to_domain(row)

    @staticmethod
    def _to_domain(row: RecoveryCaseRecord) -> RecoveryCase:
        return RecoveryCase(
            id=row.id, merchant_id=row.merchant_id, customer_id=row.customer_id,
            case_type=CaseType(row.case_type), status=RecoveryCaseStatus(row.status),
            amount_at_risk=row.amount_at_risk, currency=row.currency, payment_id=row.payment_id,
            subscription_id=row.subscription_id, invoice_id=row.invoice_id, checkout_id=row.checkout_id,
            payment_method=PaymentMethod(row.payment_method) if row.payment_method else None,
            failure_class=FailureClass(row.failure_class), error_code=row.error_code,
            error_source=row.error_source, error_step=row.error_step, error_reason=row.error_reason,
            error_description=row.error_description, attempt_count=row.attempt_count,
            recovery_retry_count=row.recovery_retry_count, previous_contacts=row.previous_contacts,
            predicted_recovery_probability=_probability(row.predicted_recovery_probability),
            expected_recovery_value=row.expected_recovery_value, recovered_amount=row.recovered_amount,
            selected_action_id=row.selected_action_id, metadata=row.metadata_json or {},
            created_at=row.created_at, updated_at=row.updated_at, closed_at=row.closed_at,
        )


class RecoveryActionRepository:
    def __init__(self, session: Session): self.session = session

    def save(self, action: RecoveryAction) -> RecoveryAction:
        self.session.merge(RecoveryActionRecord(
            id=action.id, case_id=action.case_id, action_type=action.action_type.value,
            channel=action.channel.value, status=action.status.value, scheduled_for=action.scheduled_for,
            amount=action.amount,
            predicted_recovery_probability=Decimal(str(action.predicted_recovery_probability)) if action.predicted_recovery_probability is not None else None,
            expected_recovery_value=action.expected_recovery_value, reason=action.reason,
            metadata_json=action.metadata, created_at=action.created_at, executed_at=action.executed_at,
        ))
        self.session.flush()
        return action

    def get(self, action_id: str) -> RecoveryAction | None:
        row = self.session.get(RecoveryActionRecord, action_id)
        return None if row is None else self._to_domain(row)

    def list_for_case(self, case_id: str) -> list[RecoveryAction]:
        stmt = select(RecoveryActionRecord).where(RecoveryActionRecord.case_id == case_id).order_by(RecoveryActionRecord.created_at)
        return [self._to_domain(row) for row in self.session.scalars(stmt).all()]

    @staticmethod
    def _to_domain(row: RecoveryActionRecord) -> RecoveryAction:
        return RecoveryAction(
            id=row.id, case_id=row.case_id, action_type=RecoveryActionType(row.action_type),
            channel=CommunicationChannel(row.channel), status=ActionStatus(row.status),
            scheduled_for=row.scheduled_for, amount=row.amount,
            predicted_recovery_probability=_probability(row.predicted_recovery_probability),
            expected_recovery_value=row.expected_recovery_value, reason=row.reason,
            metadata=row.metadata_json or {}, created_at=row.created_at, executed_at=row.executed_at,
        )


class AuditEventRepository:
    def __init__(self, session: Session): self.session = session

    def append(self, event: AuditEvent) -> AuditEvent:
        self.session.add(AuditEventRecord(
            id=event.id, case_id=event.case_id, event_type=event.event_type.value,
            actor=event.actor.value, message=event.message, data_json=event.data,
            created_at=event.created_at,
        ))
        self.session.flush()
        return event

    def list_for_case(self, case_id: str) -> list[AuditEvent]:
        stmt = select(AuditEventRecord).where(AuditEventRecord.case_id == case_id).order_by(AuditEventRecord.created_at, AuditEventRecord.id)
        return [AuditEvent(
            id=row.id, case_id=row.case_id, event_type=AuditEventType(row.event_type),
            actor=AuditActor(row.actor), message=row.message, data=row.data_json or {},
            created_at=row.created_at,
        ) for row in self.session.scalars(stmt).all()]


class MerchantPolicyRepository:
    def __init__(self, session: Session): self.session = session

    def save(self, policy: MerchantPolicy) -> MerchantPolicy:
        self.session.merge(MerchantPolicyRecord(
            merchant_id=policy.merchant_id, max_contacts_per_case=policy.max_contacts_per_case,
            contact_window_days=policy.contact_window_days, max_payment_retries=policy.max_payment_retries,
            max_recovery_window_days=policy.max_recovery_window_days,
            human_approval_threshold=policy.human_approval_threshold,
            allow_partial_payments=policy.allow_partial_payments, allow_voice_calls=policy.allow_voice_calls,
            quiet_hours_start=policy.quiet_hours_start, quiet_hours_end=policy.quiet_hours_end,
            timezone=policy.timezone,
            allowed_channels=sorted(channel.value for channel in policy.allowed_channels),
            allowed_actions=sorted(action.value for action in policy.allowed_actions),
        ))
        self.session.flush()
        return policy

    def get(self, merchant_id: str) -> MerchantPolicy | None:
        row = self.session.get(MerchantPolicyRecord, merchant_id)
        if row is None: return None
        return MerchantPolicy(
            merchant_id=row.merchant_id, max_contacts_per_case=row.max_contacts_per_case,
            contact_window_days=row.contact_window_days, max_payment_retries=row.max_payment_retries,
            max_recovery_window_days=row.max_recovery_window_days,
            human_approval_threshold=row.human_approval_threshold,
            allow_partial_payments=row.allow_partial_payments, allow_voice_calls=row.allow_voice_calls,
            quiet_hours_start=row.quiet_hours_start, quiet_hours_end=row.quiet_hours_end,
            timezone=row.timezone,
            allowed_channels={CommunicationChannel(value) for value in row.allowed_channels},
            allowed_actions={RecoveryActionType(value) for value in row.allowed_actions},
        )


class WebhookEventRepository:
    def __init__(self, session: Session):
        self.session = session

    def get(self, event_id: str) -> WebhookEventRecord | None:
        return self.session.get(WebhookEventRecord, event_id)

    def create_received(
        self,
        *,
        event_id: str,
        event_type: str,
        account_id: str,
        payload: dict,
    ) -> WebhookEventRecord:
        from datetime import datetime, timezone

        row = WebhookEventRecord(
            event_id=event_id,
            event_type=event_type,
            account_id=account_id,
            processing_status="received",
            signature_verified=True,
            case_id=None,
            payload_json=payload,
            received_at=datetime.now(timezone.utc),
            processed_at=None,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def list_recent(self, *, limit: int = 100) -> list[WebhookEventRecord]:
        stmt = (
            select(WebhookEventRecord)
            .order_by(WebhookEventRecord.received_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt).all())
