from __future__ import annotations

from datetime import datetime, time
from decimal import Decimal
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, Numeric, String, Text, Time
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


MONEY = Numeric(18, 2)
PROBABILITY = Numeric(8, 7)


class CustomerRecord(Base):
    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    merchant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    email: Mapped[str | None] = mapped_column(String(320))
    phone: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    lifetime_value: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    successful_payments: Mapped[int] = mapped_column(Integer, nullable=False)
    failed_payments: Mapped[int] = mapped_column(Integer, nullable=False)
    historical_payment_success_rate: Mapped[Decimal] = mapped_column(PROBABILITY, nullable=False)
    previous_recovery_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    previous_recovery_successes: Mapped[int] = mapped_column(Integer, nullable=False)
    preferred_payment_method: Mapped[str | None] = mapped_column(String(64))
    preferred_channel: Mapped[str | None] = mapped_column(String(64))
    language_preference: Mapped[str] = mapped_column(String(16), nullable=False)
    do_not_contact: Mapped[bool] = mapped_column(Boolean, nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)


class PaymentRecord(Base):
    __tablename__ = "payments"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    merchant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    customer_id: Mapped[str | None] = mapped_column(String(128), index=True)
    order_id: Mapped[str | None] = mapped_column(String(128), index=True)
    amount: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    method: Mapped[str] = mapped_column(String(64), nullable=False)
    bank: Mapped[str | None] = mapped_column(String(128))
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_source: Mapped[str | None] = mapped_column(String(128))
    error_step: Mapped[str | None] = mapped_column(String(128))
    error_reason: Mapped[str | None] = mapped_column(String(256))
    error_description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)


class SubscriptionRecord(Base):
    __tablename__ = "subscriptions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    merchant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    customer_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    amount: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    preferred_payment_method: Mapped[str | None] = mapped_column(String(64))
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False)
    mandate_active: Mapped[bool] = mapped_column(Boolean, nullable=False)
    current_period_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    current_period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_charge_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)


class InvoiceRecord(Base):
    __tablename__ = "invoices"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    merchant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    customer_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    amount_due: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    amount_paid: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    days_overdue: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)


class RecoveryCaseRecord(Base):
    __tablename__ = "recovery_cases"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    merchant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    customer_id: Mapped[str | None] = mapped_column(String(128), index=True)
    case_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    amount_at_risk: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False)
    payment_id: Mapped[str | None] = mapped_column(String(128), index=True)
    subscription_id: Mapped[str | None] = mapped_column(String(128), index=True)
    invoice_id: Mapped[str | None] = mapped_column(String(128), index=True)
    checkout_id: Mapped[str | None] = mapped_column(String(128), index=True)
    payment_method: Mapped[str | None] = mapped_column(String(64))
    failure_class: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_source: Mapped[str | None] = mapped_column(String(128))
    error_step: Mapped[str | None] = mapped_column(String(128))
    error_reason: Mapped[str | None] = mapped_column(String(256))
    error_description: Mapped[str | None] = mapped_column(Text)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    recovery_retry_count: Mapped[int] = mapped_column(Integer, nullable=False)
    previous_contacts: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_recovery_probability: Mapped[Decimal | None] = mapped_column(PROBABILITY)
    expected_recovery_value: Mapped[Decimal | None] = mapped_column(MONEY)
    recovered_amount: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    selected_action_id: Mapped[str | None] = mapped_column(String(128), index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    actions: Mapped[list["RecoveryActionRecord"]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    audit_events: Mapped[list["AuditEventRecord"]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class RecoveryActionRecord(Base):
    __tablename__ = "recovery_actions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    case_id: Mapped[str] = mapped_column(
        ForeignKey("recovery_cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    action_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    channel: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    scheduled_for: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    amount: Mapped[Decimal | None] = mapped_column(MONEY)
    predicted_recovery_probability: Mapped[Decimal | None] = mapped_column(PROBABILITY)
    expected_recovery_value: Mapped[Decimal | None] = mapped_column(MONEY)
    reason: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    case: Mapped[RecoveryCaseRecord] = relationship(back_populates="actions")


class AuditEventRecord(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    case_id: Mapped[str] = mapped_column(
        ForeignKey("recovery_cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    actor: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    data_json: Mapped[dict[str, Any]] = mapped_column("data", JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    case: Mapped[RecoveryCaseRecord] = relationship(back_populates="audit_events")




class WebhookEventRecord(Base):
    __tablename__ = "webhook_events"

    event_id: Mapped[str] = mapped_column(String(192), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    processing_status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    signature_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    case_id: Mapped[str | None] = mapped_column(String(128), index=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column("payload", JSON, nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

class MerchantPolicyRecord(Base):
    __tablename__ = "merchant_policies"

    merchant_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    max_contacts_per_case: Mapped[int] = mapped_column(Integer, nullable=False)
    contact_window_days: Mapped[int] = mapped_column(Integer, nullable=False)
    max_payment_retries: Mapped[int] = mapped_column(Integer, nullable=False)
    max_recovery_window_days: Mapped[int] = mapped_column(Integer, nullable=False)
    human_approval_threshold: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    allow_partial_payments: Mapped[bool] = mapped_column(Boolean, nullable=False)
    allow_voice_calls: Mapped[bool] = mapped_column(Boolean, nullable=False)
    quiet_hours_start: Mapped[time] = mapped_column(Time, nullable=False)
    quiet_hours_end: Mapped[time] = mapped_column(Time, nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    allowed_channels: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    allowed_actions: Mapped[list[str]] = mapped_column(JSON, nullable=False)
