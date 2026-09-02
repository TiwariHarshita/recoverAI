from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.domain.action_scoring import MerchantScoringProfile
from app.domain.actions import RecoveryAction
from app.domain.audit import AuditEvent
from app.domain.customer import Customer
from app.domain.enums import (
    AuditActor,
    AuditEventType,
    CaseType,
    CommunicationChannel,
    FailureClass,
    PaymentMethod,
    RecoveryActionType,
)
from app.domain.merchant import Merchant
from app.domain.payment import Payment
from app.domain.payment_attempt import PaymentAttempt
from app.domain.recovery_case import RecoveryCase


def test_create_customer():
    customer = Customer(
        id="customer_1",
        merchant_id="merchant_1",
        lifetime_value=Decimal("50000"),
        successful_payments=9,
        failed_payments=1,
        historical_payment_success_rate=0.90,
    )

    assert customer.id == "customer_1"
    assert customer.historical_payment_success_rate == 0.90


def test_canonical_merchant_accepts_legacy_scoring_identity_and_serializes_canonically():
    merchant = Merchant(
        merchant_id="merchant_1",
        name="Example Merchant",
        archetype="ecommerce",
        average_order_value=Decimal("1250.50"),
        currency="inr",
    )

    assert merchant.id == "merchant_1"
    assert merchant.merchant_id == "merchant_1"
    assert merchant.currency == "INR"
    assert merchant.model_dump()["id"] == "merchant_1"
    assert "merchant_id" not in merchant.model_dump()
    assert MerchantScoringProfile is Merchant


def test_canonical_merchant_rejects_invalid_business_profile():
    with pytest.raises(ValidationError):
        Merchant(id="merchant_1", average_order_value=Decimal("0"))


def test_payment_attempt_is_provider_independent_and_payment_is_compatible():
    attempt = PaymentAttempt(
        id="attempt_1",
        merchant_id="merchant_1",
        customer_id="customer_1",
        amount=Decimal("499.00"),
        currency="inr",
        status="failed",
        method="upi",
        attempt_number=2,
        error_reason="insufficient_funds",
    )
    legacy_payment = Payment(**attempt.model_dump(), raw_payload={"provider": "test"})

    assert attempt.currency == "INR"
    assert attempt.status.value == "failed"
    assert attempt.method == PaymentMethod.UPI
    assert "raw_payload" not in attempt.model_dump()
    assert isinstance(legacy_payment, PaymentAttempt)
    assert legacy_payment.raw_payload == {"provider": "test"}


def test_payment_attempt_validates_attempt_number_and_currency():
    with pytest.raises(ValidationError):
        PaymentAttempt(
            id="attempt_1",
            merchant_id="merchant_1",
            amount=Decimal("10"),
            currency="rupees",
            status="failed",
            attempt_number=0,
        )


def test_create_recovery_case():
    case = RecoveryCase(
        merchant_id="merchant_1",
        customer_id="customer_1",
        case_type=CaseType.PAYMENT_FAILURE,
        amount_at_risk=Decimal("4999"),
        payment_id="pay_123",
        payment_method=PaymentMethod.CARD,
        failure_class=FailureClass.AUTHENTICATION_FAILURE,
        error_source="customer",
        error_step="payment_authentication",
        error_reason="incorrect_otp",
    )

    assert case.amount_at_risk == Decimal("4999")
    assert case.error_reason == "incorrect_otp"

    restored = RecoveryCase.model_validate_json(case.model_dump_json())
    assert restored == case


def test_recovery_case_preserves_probability_validation():
    with pytest.raises(ValidationError):
        RecoveryCase(
            merchant_id="merchant_1",
            case_type=CaseType.PAYMENT_FAILURE,
            amount_at_risk=Decimal("4999"),
            predicted_recovery_probability=1.01,
        )


def test_create_recovery_action():
    action = RecoveryAction(
        case_id="rc_test",
        action_type=RecoveryActionType.CREATE_PAYMENT_LINK,
        channel=CommunicationChannel.SMS,
        predicted_recovery_probability=0.82,
        expected_recovery_value=Decimal("4099.18"),
    )

    assert action.predicted_recovery_probability == 0.82


def test_create_audit_event():
    event = AuditEvent(
        case_id="rc_test",
        event_type=AuditEventType.ACTION_SELECTED,
        actor=AuditActor.ML_MODEL,
        message="Payment link selected.",
        data={
            "probability": 0.82
        },
    )

    assert event.actor == AuditActor.ML_MODEL
