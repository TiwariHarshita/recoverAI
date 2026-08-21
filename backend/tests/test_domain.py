from decimal import Decimal

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