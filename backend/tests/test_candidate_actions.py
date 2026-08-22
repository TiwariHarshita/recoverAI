from decimal import Decimal

from app.domain.enums import (
    CaseType,
    FailureClass,
    RecoveryActionType,
)
from app.domain.recovery_case import RecoveryCase
from app.services.candidate_actions import generate_candidate_actions
from app.services.diagnosis import diagnose_case


def make_case(
    failure_class: FailureClass,
) -> RecoveryCase:
    return RecoveryCase(
        merchant_id="merchant_test",
        customer_id="customer_test",
        case_type=CaseType.PAYMENT_FAILURE,
        amount_at_risk=Decimal("5000"),
        failure_class=failure_class,
    )


def action_types(result):
    return {
        action.action_type
        for action in result.actions
    }


def test_authentication_candidates():
    case = make_case(
        FailureClass.AUTHENTICATION_FAILURE
    )

    diagnosis = diagnose_case(
        RecoveryCase(
            merchant_id="merchant_test",
            customer_id="customer_test",
            case_type=CaseType.PAYMENT_FAILURE,
            amount_at_risk=Decimal("5000"),
            error_reason="invalid_otp",
        )
    )

    result = generate_candidate_actions(
        case,
        diagnosis,
    )

    actions = action_types(result)

    assert RecoveryActionType.CREATE_PAYMENT_LINK in actions
    assert RecoveryActionType.IMMEDIATE_RETRY in actions
    assert RecoveryActionType.SEND_REMINDER in actions
    assert RecoveryActionType.WAIT in actions


def test_insufficient_funds_excludes_immediate_retry():
    case = make_case(
        FailureClass.INSUFFICIENT_FUNDS
    )

    diagnosis_case = RecoveryCase(
        merchant_id="merchant_test",
        customer_id="customer_test",
        case_type=CaseType.PAYMENT_FAILURE,
        amount_at_risk=Decimal("5000"),
        error_reason="insufficient_funds",
    )

    diagnosis = diagnose_case(diagnosis_case)

    result = generate_candidate_actions(
        case,
        diagnosis,
    )

    actions = action_types(result)

    assert RecoveryActionType.DELAYED_RETRY in actions
    assert RecoveryActionType.IMMEDIATE_RETRY not in actions


def test_expired_instrument_requires_new_method():
    diagnosis_case = RecoveryCase(
        merchant_id="merchant_test",
        customer_id="customer_test",
        case_type=CaseType.PAYMENT_FAILURE,
        amount_at_risk=Decimal("5000"),
        error_reason="card_expired",
    )

    diagnosis = diagnose_case(diagnosis_case)

    result = generate_candidate_actions(
        diagnosis_case,
        diagnosis,
    )

    actions = action_types(result)

    assert (
        RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD
        in actions
    )

    assert (
        RecoveryActionType.IMMEDIATE_RETRY
        not in actions
    )


def test_business_configuration_has_no_customer_recovery():
    diagnosis_case = RecoveryCase(
        merchant_id="merchant_test",
        customer_id="customer_test",
        case_type=CaseType.PAYMENT_FAILURE,
        amount_at_risk=Decimal("5000"),
        error_reason="payment_method_not_enabled",
        error_source="business",
    )

    diagnosis = diagnose_case(diagnosis_case)

    result = generate_candidate_actions(
        diagnosis_case,
        diagnosis,
    )

    actions = action_types(result)

    assert RecoveryActionType.ESCALATE_TO_HUMAN in actions
    assert RecoveryActionType.STOP in actions

    assert RecoveryActionType.CREATE_PAYMENT_LINK not in actions
    assert RecoveryActionType.SEND_REMINDER not in actions


def test_unknown_failure_is_conservative():
    diagnosis_case = RecoveryCase(
        merchant_id="merchant_test",
        customer_id="customer_test",
        case_type=CaseType.PAYMENT_FAILURE,
        amount_at_risk=Decimal("5000"),
        error_reason="completely_unknown_provider_reason",
    )

    diagnosis = diagnose_case(diagnosis_case)

    result = generate_candidate_actions(
        diagnosis_case,
        diagnosis,
    )

    actions = action_types(result)

    assert actions == {
        RecoveryActionType.ESCALATE_TO_HUMAN,
        RecoveryActionType.STOP,
    }


def test_checkout_abandonment_candidates():
    case = RecoveryCase(
        merchant_id="merchant_test",
        customer_id="customer_test",
        case_type=CaseType.CHECKOUT_ABANDONMENT,
        amount_at_risk=Decimal("3499"),
    )

    diagnosis = diagnose_case(case)

    result = generate_candidate_actions(
        case,
        diagnosis,
    )

    actions = action_types(result)

    assert RecoveryActionType.CREATE_PAYMENT_LINK in actions
    assert RecoveryActionType.SEND_REMINDER in actions


def test_overdue_receivable_candidates():
    case = RecoveryCase(
        merchant_id="merchant_test",
        customer_id="company_test",
        case_type=CaseType.OVERDUE_INVOICE,
        amount_at_risk=Decimal("150000"),
    )

    diagnosis = diagnose_case(case)

    result = generate_candidate_actions(
        case,
        diagnosis,
    )

    actions = action_types(result)

    assert RecoveryActionType.SEND_REMINDER in actions

    assert (
        RecoveryActionType.REQUEST_PROMISE_TO_PAY
        in actions
    )

    assert (
        RecoveryActionType.OFFER_PARTIAL_PAYMENT
        in actions
    )

    assert RecoveryActionType.ESCALATE_TO_HUMAN in actions