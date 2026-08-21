from decimal import Decimal

from app.domain.enums import (
    CaseType,
    DiagnosisCertainty,
    FailureClass,
    PaymentMethod,
    RecoveryCaseStatus,
)
from app.domain.recovery_case import RecoveryCase
from app.services.diagnosis import (
    apply_diagnosis,
    diagnose_case,
)


def make_payment_case(
    *,
    error_reason: str | None = None,
    error_step: str | None = None,
    error_source: str | None = None,
) -> RecoveryCase:

    return RecoveryCase(
        merchant_id="merchant_test",
        customer_id="customer_test",
        case_type=CaseType.PAYMENT_FAILURE,
        amount_at_risk=Decimal("4999"),
        payment_method=PaymentMethod.CARD,
        error_reason=error_reason,
        error_step=error_step,
        error_source=error_source,
    )


# ============================================================
# EXACT REASON TESTS
# ============================================================


def test_invalid_otp_exact_match():
    case = make_payment_case(
        error_reason="invalid_otp",
        error_step="payment_authentication",
        error_source="customer",
    )

    result = diagnose_case(case)

    assert (
        result.failure_class
        == FailureClass.AUTHENTICATION_FAILURE
    )

    assert result.certainty == DiagnosisCertainty.EXACT

    assert result.temporary_failure is True

    assert result.retry_same_method_reasonable is True

    assert result.requires_new_payment_method is False


def test_incorrect_otp_alias():
    case = make_payment_case(
        error_reason="incorrect_otp"
    )

    result = diagnose_case(case)

    assert (
        result.failure_class
        == FailureClass.AUTHENTICATION_FAILURE
    )


def test_insufficient_funds():
    case = make_payment_case(
        error_reason="insufficient_funds"
    )

    result = diagnose_case(case)

    assert (
        result.failure_class
        == FailureClass.INSUFFICIENT_FUNDS
    )

    assert result.temporary_failure is True


def test_card_expired_requires_new_method():
    case = make_payment_case(
        error_reason="card_expired"
    )

    result = diagnose_case(case)

    assert (
        result.failure_class
        == FailureClass.EXPIRED_INSTRUMENT
    )

    assert result.retry_same_method_reasonable is False

    assert result.requires_new_payment_method is True


def test_blocked_card():
    case = make_payment_case(
        error_reason="debit_instrument_blocked"
    )

    result = diagnose_case(case)

    assert (
        result.failure_class
        == FailureClass.BLOCKED_INSTRUMENT
    )

    assert result.requires_new_payment_method is True


def test_inactive_card():
    case = make_payment_case(
        error_reason="card_not_enrolled"
    )

    result = diagnose_case(case)

    assert (
        result.failure_class
        == FailureClass.INACTIVE_INSTRUMENT
    )


def test_transaction_limit():
    case = make_payment_case(
        error_reason="transaction_limit_exceeded"
    )

    result = diagnose_case(case)

    assert (
        result.failure_class
        == FailureClass.TRANSACTION_LIMIT
    )

    assert result.requires_new_payment_method is True


def test_payment_cancelled():
    case = make_payment_case(
        error_reason="payment_cancelled",
        error_step="payment_authentication",
    )

    result = diagnose_case(case)

    # Exact reason MUST take precedence over the broader
    # payment_authentication step.
    assert (
        result.failure_class
        == FailureClass.CUSTOMER_CANCELLED
    )

    assert result.certainty == DiagnosisCertainty.EXACT


def test_payment_timeout():
    case = make_payment_case(
        error_reason="payment_timed_out"
    )

    result = diagnose_case(case)

    assert (
        result.failure_class
        == FailureClass.PAYMENT_TIMEOUT
    )

    assert result.temporary_failure is True


def test_gateway_technical_error():
    case = make_payment_case(
        error_reason="gateway_technical_error",
        error_source="gateway",
    )

    result = diagnose_case(case)

    assert (
        result.failure_class
        == FailureClass.NETWORK_OR_GATEWAY
    )

    assert result.certainty == DiagnosisCertainty.EXACT


def test_risk_decline():
    case = make_payment_case(
        error_reason="payment_risk_check_failed"
    )

    result = diagnose_case(case)

    assert (
        result.failure_class
        == FailureClass.RISK_DECLINE
    )


def test_payment_method_not_enabled_is_merchant_problem():
    case = make_payment_case(
        error_reason="payment_method_not_enabled",
        error_source="business",
    )

    result = diagnose_case(case)

    assert (
        result.failure_class
        == FailureClass.BUSINESS_CONFIGURATION
    )

    assert result.merchant_action_required is True

    assert result.customer_action_required is False


# ============================================================
# GENERIC PAYMENT_FAILED
# ============================================================


def test_generic_payment_failed_from_issuer():
    case = make_payment_case(
        error_reason="payment_failed",
        error_source="issuer_bank",
    )

    result = diagnose_case(case)

    assert (
        result.failure_class
        == FailureClass.BANK_DECLINE
    )

    assert result.certainty == DiagnosisCertainty.INFERRED


def test_generic_payment_failed_from_gateway():
    case = make_payment_case(
        error_reason="payment_failed",
        error_source="gateway",
    )

    result = diagnose_case(case)

    assert (
        result.failure_class
        == FailureClass.NETWORK_OR_GATEWAY
    )

    assert result.certainty == DiagnosisCertainty.INFERRED


def test_generic_payment_failed_from_business():
    case = make_payment_case(
        error_reason="payment_failed",
        error_source="business",
    )

    result = diagnose_case(case)

    assert (
        result.failure_class
        == FailureClass.BUSINESS_CONFIGURATION
    )


# ============================================================
# INFERENCE FALLBACKS
# ============================================================


def test_infer_authentication_from_step():
    case = make_payment_case(
        error_reason="some_future_unknown_reason",
        error_step="payment_authentication",
    )

    result = diagnose_case(case)

    assert (
        result.failure_class
        == FailureClass.AUTHENTICATION_FAILURE
    )

    assert result.certainty == DiagnosisCertainty.INFERRED


def test_infer_network_from_source():
    case = make_payment_case(
        error_source="network"
    )

    result = diagnose_case(case)

    assert (
        result.failure_class
        == FailureClass.NETWORK_OR_GATEWAY
    )

    assert result.certainty == DiagnosisCertainty.INFERRED


def test_infer_merchant_problem_from_source():
    case = make_payment_case(
        error_source="business"
    )

    result = diagnose_case(case)

    assert (
        result.failure_class
        == FailureClass.BUSINESS_CONFIGURATION
    )

    assert result.merchant_action_required is True


# ============================================================
# CASE TYPE DIAGNOSIS
# ============================================================


def test_checkout_abandonment():
    case = RecoveryCase(
        merchant_id="merchant_test",
        customer_id="customer_test",
        case_type=CaseType.CHECKOUT_ABANDONMENT,
        amount_at_risk=Decimal("3499"),
    )

    result = diagnose_case(case)

    assert (
        result.failure_class
        == FailureClass.CHECKOUT_ABANDONMENT
    )

    assert result.certainty == DiagnosisCertainty.EXACT


def test_overdue_invoice():
    case = RecoveryCase(
        merchant_id="merchant_test",
        customer_id="company_test",
        case_type=CaseType.OVERDUE_INVOICE,
        amount_at_risk=Decimal("150000"),
    )

    result = diagnose_case(case)

    assert (
        result.failure_class
        == FailureClass.OVERDUE_RECEIVABLE
    )


# ============================================================
# NORMALIZATION
# ============================================================


def test_whitespace_and_case_normalization():
    case = make_payment_case(
        error_reason="  INVALID_OTP  "
    )

    result = diagnose_case(case)

    assert (
        result.failure_class
        == FailureClass.AUTHENTICATION_FAILURE
    )


# ============================================================
# UNKNOWN
# ============================================================


def test_unknown_failure():
    case = make_payment_case(
        error_reason="brand_new_provider_error_123"
    )

    result = diagnose_case(case)

    assert result.failure_class == FailureClass.UNKNOWN

    assert result.certainty == DiagnosisCertainty.UNKNOWN

    assert len(result.evidence) == 1


def test_completely_missing_failure_information():
    case = make_payment_case()

    result = diagnose_case(case)

    assert result.failure_class == FailureClass.UNKNOWN

    assert result.certainty == DiagnosisCertainty.UNKNOWN

    assert result.evidence == []


# ============================================================
# APPLY DIAGNOSIS
# ============================================================


def test_apply_diagnosis_updates_case():
    case = make_payment_case(
        error_reason="insufficient_funds"
    )

    assert case.failure_class == FailureClass.UNKNOWN

    assert case.status == RecoveryCaseStatus.OPEN

    result = apply_diagnosis(case)

    assert (
        result.failure_class
        == FailureClass.INSUFFICIENT_FUNDS
    )

    assert (
        case.failure_class
        == FailureClass.INSUFFICIENT_FUNDS
    )

    assert case.status == RecoveryCaseStatus.DIAGNOSED


def test_raw_provider_values_are_preserved():
    case = make_payment_case(
        error_reason="  INVALID_OTP  ",
        error_step="payment_authentication",
        error_source="customer",
    )

    apply_diagnosis(case)

    # Diagnosis normalizes internally for matching.
    # Raw provider evidence must remain untouched.
    assert case.error_reason == "  INVALID_OTP  "

    assert case.error_step == "payment_authentication"

    assert case.error_source == "customer"