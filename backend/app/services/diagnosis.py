from dataclasses import dataclass
from datetime import datetime, timezone

from app.domain.diagnosis import DiagnosisEvidence, DiagnosisResult
from app.domain.enums import (
    CaseType,
    DiagnosisCertainty,
    FailureClass,
    RecoveryCaseStatus,
)
from app.domain.recovery_case import RecoveryCase


# ============================================================
# INTERNAL RULE MODEL
# ============================================================


@dataclass(frozen=True)
class DiagnosisRule:
    """
    Internal deterministic rule used to convert provider failure
    information into RecoverAI's normalized failure taxonomy.
    """

    failure_class: FailureClass

    summary: str

    temporary_failure: bool

    retry_same_method_reasonable: bool

    requires_new_payment_method: bool

    customer_action_required: bool

    merchant_action_required: bool


# ============================================================
# NORMALIZATION
# ============================================================


def normalize_token(value: str | None) -> str:
    """
    Normalize provider strings before rule matching.

    We preserve the raw values on RecoveryCase.
    This normalization is only used during matching.
    """

    if value is None:
        return ""

    return value.strip().casefold()


# ============================================================
# RULE DEFINITIONS
# ============================================================


AUTHENTICATION_RULE = DiagnosisRule(
    failure_class=FailureClass.AUTHENTICATION_FAILURE,
    summary=(
        "The payment failed during customer authentication or "
        "verification."
    ),
    temporary_failure=True,
    retry_same_method_reasonable=True,
    requires_new_payment_method=False,
    customer_action_required=True,
    merchant_action_required=False,
)


INSUFFICIENT_FUNDS_RULE = DiagnosisRule(
    failure_class=FailureClass.INSUFFICIENT_FUNDS,
    summary=(
        "The payment could not be completed because sufficient funds "
        "were not available at the time of the attempt."
    ),
    temporary_failure=True,
    retry_same_method_reasonable=True,
    requires_new_payment_method=False,
    customer_action_required=True,
    merchant_action_required=False,
)


EXPIRED_INSTRUMENT_RULE = DiagnosisRule(
    failure_class=FailureClass.EXPIRED_INSTRUMENT,
    summary=(
        "The payment instrument has expired and should not be retried "
        "without updating the payment method."
    ),
    temporary_failure=False,
    retry_same_method_reasonable=False,
    requires_new_payment_method=True,
    customer_action_required=True,
    merchant_action_required=False,
)


BLOCKED_INSTRUMENT_RULE = DiagnosisRule(
    failure_class=FailureClass.BLOCKED_INSTRUMENT,
    summary=(
        "The payment instrument appears to be blocked and cannot "
        "currently be used successfully."
    ),
    temporary_failure=False,
    retry_same_method_reasonable=False,
    requires_new_payment_method=True,
    customer_action_required=True,
    merchant_action_required=False,
)


INACTIVE_INSTRUMENT_RULE = DiagnosisRule(
    failure_class=FailureClass.INACTIVE_INSTRUMENT,
    summary=(
        "The payment instrument is not currently enabled or active "
        "for the attempted transaction."
    ),
    temporary_failure=False,
    retry_same_method_reasonable=False,
    requires_new_payment_method=True,
    customer_action_required=True,
    merchant_action_required=False,
)


TRANSACTION_LIMIT_RULE = DiagnosisRule(
    failure_class=FailureClass.TRANSACTION_LIMIT,
    summary=(
        "The attempted transaction exceeded a limit associated with "
        "the customer's payment instrument or account."
    ),
    temporary_failure=True,
    retry_same_method_reasonable=False,
    requires_new_payment_method=True,
    customer_action_required=True,
    merchant_action_required=False,
)


BANK_DECLINE_RULE = DiagnosisRule(
    failure_class=FailureClass.BANK_DECLINE,
    summary=(
        "The payment was declined by the issuing bank or payment "
        "provider."
    ),
    temporary_failure=False,
    retry_same_method_reasonable=False,
    requires_new_payment_method=True,
    customer_action_required=True,
    merchant_action_required=False,
)


RISK_DECLINE_RULE = DiagnosisRule(
    failure_class=FailureClass.RISK_DECLINE,
    summary=(
        "The transaction was declined by a payment risk-control "
        "system."
    ),
    temporary_failure=False,
    retry_same_method_reasonable=False,
    requires_new_payment_method=True,
    customer_action_required=True,
    merchant_action_required=False,
)


CUSTOMER_CANCELLED_RULE = DiagnosisRule(
    failure_class=FailureClass.CUSTOMER_CANCELLED,
    summary=(
        "The customer cancelled or exited the payment flow before "
        "completion."
    ),
    temporary_failure=True,
    retry_same_method_reasonable=True,
    requires_new_payment_method=False,
    customer_action_required=True,
    merchant_action_required=False,
)


PAYMENT_TIMEOUT_RULE = DiagnosisRule(
    failure_class=FailureClass.PAYMENT_TIMEOUT,
    summary=(
        "The payment session or payment attempt expired before it "
        "completed."
    ),
    temporary_failure=True,
    retry_same_method_reasonable=True,
    requires_new_payment_method=False,
    customer_action_required=True,
    merchant_action_required=False,
)


NETWORK_RULE = DiagnosisRule(
    failure_class=FailureClass.NETWORK_OR_GATEWAY,
    summary=(
        "The failure appears to have originated from a gateway, "
        "network, PSP or temporary banking-system issue."
    ),
    temporary_failure=True,
    retry_same_method_reasonable=True,
    requires_new_payment_method=False,
    customer_action_required=False,
    merchant_action_required=False,
)


MANDATE_FAILURE_RULE = DiagnosisRule(
    failure_class=FailureClass.MANDATE_FAILURE,
    summary=(
        "The recurring-payment mandate could not be created or "
        "processed successfully."
    ),
    temporary_failure=True,
    retry_same_method_reasonable=True,
    requires_new_payment_method=False,
    customer_action_required=True,
    merchant_action_required=False,
)


MANDATE_CANCELLED_RULE = DiagnosisRule(
    failure_class=FailureClass.MANDATE_CANCELLED,
    summary=(
        "The recurring-payment mandate is no longer available for "
        "future automatic charges."
    ),
    temporary_failure=False,
    retry_same_method_reasonable=False,
    requires_new_payment_method=True,
    customer_action_required=True,
    merchant_action_required=False,
)


BUSINESS_CONFIGURATION_RULE = DiagnosisRule(
    failure_class=FailureClass.BUSINESS_CONFIGURATION,
    summary=(
        "The payment failed because the merchant integration, request "
        "or payment configuration requires correction."
    ),
    temporary_failure=False,
    retry_same_method_reasonable=False,
    requires_new_payment_method=False,
    customer_action_required=False,
    merchant_action_required=True,
)


CHECKOUT_ABANDONMENT_RULE = DiagnosisRule(
    failure_class=FailureClass.CHECKOUT_ABANDONMENT,
    summary=(
        "The customer entered the checkout journey but no completed "
        "payment was observed within the configured abandonment window."
    ),
    temporary_failure=True,
    retry_same_method_reasonable=True,
    requires_new_payment_method=False,
    customer_action_required=True,
    merchant_action_required=False,
)


OVERDUE_RECEIVABLE_RULE = DiagnosisRule(
    failure_class=FailureClass.OVERDUE_RECEIVABLE,
    summary=(
        "The receivable has passed its due date while some or all of "
        "the amount remains unpaid."
    ),
    temporary_failure=False,
    retry_same_method_reasonable=False,
    requires_new_payment_method=False,
    customer_action_required=True,
    merchant_action_required=False,
)


UNKNOWN_RULE = DiagnosisRule(
    failure_class=FailureClass.UNKNOWN,
    summary=(
        "The available information is insufficient to map this case "
        "to a known RecoverAI failure category."
    ),
    temporary_failure=False,
    retry_same_method_reasonable=False,
    requires_new_payment_method=False,
    customer_action_required=False,
    merchant_action_required=False,
)


# ============================================================
# EXACT REASON MAPPINGS
# ============================================================

#
# These are provider/internal reason aliases.
#
# Razorpay has used both values such as `invalid_otp` in its error
# examples and `incorrect_otp` in payment entity examples, so we
# intentionally support aliases instead of assuming a single spelling.
#


REASON_RULES: dict[str, DiagnosisRule] = {

    # --------------------------------------------------------
    # Authentication / verification
    # --------------------------------------------------------

    "invalid_otp": AUTHENTICATION_RULE,
    "incorrect_otp": AUTHENTICATION_RULE,
    "authentication_failed": AUTHENTICATION_RULE,
    "otp_attempts_exceeded": AUTHENTICATION_RULE,
    "otp_expired": AUTHENTICATION_RULE,
    "incorrect_cvv": AUTHENTICATION_RULE,

    # --------------------------------------------------------
    # Insufficient funds
    # --------------------------------------------------------

    "insufficient_funds": INSUFFICIENT_FUNDS_RULE,

    # Razorpay test documentation has also exposed the singular
    # spelling in some test-card scenarios.
    "insufficient_fund": INSUFFICIENT_FUNDS_RULE,

    # Internal/simulator alias.
    "insufficient_balance": INSUFFICIENT_FUNDS_RULE,

    # --------------------------------------------------------
    # Expired instrument
    # --------------------------------------------------------

    "card_expired": EXPIRED_INSTRUMENT_RULE,
    "expired_card": EXPIRED_INSTRUMENT_RULE,

    # --------------------------------------------------------
    # Blocked instrument
    # --------------------------------------------------------

    "debit_instrument_blocked": BLOCKED_INSTRUMENT_RULE,
    "card_blocked": BLOCKED_INSTRUMENT_RULE,
    "blocked_card": BLOCKED_INSTRUMENT_RULE,

    # --------------------------------------------------------
    # Inactive / not enabled instrument
    # --------------------------------------------------------

    "card_not_enrolled": INACTIVE_INSTRUMENT_RULE,
    "debit_instrument_inactive": INACTIVE_INSTRUMENT_RULE,

    # --------------------------------------------------------
    # Transaction limits
    # --------------------------------------------------------

    "transaction_limit_exceeded": TRANSACTION_LIMIT_RULE,
    "daily_limit_exceeded": TRANSACTION_LIMIT_RULE,
    "amount_limit_exceeded": TRANSACTION_LIMIT_RULE,

    # --------------------------------------------------------
    # Customer cancelled
    # --------------------------------------------------------

    "payment_cancelled": CUSTOMER_CANCELLED_RULE,

    # --------------------------------------------------------
    # Timeout / expired payment session
    # --------------------------------------------------------

    "payment_timed_out": PAYMENT_TIMEOUT_RULE,
    "payment_session_expired": PAYMENT_TIMEOUT_RULE,
    "payment_collect_request_expired": PAYMENT_TIMEOUT_RULE,

    # --------------------------------------------------------
    # Bank / issuer decline
    # --------------------------------------------------------

    "payment_declined": BANK_DECLINE_RULE,
    "card_declined": BANK_DECLINE_RULE,

    # --------------------------------------------------------
    # Risk decline
    # --------------------------------------------------------

    "payment_risk_check_failed": RISK_DECLINE_RULE,

    # --------------------------------------------------------
    # Technical / gateway problems
    # --------------------------------------------------------

    "gateway_technical_error": NETWORK_RULE,
    "bank_not_available": NETWORK_RULE,
    "bank_technical_error": NETWORK_RULE,
    "bank_cutoff_in_progress": NETWORK_RULE,
    "authorisation_declined_by_psp": NETWORK_RULE,
    "psp_app_not_available": NETWORK_RULE,
    "payment_declined_due_to_high_traffic": NETWORK_RULE,

    # Internal/simulator aliases.
    "gateway_error": NETWORK_RULE,
    "network_error": NETWORK_RULE,
    "request_timed_out": NETWORK_RULE,

    # --------------------------------------------------------
    # Mandate failures
    # --------------------------------------------------------

    "mandate_creation_failed": MANDATE_FAILURE_RULE,
    "mandate_creation_timeout": MANDATE_FAILURE_RULE,

    # These aliases allow our simulator/normalized integrations to
    # describe an unusable mandate explicitly.
    "mandate_cancelled": MANDATE_CANCELLED_RULE,
    "mandate_revoked": MANDATE_CANCELLED_RULE,
    "mandate_expired": MANDATE_CANCELLED_RULE,

    # --------------------------------------------------------
    # Merchant / integration problem
    # --------------------------------------------------------

    "payment_method_not_enabled": BUSINESS_CONFIGURATION_RULE,
    "order_amount_mismatch": BUSINESS_CONFIGURATION_RULE,
    "invalid_amount": BUSINESS_CONFIGURATION_RULE,
    "invalid_currency": BUSINESS_CONFIGURATION_RULE,
    "invalid_order_id": BUSINESS_CONFIGURATION_RULE,
}


# ============================================================
# SOURCE GROUPS
# ============================================================


TECHNICAL_SOURCES = {
    "gateway",
    "network",
    "internal",
    "razorpay",
    "customer_psp",
    "beneficiary_bank",
}


BANK_SOURCES = {
    "issuer_bank",
    "bank",
    "issuer",
}


BUSINESS_SOURCES = {
    "business",
    "merchant",
}


# ============================================================
# RESULT BUILDER
# ============================================================


def build_result(
    rule: DiagnosisRule,
    certainty: DiagnosisCertainty,
    evidence: list[DiagnosisEvidence],
) -> DiagnosisResult:

    return DiagnosisResult(
        failure_class=rule.failure_class,
        certainty=certainty,
        summary=rule.summary,
        temporary_failure=rule.temporary_failure,
        retry_same_method_reasonable=(
            rule.retry_same_method_reasonable
        ),
        requires_new_payment_method=(
            rule.requires_new_payment_method
        ),
        customer_action_required=(
            rule.customer_action_required
        ),
        merchant_action_required=(
            rule.merchant_action_required
        ),
        evidence=evidence,
    )


# ============================================================
# SPECIAL HANDLING FOR GENERIC PAYMENT_FAILED
# ============================================================


def diagnose_generic_payment_failed(
    source: str,
    reason: str,
) -> DiagnosisResult:

    evidence = [
        DiagnosisEvidence(
            field="error_reason",
            value=reason,
            note=(
                "The provider returned the generic "
                "'payment_failed' reason."
            ),
        )
    ]

    if source:
        evidence.append(
            DiagnosisEvidence(
                field="error_source",
                value=source,
                note=(
                    "The source is used to refine the generic "
                    "payment failure."
                ),
            )
        )

    if source in BUSINESS_SOURCES:
        return build_result(
            BUSINESS_CONFIGURATION_RULE,
            DiagnosisCertainty.INFERRED,
            evidence,
        )

    if source in BANK_SOURCES:
        return build_result(
            BANK_DECLINE_RULE,
            DiagnosisCertainty.INFERRED,
            evidence,
        )

    if source in TECHNICAL_SOURCES:
        return build_result(
            NETWORK_RULE,
            DiagnosisCertainty.INFERRED,
            evidence,
        )

    return build_result(
        UNKNOWN_RULE,
        DiagnosisCertainty.UNKNOWN,
        evidence,
    )


# ============================================================
# MAIN DIAGNOSIS FUNCTION
# ============================================================


def diagnose_case(case: RecoveryCase) -> DiagnosisResult:
    """
    Diagnose a RecoverAI RecoveryCase.

    Priority:

    1. Case types that already identify the revenue problem.
    2. Exact provider error_reason.
    3. Special handling for generic `payment_failed`.
    4. Infer from error_step.
    5. Infer from error_source.
    6. UNKNOWN fallback.

    Raw provider information is never modified.
    """

    # --------------------------------------------------------
    # Case-level diagnoses
    # --------------------------------------------------------

    if case.case_type == CaseType.CHECKOUT_ABANDONMENT:
        return build_result(
            CHECKOUT_ABANDONMENT_RULE,
            DiagnosisCertainty.EXACT,
            [
                DiagnosisEvidence(
                    field="case_type",
                    value=case.case_type.value,
                    note=(
                        "The recovery case itself was created from "
                        "checkout abandonment detection."
                    ),
                )
            ],
        )

    if case.case_type == CaseType.OVERDUE_INVOICE:
        return build_result(
            OVERDUE_RECEIVABLE_RULE,
            DiagnosisCertainty.EXACT,
            [
                DiagnosisEvidence(
                    field="case_type",
                    value=case.case_type.value,
                    note=(
                        "The recovery case itself represents an "
                        "overdue receivable."
                    ),
                )
            ],
        )

    # --------------------------------------------------------
    # Normalize provider fields for matching only.
    # --------------------------------------------------------

    reason = normalize_token(case.error_reason)

    step = normalize_token(case.error_step)

    source = normalize_token(case.error_source)

    # --------------------------------------------------------
    # Exact reason mapping
    # --------------------------------------------------------

    if reason in REASON_RULES:
        rule = REASON_RULES[reason]

        return build_result(
            rule,
            DiagnosisCertainty.EXACT,
            [
                DiagnosisEvidence(
                    field="error_reason",
                    value=case.error_reason or "",
                    note=(
                        "Matched an explicit known failure reason."
                    ),
                )
            ],
        )

    # --------------------------------------------------------
    # Generic gateway/bank payment_failed needs its source
    # to make a useful distinction.
    # --------------------------------------------------------

    if reason == "payment_failed":
        return diagnose_generic_payment_failed(
            source=source,
            reason=reason,
        )

    # --------------------------------------------------------
    # Infer authentication only when no exact reason matched.
    # --------------------------------------------------------

    if step == "payment_authentication":
        return build_result(
            AUTHENTICATION_RULE,
            DiagnosisCertainty.INFERRED,
            [
                DiagnosisEvidence(
                    field="error_step",
                    value=case.error_step or "",
                    note=(
                        "No known exact reason matched, but the "
                        "failure occurred during payment authentication."
                    ),
                )
            ],
        )

    # --------------------------------------------------------
    # Merchant/integration source
    # --------------------------------------------------------

    if source in BUSINESS_SOURCES:
        return build_result(
            BUSINESS_CONFIGURATION_RULE,
            DiagnosisCertainty.INFERRED,
            [
                DiagnosisEvidence(
                    field="error_source",
                    value=case.error_source or "",
                    note=(
                        "The provider identified the merchant/business "
                        "integration as the source of the failure."
                    ),
                )
            ],
        )

    # --------------------------------------------------------
    # Technical/network/gateway source
    # --------------------------------------------------------

    if source in TECHNICAL_SOURCES:
        return build_result(
            NETWORK_RULE,
            DiagnosisCertainty.INFERRED,
            [
                DiagnosisEvidence(
                    field="error_source",
                    value=case.error_source or "",
                    note=(
                        "The provider identified a technical payment "
                        "participant as the source of the failure."
                    ),
                )
            ],
        )

    # --------------------------------------------------------
    # Bank / issuer source
    # --------------------------------------------------------

    if source in BANK_SOURCES:
        return build_result(
            BANK_DECLINE_RULE,
            DiagnosisCertainty.INFERRED,
            [
                DiagnosisEvidence(
                    field="error_source",
                    value=case.error_source or "",
                    note=(
                        "No more-specific reason matched, but the "
                        "failure originated at the bank or issuer."
                    ),
                )
            ],
        )

    # --------------------------------------------------------
    # Unknown
    # --------------------------------------------------------

    evidence: list[DiagnosisEvidence] = []

    if case.error_reason:
        evidence.append(
            DiagnosisEvidence(
                field="error_reason",
                value=case.error_reason,
                note="No diagnosis rule currently matches this reason.",
            )
        )

    if case.error_step:
        evidence.append(
            DiagnosisEvidence(
                field="error_step",
                value=case.error_step,
                note="No diagnosis rule currently matches this step.",
            )
        )

    if case.error_source:
        evidence.append(
            DiagnosisEvidence(
                field="error_source",
                value=case.error_source,
                note="No diagnosis rule currently matches this source.",
            )
        )

    return build_result(
        UNKNOWN_RULE,
        DiagnosisCertainty.UNKNOWN,
        evidence,
    )


# ============================================================
# APPLY DIAGNOSIS TO CASE
# ============================================================


def apply_diagnosis(
    case: RecoveryCase,
) -> DiagnosisResult:
    """
    Diagnose the case and apply the normalized classification to
    the RecoveryCase.

    This function does NOT destroy or replace the original Razorpay
    failure information.
    """

    result = diagnose_case(case)

    case.failure_class = result.failure_class

    case.status = RecoveryCaseStatus.DIAGNOSED

    case.updated_at = datetime.now(timezone.utc)

    return result