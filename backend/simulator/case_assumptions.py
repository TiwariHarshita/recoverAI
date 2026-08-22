from dataclasses import dataclass

from app.domain.enums import (
    FailureClass,
    PaymentMethod,
)


@dataclass(frozen=True)
class FailureTemplate:
    """
    Provider-like failure payload used by the synthetic
    RecoveryCase generator.

    The normalized FailureClass is simulator ground truth.
    RecoveryCase itself will still start with FailureClass.UNKNOWN.
    """

    failure_class: FailureClass

    error_reason: str

    error_code: str

    error_source: str

    error_step: str

    error_description: str

    compatible_methods: frozenset[
        PaymentMethod
    ]


# ============================================================
# PAYMENT-METHOD GROUPS
# ============================================================


CARD_LIKE = frozenset(
    {
        PaymentMethod.CARD,
        PaymentMethod.EMI,
    }
)


BANKED_METHODS = frozenset(
    {
        PaymentMethod.CARD,
        PaymentMethod.UPI,
        PaymentMethod.NETBANKING,
        PaymentMethod.EMI,
        PaymentMethod.BANK_TRANSFER,
    }
)


ALL_ACTIVE_METHODS = frozenset(
    {
        PaymentMethod.CARD,
        PaymentMethod.UPI,
        PaymentMethod.NETBANKING,
        PaymentMethod.WALLET,
        PaymentMethod.EMI,
        PaymentMethod.BANK_TRANSFER,
    }
)


MANDATE_METHODS = frozenset(
    {
        PaymentMethod.CARD,
        PaymentMethod.UPI,
    }
)


# ============================================================
# FAILURE DISTRIBUTIONS
# ============================================================

# Synthetic assumptions only.
#
# These are not claimed to be Razorpay production distributions.
# Their job is to create a varied and believable simulated world.


PAYMENT_FAILURE_WEIGHTS: dict[
    FailureClass,
    float,
] = {
    FailureClass.AUTHENTICATION_FAILURE: 0.16,
    FailureClass.INSUFFICIENT_FUNDS: 0.18,
    FailureClass.EXPIRED_INSTRUMENT: 0.06,
    FailureClass.BLOCKED_INSTRUMENT: 0.04,
    FailureClass.INACTIVE_INSTRUMENT: 0.03,
    FailureClass.TRANSACTION_LIMIT: 0.05,
    FailureClass.BANK_DECLINE: 0.12,
    FailureClass.RISK_DECLINE: 0.05,
    FailureClass.CUSTOMER_CANCELLED: 0.07,
    FailureClass.PAYMENT_TIMEOUT: 0.08,
    FailureClass.NETWORK_OR_GATEWAY: 0.11,
    FailureClass.BUSINESS_CONFIGURATION: 0.03,
    FailureClass.UNKNOWN: 0.02,
}


SUBSCRIPTION_FAILURE_WEIGHTS: dict[
    FailureClass,
    float,
] = {
    FailureClass.MANDATE_FAILURE: 0.32,
    FailureClass.MANDATE_CANCELLED: 0.15,
    FailureClass.INSUFFICIENT_FUNDS: 0.20,
    FailureClass.EXPIRED_INSTRUMENT: 0.08,
    FailureClass.BANK_DECLINE: 0.08,
    FailureClass.AUTHENTICATION_FAILURE: 0.05,
    FailureClass.NETWORK_OR_GATEWAY: 0.07,
    FailureClass.TRANSACTION_LIMIT: 0.03,
    FailureClass.UNKNOWN: 0.02,
}


# ============================================================
# PROVIDER-LIKE ERROR TEMPLATES
# ============================================================


FAILURE_TEMPLATES: dict[
    FailureClass,
    tuple[FailureTemplate, ...],
] = {

    # --------------------------------------------------------
    # Authentication
    # --------------------------------------------------------

    FailureClass.AUTHENTICATION_FAILURE: (
        FailureTemplate(
            failure_class=(
                FailureClass.AUTHENTICATION_FAILURE
            ),
            error_reason="invalid_otp",
            error_code="BAD_REQUEST_ERROR",
            error_source="customer",
            error_step="payment_authentication",
            error_description=(
                "The OTP entered by the customer was invalid."
            ),
            compatible_methods=frozenset(
                {
                    PaymentMethod.CARD,
                    PaymentMethod.UPI,
                }
            ),
        ),
        FailureTemplate(
            failure_class=(
                FailureClass.AUTHENTICATION_FAILURE
            ),
            error_reason="incorrect_cvv",
            error_code="BAD_REQUEST_ERROR",
            error_source="customer",
            error_step="payment_authentication",
            error_description=(
                "The card verification value was incorrect."
            ),
            compatible_methods=CARD_LIKE,
        ),
    ),

    # --------------------------------------------------------
    # Insufficient funds
    # --------------------------------------------------------

    FailureClass.INSUFFICIENT_FUNDS: (
        FailureTemplate(
            failure_class=(
                FailureClass.INSUFFICIENT_FUNDS
            ),
            error_reason="insufficient_funds",
            error_code="BAD_REQUEST_ERROR",
            error_source="issuer_bank",
            error_step="payment_authorization",
            error_description=(
                "The customer did not have sufficient "
                "available funds."
            ),
            compatible_methods=BANKED_METHODS,
        ),
    ),

    # --------------------------------------------------------
    # Expired instrument
    # --------------------------------------------------------

    FailureClass.EXPIRED_INSTRUMENT: (
        FailureTemplate(
            failure_class=(
                FailureClass.EXPIRED_INSTRUMENT
            ),
            error_reason="card_expired",
            error_code="BAD_REQUEST_ERROR",
            error_source="issuer_bank",
            error_step="payment_authorization",
            error_description=(
                "The payment card has expired."
            ),
            compatible_methods=CARD_LIKE,
        ),
    ),

    # --------------------------------------------------------
    # Blocked instrument
    # --------------------------------------------------------

    FailureClass.BLOCKED_INSTRUMENT: (
        FailureTemplate(
            failure_class=(
                FailureClass.BLOCKED_INSTRUMENT
            ),
            error_reason=(
                "debit_instrument_blocked"
            ),
            error_code="BAD_REQUEST_ERROR",
            error_source="issuer_bank",
            error_step="payment_authorization",
            error_description=(
                "The payment instrument is blocked "
                "by the issuer."
            ),
            compatible_methods=BANKED_METHODS,
        ),
    ),

    # --------------------------------------------------------
    # Inactive instrument
    # --------------------------------------------------------

    FailureClass.INACTIVE_INSTRUMENT: (
        FailureTemplate(
            failure_class=(
                FailureClass.INACTIVE_INSTRUMENT
            ),
            error_reason=(
                "debit_instrument_inactive"
            ),
            error_code="BAD_REQUEST_ERROR",
            error_source="issuer_bank",
            error_step="payment_authorization",
            error_description=(
                "The payment instrument is not active "
                "for this transaction."
            ),
            compatible_methods=BANKED_METHODS,
        ),
    ),

    # --------------------------------------------------------
    # Transaction limit
    # --------------------------------------------------------

    FailureClass.TRANSACTION_LIMIT: (
        FailureTemplate(
            failure_class=(
                FailureClass.TRANSACTION_LIMIT
            ),
            error_reason=(
                "transaction_limit_exceeded"
            ),
            error_code="BAD_REQUEST_ERROR",
            error_source="issuer_bank",
            error_step="payment_authorization",
            error_description=(
                "The attempted payment exceeded an "
                "account or instrument limit."
            ),
            compatible_methods=BANKED_METHODS,
        ),
    ),

    # --------------------------------------------------------
    # Bank decline
    # --------------------------------------------------------

    FailureClass.BANK_DECLINE: (
        FailureTemplate(
            failure_class=(
                FailureClass.BANK_DECLINE
            ),
            error_reason="payment_declined",
            error_code="BAD_REQUEST_ERROR",
            error_source="issuer_bank",
            error_step="payment_authorization",
            error_description=(
                "The issuing bank declined the payment."
            ),
            compatible_methods=BANKED_METHODS,
        ),
        FailureTemplate(
            failure_class=(
                FailureClass.BANK_DECLINE
            ),
            error_reason="payment_failed",
            error_code="BAD_REQUEST_ERROR",
            error_source="issuer_bank",
            error_step="payment_authorization",
            error_description=(
                "The bank returned a generic "
                "payment failure."
            ),
            compatible_methods=BANKED_METHODS,
        ),
    ),

    # --------------------------------------------------------
    # Risk decline
    # --------------------------------------------------------

    FailureClass.RISK_DECLINE: (
        FailureTemplate(
            failure_class=(
                FailureClass.RISK_DECLINE
            ),
            error_reason=(
                "payment_risk_check_failed"
            ),
            error_code="BAD_REQUEST_ERROR",
            error_source="razorpay",
            error_step="payment_authorization",
            error_description=(
                "The transaction was rejected by a "
                "risk-control check."
            ),
            compatible_methods=ALL_ACTIVE_METHODS,
        ),
    ),

    # --------------------------------------------------------
    # Customer cancelled
    # --------------------------------------------------------

    FailureClass.CUSTOMER_CANCELLED: (
        FailureTemplate(
            failure_class=(
                FailureClass.CUSTOMER_CANCELLED
            ),
            error_reason="payment_cancelled",
            error_code="BAD_REQUEST_ERROR",
            error_source="customer",
            error_step="payment_authentication",
            error_description=(
                "The customer cancelled the payment flow."
            ),
            compatible_methods=ALL_ACTIVE_METHODS,
        ),
    ),

    # --------------------------------------------------------
    # Timeout
    # --------------------------------------------------------

    FailureClass.PAYMENT_TIMEOUT: (
        FailureTemplate(
            failure_class=(
                FailureClass.PAYMENT_TIMEOUT
            ),
            error_reason="payment_timed_out",
            error_code="GATEWAY_ERROR",
            error_source="gateway",
            error_step="payment_processing",
            error_description=(
                "The payment attempt timed out "
                "before completion."
            ),
            compatible_methods=ALL_ACTIVE_METHODS,
        ),
    ),

    # --------------------------------------------------------
    # Network / gateway
    # --------------------------------------------------------

    FailureClass.NETWORK_OR_GATEWAY: (
        FailureTemplate(
            failure_class=(
                FailureClass.NETWORK_OR_GATEWAY
            ),
            error_reason=(
                "gateway_technical_error"
            ),
            error_code="GATEWAY_ERROR",
            error_source="gateway",
            error_step="payment_processing",
            error_description=(
                "A temporary gateway error interrupted "
                "the payment."
            ),
            compatible_methods=ALL_ACTIVE_METHODS,
        ),
        FailureTemplate(
            failure_class=(
                FailureClass.NETWORK_OR_GATEWAY
            ),
            error_reason="payment_failed",
            error_code="GATEWAY_ERROR",
            error_source="gateway",
            error_step="payment_processing",
            error_description=(
                "The gateway returned a generic "
                "payment failure."
            ),
            compatible_methods=ALL_ACTIVE_METHODS,
        ),
    ),

    # --------------------------------------------------------
    # Mandate failure
    # --------------------------------------------------------

    FailureClass.MANDATE_FAILURE: (
        FailureTemplate(
            failure_class=(
                FailureClass.MANDATE_FAILURE
            ),
            error_reason=(
                "mandate_creation_failed"
            ),
            error_code="BAD_REQUEST_ERROR",
            error_source="customer_psp",
            error_step="mandate_creation",
            error_description=(
                "The recurring-payment mandate "
                "could not be processed."
            ),
            compatible_methods=MANDATE_METHODS,
        ),
    ),

    # --------------------------------------------------------
    # Mandate cancelled
    # --------------------------------------------------------

    FailureClass.MANDATE_CANCELLED: (
        FailureTemplate(
            failure_class=(
                FailureClass.MANDATE_CANCELLED
            ),
            error_reason="mandate_cancelled",
            error_code="BAD_REQUEST_ERROR",
            error_source="customer",
            error_step="mandate_processing",
            error_description=(
                "The recurring-payment mandate "
                "was cancelled."
            ),
            compatible_methods=MANDATE_METHODS,
        ),
    ),

    # --------------------------------------------------------
    # Merchant configuration
    # --------------------------------------------------------

    FailureClass.BUSINESS_CONFIGURATION: (
        FailureTemplate(
            failure_class=(
                FailureClass.BUSINESS_CONFIGURATION
            ),
            error_reason=(
                "payment_method_not_enabled"
            ),
            error_code="BAD_REQUEST_ERROR",
            error_source="business",
            error_step="payment_initiation",
            error_description=(
                "The requested payment method is not "
                "enabled for the merchant."
            ),
            compatible_methods=ALL_ACTIVE_METHODS,
        ),
        FailureTemplate(
            failure_class=(
                FailureClass.BUSINESS_CONFIGURATION
            ),
            error_reason="payment_failed",
            error_code="BAD_REQUEST_ERROR",
            error_source="business",
            error_step="payment_initiation",
            error_description=(
                "The merchant integration returned "
                "a generic payment failure."
            ),
            compatible_methods=ALL_ACTIVE_METHODS,
        ),
    ),

    # --------------------------------------------------------
    # Unknown
    # --------------------------------------------------------

    FailureClass.UNKNOWN: (
        FailureTemplate(
            failure_class=FailureClass.UNKNOWN,
            error_reason=(
                "provider_unclassified_error"
            ),
            error_code="UNKNOWN_ERROR",
            error_source="external_provider",
            error_step="payment_processing",
            error_description=(
                "The provider returned an "
                "unclassified payment error."
            ),
            compatible_methods=ALL_ACTIVE_METHODS,
        ),
    ),
}


# ============================================================
# SYNTHETIC BANK POOLS
# ============================================================


BANKS_BY_METHOD: dict[
    PaymentMethod,
    tuple[str, ...],
] = {

    PaymentMethod.CARD: (
        "HDFC",
        "ICICI",
        "SBI",
        "AXIS",
        "KOTAK",
    ),

    PaymentMethod.UPI: (
        "HDFC",
        "ICICI",
        "SBI",
        "AXIS",
        "YES_BANK",
    ),

    PaymentMethod.NETBANKING: (
        "HDFC",
        "ICICI",
        "SBI",
        "AXIS",
        "KOTAK",
    ),

    PaymentMethod.EMI: (
        "HDFC",
        "ICICI",
        "SBI",
        "AXIS",
    ),

    PaymentMethod.BANK_TRANSFER: (
        "HDFC",
        "ICICI",
        "SBI",
        "AXIS",
        "KOTAK",
    ),
}