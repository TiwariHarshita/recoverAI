from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from app.domain.enums import (
    CaseType,
    CommunicationChannel,
    PaymentMethod,
)


class MerchantArchetype(str, Enum):
    """
    Synthetic merchant types used only by the simulator.

    These are NOT Razorpay domain enums and therefore intentionally
    live inside the simulator package.
    """

    ECOMMERCE = "ecommerce"
    SUBSCRIPTION = "subscription"
    B2B_INVOICING = "b2b_invoicing"
    DIGITAL_SERVICES = "digital_services"


@dataclass(frozen=True)
class MerchantArchetypeAssumptions:
    """
    Behavioural assumptions used to generate a synthetic merchant
    and its customers.

    These parameters later drive case generation and the recovery
    environment, but they are not ML predictions.
    """

    average_order_value: Decimal

    payment_method_weights: dict[
        PaymentMethod,
        float,
    ]

    case_type_weights: dict[
        CaseType,
        float,
    ]

    # Beta distribution controlling historical payment success.
    customer_success_alpha: float
    customer_success_beta: float

    # Beta distribution controlling historical recovery success.
    recovery_success_alpha: float
    recovery_success_beta: float

    do_not_contact_rate: float

    max_contacts_per_case: int
    contact_window_days: int

    max_payment_retries: int

    max_recovery_window_days: int

    human_approval_threshold: Decimal

    allow_partial_payments: bool

    allow_voice_calls: bool

    allowed_channels: frozenset[
        CommunicationChannel
    ]


# Approximate mix used when generating merchants.
#
# This does NOT need to represent Razorpay's real merchant mix.
# It simply creates a heterogeneous synthetic training world.
MERCHANT_ARCHETYPE_WEIGHTS: dict[
    MerchantArchetype,
    float,
] = {
    MerchantArchetype.ECOMMERCE: 0.40,
    MerchantArchetype.SUBSCRIPTION: 0.25,
    MerchantArchetype.B2B_INVOICING: 0.20,
    MerchantArchetype.DIGITAL_SERVICES: 0.15,
}


MERCHANT_ASSUMPTIONS: dict[
    MerchantArchetype,
    MerchantArchetypeAssumptions,
] = {

    # ============================================================
    # E-COMMERCE
    # ============================================================

    MerchantArchetype.ECOMMERCE: (
        MerchantArchetypeAssumptions(
            average_order_value=Decimal("2200"),

            payment_method_weights={
                PaymentMethod.UPI: 0.50,
                PaymentMethod.CARD: 0.30,
                PaymentMethod.WALLET: 0.10,
                PaymentMethod.NETBANKING: 0.10,
            },

            case_type_weights={
                CaseType.PAYMENT_FAILURE: 0.55,
                CaseType.CHECKOUT_ABANDONMENT: 0.35,
                CaseType.SUBSCRIPTION_FAILURE: 0.05,
                CaseType.OVERDUE_INVOICE: 0.05,
            },

            customer_success_alpha=9.0,
            customer_success_beta=2.0,

            recovery_success_alpha=5.0,
            recovery_success_beta=5.0,

            do_not_contact_rate=0.02,

            max_contacts_per_case=3,
            contact_window_days=7,

            max_payment_retries=2,

            max_recovery_window_days=7,

            human_approval_threshold=Decimal(
                "25000"
            ),

            allow_partial_payments=True,

            allow_voice_calls=False,

            allowed_channels=frozenset(
                {
                    CommunicationChannel.SMS,
                    CommunicationChannel.EMAIL,
                    CommunicationChannel.WHATSAPP,
                }
            ),
        )
    ),

    # ============================================================
    # SUBSCRIPTION / SAAS
    # ============================================================

    MerchantArchetype.SUBSCRIPTION: (
        MerchantArchetypeAssumptions(
            average_order_value=Decimal("1800"),

            payment_method_weights={
                PaymentMethod.CARD: 0.55,
                PaymentMethod.UPI: 0.30,
                PaymentMethod.NETBANKING: 0.10,
                PaymentMethod.WALLET: 0.05,
            },

            case_type_weights={
                CaseType.SUBSCRIPTION_FAILURE: 0.65,
                CaseType.PAYMENT_FAILURE: 0.20,
                CaseType.CHECKOUT_ABANDONMENT: 0.10,
                CaseType.OVERDUE_INVOICE: 0.05,
            },

            customer_success_alpha=10.0,
            customer_success_beta=2.0,

            recovery_success_alpha=6.0,
            recovery_success_beta=4.0,

            do_not_contact_rate=0.015,

            max_contacts_per_case=3,
            contact_window_days=7,

            max_payment_retries=2,

            max_recovery_window_days=7,

            human_approval_threshold=Decimal(
                "25000"
            ),

            allow_partial_payments=False,

            allow_voice_calls=False,

            allowed_channels=frozenset(
                {
                    CommunicationChannel.EMAIL,
                    CommunicationChannel.SMS,
                }
            ),
        )
    ),

    # ============================================================
    # B2B INVOICING
    # ============================================================

    MerchantArchetype.B2B_INVOICING: (
        MerchantArchetypeAssumptions(
            average_order_value=Decimal(
                "75000"
            ),

            payment_method_weights={
                PaymentMethod.BANK_TRANSFER: 0.50,
                PaymentMethod.NETBANKING: 0.30,
                PaymentMethod.UPI: 0.15,
                PaymentMethod.CARD: 0.05,
            },

            case_type_weights={
                CaseType.OVERDUE_INVOICE: 0.70,
                CaseType.PAYMENT_FAILURE: 0.15,
                CaseType.SUBSCRIPTION_FAILURE: 0.10,
                CaseType.CHECKOUT_ABANDONMENT: 0.05,
            },

            customer_success_alpha=12.0,
            customer_success_beta=2.0,

            recovery_success_alpha=7.0,
            recovery_success_beta=4.0,

            do_not_contact_rate=0.01,

            max_contacts_per_case=5,
            contact_window_days=14,

            max_payment_retries=2,

            # B2B merchants need a much longer recovery window.
            max_recovery_window_days=45,

            human_approval_threshold=Decimal(
                "100000"
            ),

            allow_partial_payments=True,

            allow_voice_calls=True,

            allowed_channels=frozenset(
                {
                    CommunicationChannel.EMAIL,
                    CommunicationChannel.SMS,
                    CommunicationChannel.VOICE,
                }
            ),
        )
    ),

    # ============================================================
    # DIGITAL SERVICES
    # ============================================================

    MerchantArchetype.DIGITAL_SERVICES: (
        MerchantArchetypeAssumptions(
            average_order_value=Decimal("1200"),

            payment_method_weights={
                PaymentMethod.UPI: 0.45,
                PaymentMethod.CARD: 0.40,
                PaymentMethod.WALLET: 0.10,
                PaymentMethod.NETBANKING: 0.05,
            },

            case_type_weights={
                CaseType.PAYMENT_FAILURE: 0.45,
                CaseType.SUBSCRIPTION_FAILURE: 0.25,
                CaseType.CHECKOUT_ABANDONMENT: 0.25,
                CaseType.OVERDUE_INVOICE: 0.05,
            },

            customer_success_alpha=8.0,
            customer_success_beta=2.0,

            recovery_success_alpha=5.0,
            recovery_success_beta=5.0,

            do_not_contact_rate=0.025,

            max_contacts_per_case=3,
            contact_window_days=7,

            max_payment_retries=2,

            max_recovery_window_days=7,

            human_approval_threshold=Decimal(
                "25000"
            ),

            allow_partial_payments=True,

            allow_voice_calls=False,

            allowed_channels=frozenset(
                {
                    CommunicationChannel.SMS,
                    CommunicationChannel.EMAIL,
                    CommunicationChannel.WHATSAPP,
                }
            ),
        )
    ),
}