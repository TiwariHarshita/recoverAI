from __future__ import annotations

from decimal import (
    Decimal,
    ROUND_HALF_UP,
)
from random import Random

from pydantic import (
    BaseModel,
    Field,
)

from app.domain.enums import (
    CaseType,
    PaymentMethod,
    RecoveryActionType,
)
from app.domain.policies import (
    MerchantPolicy,
)

from simulator.assumptions import (
    MERCHANT_ARCHETYPE_WEIGHTS,
    MERCHANT_ASSUMPTIONS,
    MerchantArchetype,
)
from simulator.random_utils import (
    weighted_choice,
)


class SyntheticMerchant(BaseModel):
    """
    Simulator-only merchant profile.

    We intentionally do not add a Merchant model to app.domain yet.

    The real domain object we already have is MerchantPolicy.
    SyntheticMerchant simply adds behavioural parameters needed
    to generate customers and recovery cases.
    """

    id: str

    name: str

    archetype: MerchantArchetype

    policy: MerchantPolicy

    average_order_value: Decimal = Field(
        ge=0
    )

    payment_method_weights: dict[
        PaymentMethod,
        float,
    ]

    case_type_weights: dict[
        CaseType,
        float,
    ]

    customer_success_alpha: float = Field(
        gt=0
    )

    customer_success_beta: float = Field(
        gt=0
    )

    recovery_success_alpha: float = Field(
        gt=0
    )

    recovery_success_beta: float = Field(
        gt=0
    )

    do_not_contact_rate: float = Field(
        ge=0,
        le=1,
    )


def generate_merchants(
    count: int,
    *,
    seed: int = 42,
) -> list[SyntheticMerchant]:
    """
    Generate a deterministic synthetic merchant population.

    Same count + same seed always produces the same merchants.
    """

    if count <= 0:
        raise ValueError(
            "count must be greater than zero."
        )

    rng = Random(seed)

    merchants: list[
        SyntheticMerchant
    ] = []

    for index in range(
        1,
        count + 1,
    ):
        archetype = weighted_choice(
            rng,
            MERCHANT_ARCHETYPE_WEIGHTS,
        )

        assumptions = (
            MERCHANT_ASSUMPTIONS[
                archetype
            ]
        )

        merchant_id = (
            f"merchant_{index:04d}"
        )

        # Give merchants within one archetype slightly different
        # transaction sizes instead of making them identical.
        order_value_multiplier = Decimal(
            str(
                rng.uniform(
                    0.85,
                    1.15,
                )
            )
        )

        average_order_value = (
            assumptions.average_order_value
            * order_value_multiplier
        ).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )

        policy = MerchantPolicy(
            merchant_id=merchant_id,

            max_contacts_per_case=(
                assumptions.max_contacts_per_case
            ),

            contact_window_days=(
                assumptions.contact_window_days
            ),

            max_payment_retries=(
                assumptions.max_payment_retries
            ),

            max_recovery_window_days=(
                assumptions.max_recovery_window_days
            ),

            human_approval_threshold=(
                assumptions.human_approval_threshold
            ),

            allow_partial_payments=(
                assumptions.allow_partial_payments
            ),

            allow_voice_calls=(
                assumptions.allow_voice_calls
            ),

            timezone="Asia/Kolkata",

            allowed_channels=set(
                assumptions.allowed_channels
            ),

            allowed_actions=set(
                RecoveryActionType
            ),
        )

        merchant = SyntheticMerchant(
            id=merchant_id,

            name=(
                "Synthetic "
                f"{archetype.value.replace('_', ' ').title()} "
                f"{index:04d}"
            ),

            archetype=archetype,

            policy=policy,

            average_order_value=(
                average_order_value
            ),

            payment_method_weights=dict(
                assumptions.payment_method_weights
            ),

            case_type_weights=dict(
                assumptions.case_type_weights
            ),

            customer_success_alpha=(
                assumptions.customer_success_alpha
            ),

            customer_success_beta=(
                assumptions.customer_success_beta
            ),

            recovery_success_alpha=(
                assumptions.recovery_success_alpha
            ),

            recovery_success_beta=(
                assumptions.recovery_success_beta
            ),

            do_not_contact_rate=(
                assumptions.do_not_contact_rate
            ),
        )

        merchants.append(
            merchant
        )

    return merchants