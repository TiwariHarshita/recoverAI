from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from decimal import (
    Decimal,
    ROUND_HALF_UP,
)
from random import Random

from pydantic import (
    BaseModel,
    Field,
)

from app.domain.customer import (
    Customer,
)
from app.domain.enums import (
    CommunicationChannel,
)

from simulator.merchants import (
    SyntheticMerchant,
    generate_merchants,
)
from simulator.random_utils import (
    weighted_choice,
)


# Fixed by default so the same seed creates exactly the same dataset.
# Later the RecoveryEnvironment can pass its own simulation clock.
DEFAULT_REFERENCE_TIME = datetime(
    2026,
    8,
    23,
    0,
    0,
    tzinfo=timezone.utc,
)


class SyntheticPopulation(BaseModel):
    """
    Complete merchant/customer population produced by one
    deterministic generation run.
    """

    seed: int

    reference_time: datetime

    merchants: list[
        SyntheticMerchant
    ] = Field(
        default_factory=list
    )

    customers: list[
        Customer
    ] = Field(
        default_factory=list
    )


def _choose_preferred_channel(
    rng: Random,
    merchant: SyntheticMerchant,
) -> CommunicationChannel:
    """
    Pick a preferred channel only from channels allowed by
    the merchant.
    """

    # Sorting is important. MerchantPolicy.allowed_channels is a set,
    # and set iteration order must not make seeded generation vary
    # across Python processes.
    available = sorted(
        (
            channel
            for channel
            in merchant.policy.allowed_channels
            if channel
            != CommunicationChannel.NONE
        ),
        key=lambda channel: (
            channel.value
        ),
    )

    if not available:
        return (
            CommunicationChannel.NONE
        )

    weights: dict[
        CommunicationChannel,
        float,
    ] = {}

    for channel in available:

        if (
            channel
            == CommunicationChannel.WHATSAPP
        ):
            weights[channel] = 0.40

        elif (
            channel
            == CommunicationChannel.SMS
        ):
            weights[channel] = 0.30

        elif (
            channel
            == CommunicationChannel.EMAIL
        ):
            weights[channel] = 0.25

        elif (
            channel
            == CommunicationChannel.IN_APP
        ):
            weights[channel] = 0.20

        elif (
            channel
            == CommunicationChannel.VOICE
        ):
            weights[channel] = 0.10

        else:
            weights[channel] = 0.05

    return weighted_choice(
        rng,
        weights,
    )


def _merchant_numeric_suffix(
    merchant: SyntheticMerchant,
) -> int:
    """
    Extract 0001 from merchant_0001.

    Used only to create deterministic synthetic phone identifiers.
    """

    suffix = merchant.id.rsplit(
        "_",
        1,
    )[-1]

    if suffix.isdigit():
        return int(suffix)

    return 0


def _generate_customers_with_rng(
    merchant: SyntheticMerchant,
    count: int,
    rng: Random,
    reference_time: datetime,
) -> list[Customer]:
    """
    Internal generator that accepts an existing Random instance.

    Population generation uses one shared RNG stream so the full
    synthetic dataset is reproducible.
    """

    if count <= 0:
        raise ValueError(
            "count must be greater than zero."
        )

    customers: list[
        Customer
    ] = []

    merchant_number = (
        _merchant_numeric_suffix(
            merchant
        )
    )

    for index in range(
        1,
        count + 1,
    ):
        customer_id = (
            f"cust_{merchant.id}_"
            f"{index:05d}"
        )

        # ------------------------------------------------------
        # Historical payment behaviour
        # ------------------------------------------------------

        total_payments = rng.randint(
            1,
            40,
        )

        latent_success_rate = (
            rng.betavariate(
                merchant.customer_success_alpha,
                merchant.customer_success_beta,
            )
        )

        successful_payments = min(
            total_payments,
            max(
                0,
                round(
                    total_payments
                    * latent_success_rate
                ),
            ),
        )

        failed_payments = (
            total_payments
            - successful_payments
        )

        historical_success_rate = round(
            successful_payments
            / total_payments,
            4,
        )

        # ------------------------------------------------------
        # Historical recovery behaviour
        # ------------------------------------------------------

        previous_recovery_attempts = (
            rng.randint(
                0,
                min(
                    failed_payments,
                    6,
                ),
            )
        )

        latent_recovery_rate = (
            rng.betavariate(
                merchant.recovery_success_alpha,
                merchant.recovery_success_beta,
            )
        )

        previous_recovery_successes = sum(
            1
            for _ in range(
                previous_recovery_attempts
            )
            if (
                rng.random()
                < latent_recovery_rate
            )
        )

        # ------------------------------------------------------
        # Lifetime value
        # ------------------------------------------------------

        spend_multiplier = Decimal(
            str(
                rng.uniform(
                    0.55,
                    1.85,
                )
            )
        )

        lifetime_value = (
            merchant.average_order_value
            * Decimal(
                successful_payments
            )
            * spend_multiplier
        ).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )

        # ------------------------------------------------------
        # Preferences
        # ------------------------------------------------------

        preferred_method = (
            weighted_choice(
                rng,
                merchant.payment_method_weights,
            )
        )

        preferred_channel = (
            _choose_preferred_channel(
                rng,
                merchant,
            )
        )

        # ------------------------------------------------------
        # Synthetic contact data
        # ------------------------------------------------------

        email: str | None = None
        phone: str | None = None

        if (
            rng.random() < 0.90
            or preferred_channel
            == CommunicationChannel.EMAIL
        ):
            email = (
                f"{customer_id}"
                "@example.com"
            )

        phone_channels = {
            CommunicationChannel.SMS,
            CommunicationChannel.WHATSAPP,
            CommunicationChannel.VOICE,
        }

        if (
            rng.random() < 0.95
            or preferred_channel
            in phone_channels
        ):
            synthetic_number = (
                (
                    merchant_number
                    * 100_000
                )
                + index
            ) % 1_000_000_000

            # Deliberately uses an invalid Indian mobile prefix after
            # +91 so this cannot accidentally represent a real mobile
            # number while remaining phone-shaped for tests.
            phone = (
                "+910"
                f"{synthetic_number:09d}"
            )

        # ------------------------------------------------------
        # Contact and language preferences
        # ------------------------------------------------------

        do_not_contact = (
            rng.random()
            < merchant.do_not_contact_rate
        )

        language_preference = (
            "hi"
            if rng.random() < 0.15
            else "en"
        )

        # ------------------------------------------------------
        # Account age
        # ------------------------------------------------------

        account_age_days = rng.randint(
            30,
            900,
        )

        created_at = (
            reference_time
            - timedelta(
                days=account_age_days
            )
        )

        # ------------------------------------------------------
        # Domain Customer
        # ------------------------------------------------------

        customer = Customer(
            id=customer_id,

            merchant_id=merchant.id,

            email=email,

            phone=phone,

            created_at=created_at,

            lifetime_value=(
                lifetime_value
            ),

            successful_payments=(
                successful_payments
            ),

            failed_payments=(
                failed_payments
            ),

            historical_payment_success_rate=(
                historical_success_rate
            ),

            previous_recovery_attempts=(
                previous_recovery_attempts
            ),

            previous_recovery_successes=(
                previous_recovery_successes
            ),

            preferred_payment_method=(
                preferred_method
            ),

            preferred_channel=(
                preferred_channel
            ),

            language_preference=(
                language_preference
            ),

            do_not_contact=(
                do_not_contact
            ),

            timezone=(
                merchant.policy.timezone
            ),
        )

        customers.append(
            customer
        )

    return customers


def generate_customers_for_merchant(
    merchant: SyntheticMerchant,
    count: int,
    *,
    seed: int = 42,
    reference_time: datetime = (
        DEFAULT_REFERENCE_TIME
    ),
) -> list[Customer]:
    """
    Generate customers for one merchant.

    Useful for tests, demos, or controlled simulations.
    """

    if reference_time.tzinfo is None:
        raise ValueError(
            "reference_time must be timezone-aware."
        )

    return _generate_customers_with_rng(
        merchant=merchant,
        count=count,
        rng=Random(seed),
        reference_time=reference_time,
    )


def generate_synthetic_population(
    *,
    merchant_count: int = 8,
    customers_per_merchant: int = 50,
    seed: int = 42,
    reference_time: datetime = (
        DEFAULT_REFERENCE_TIME
    ),
) -> SyntheticPopulation:
    """
    Generate a complete deterministic merchant/customer population.

    Example:
        8 merchants * 50 customers = 400 customers.
    """

    if merchant_count <= 0:
        raise ValueError(
            "merchant_count must be greater than zero."
        )

    if customers_per_merchant <= 0:
        raise ValueError(
            "customers_per_merchant must be greater than zero."
        )

    if reference_time.tzinfo is None:
        raise ValueError(
            "reference_time must be timezone-aware."
        )

    merchants = generate_merchants(
        merchant_count,
        seed=seed,
    )

    # Separate deterministic RNG stream for customers.
    rng = Random(
        seed + 1
    )

    customers: list[
        Customer
    ] = []

    for merchant in merchants:

        merchant_customers = (
            _generate_customers_with_rng(
                merchant=merchant,
                count=customers_per_merchant,
                rng=rng,
                reference_time=reference_time,
            )
        )

        customers.extend(
            merchant_customers
        )

    return SyntheticPopulation(
        seed=seed,
        reference_time=reference_time,
        merchants=merchants,
        customers=customers,
    )