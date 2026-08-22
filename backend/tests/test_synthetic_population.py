from datetime import (
    datetime,
)
from decimal import Decimal

import pytest

from app.domain.enums import (
    CommunicationChannel,
)

from simulator import (
    MerchantArchetype,
    generate_customers_for_merchant,
    generate_merchants,
    generate_synthetic_population,
)


# ============================================================
# MERCHANT GENERATION
# ============================================================


def test_merchant_generation_is_reproducible():
    first = generate_merchants(
        8,
        seed=123,
    )

    second = generate_merchants(
        8,
        seed=123,
    )

    assert first == second


def test_different_seed_changes_merchant_population():
    first = generate_merchants(
        8,
        seed=123,
    )

    second = generate_merchants(
        8,
        seed=456,
    )

    assert first != second


def test_generated_merchants_have_unique_ids():
    merchants = generate_merchants(
        20,
        seed=10,
    )

    ids = [
        merchant.id
        for merchant in merchants
    ]

    assert len(ids) == len(
        set(ids)
    )


def test_policy_merchant_id_matches_profile():
    merchants = generate_merchants(
        20,
        seed=10,
    )

    for merchant in merchants:

        assert (
            merchant.policy.merchant_id
            == merchant.id
        )


def test_generated_merchants_have_positive_order_value():
    merchants = generate_merchants(
        20,
        seed=10,
    )

    for merchant in merchants:

        assert (
            merchant.average_order_value
            > Decimal("0")
        )


def test_recoverai_retry_limit_remains_two():
    merchants = generate_merchants(
        20,
        seed=10,
    )

    for merchant in merchants:

        assert (
            merchant.policy.max_payment_retries
            == 2
        )


def test_merchant_weight_maps_are_valid():
    merchants = generate_merchants(
        20,
        seed=11,
    )

    for merchant in merchants:

        assert sum(
            merchant.payment_method_weights.values()
        ) == pytest.approx(
            1.0
        )

        assert sum(
            merchant.case_type_weights.values()
        ) == pytest.approx(
            1.0
        )

        assert all(
            weight >= 0
            for weight
            in merchant.payment_method_weights.values()
        )

        assert all(
            weight >= 0
            for weight
            in merchant.case_type_weights.values()
        )


def test_b2b_merchants_get_longer_recovery_window():
    merchants = generate_merchants(
        100,
        seed=99,
    )

    b2b_merchants = [
        merchant
        for merchant in merchants
        if (
            merchant.archetype
            == MerchantArchetype.B2B_INVOICING
        )
    ]

    assert b2b_merchants

    for merchant in b2b_merchants:

        assert (
            merchant.policy.max_recovery_window_days
            == 45
        )

        assert (
            merchant.policy.human_approval_threshold
            == Decimal("100000")
        )


# ============================================================
# CUSTOMER GENERATION
# ============================================================


def test_customer_generation_is_reproducible():
    merchant = generate_merchants(
        1,
        seed=5,
    )[0]

    first = (
        generate_customers_for_merchant(
            merchant,
            25,
            seed=12,
        )
    )

    second = (
        generate_customers_for_merchant(
            merchant,
            25,
            seed=12,
        )
    )

    assert first == second


def test_generated_customers_have_unique_ids():
    merchant = generate_merchants(
        1,
        seed=7,
    )[0]

    customers = (
        generate_customers_for_merchant(
            merchant,
            100,
            seed=8,
        )
    )

    assert len(customers) == 100

    assert len(
        {
            customer.id
            for customer in customers
        }
    ) == 100


def test_generated_customer_invariants():
    merchant = generate_merchants(
        1,
        seed=7,
    )[0]

    customers = (
        generate_customers_for_merchant(
            merchant,
            100,
            seed=8,
        )
    )

    for customer in customers:

        total_payments = (
            customer.successful_payments
            + customer.failed_payments
        )

        assert (
            customer.merchant_id
            == merchant.id
        )

        assert total_payments >= 1

        assert (
            customer.historical_payment_success_rate
            == pytest.approx(
                customer.successful_payments
                / total_payments,
                abs=0.0001,
            )
        )

        assert (
            0
            <= customer.previous_recovery_successes
            <= customer.previous_recovery_attempts
        )

        assert (
            customer.previous_recovery_attempts
            <= customer.failed_payments
        )

        assert (
            customer.lifetime_value
            >= Decimal("0")
        )

        assert (
            customer.preferred_payment_method
            in merchant.payment_method_weights
        )

        assert (
            customer.preferred_channel
            in merchant.policy.allowed_channels
        )

        assert (
            customer.preferred_channel
            != CommunicationChannel.NONE
        )

        assert (
            customer.timezone
            == merchant.policy.timezone
        )


def test_preferred_channel_has_required_contact_detail():
    merchant = generate_merchants(
        1,
        seed=20,
    )[0]

    customers = (
        generate_customers_for_merchant(
            merchant,
            200,
            seed=21,
        )
    )

    phone_channels = {
        CommunicationChannel.SMS,
        CommunicationChannel.WHATSAPP,
        CommunicationChannel.VOICE,
    }

    for customer in customers:

        if (
            customer.preferred_channel
            == CommunicationChannel.EMAIL
        ):
            assert (
                customer.email
                is not None
            )

        if (
            customer.preferred_channel
            in phone_channels
        ):
            assert (
                customer.phone
                is not None
            )


def test_customer_created_at_is_before_reference_time():
    population = (
        generate_synthetic_population(
            merchant_count=3,
            customers_per_merchant=20,
            seed=200,
        )
    )

    for customer in population.customers:

        assert (
            customer.created_at
            < population.reference_time
        )


# ============================================================
# COMPLETE POPULATION
# ============================================================


def test_population_has_expected_counts():
    population = (
        generate_synthetic_population(
            merchant_count=4,
            customers_per_merchant=30,
            seed=100,
        )
    )

    assert (
        len(population.merchants)
        == 4
    )

    assert (
        len(population.customers)
        == 120
    )


def test_every_customer_belongs_to_generated_merchant():
    population = (
        generate_synthetic_population(
            merchant_count=4,
            customers_per_merchant=30,
            seed=100,
        )
    )

    merchant_ids = {
        merchant.id
        for merchant
        in population.merchants
    }

    assert all(
        customer.merchant_id
        in merchant_ids
        for customer
        in population.customers
    )


def test_each_merchant_gets_expected_customer_count():
    population = (
        generate_synthetic_population(
            merchant_count=4,
            customers_per_merchant=30,
            seed=100,
        )
    )

    counts = {
        merchant.id: 0
        for merchant
        in population.merchants
    }

    for customer in population.customers:
        counts[
            customer.merchant_id
        ] += 1

    assert (
        set(counts.values())
        == {30}
    )


def test_population_generation_is_reproducible():
    first = (
        generate_synthetic_population(
            merchant_count=5,
            customers_per_merchant=10,
            seed=44,
        )
    )

    second = (
        generate_synthetic_population(
            merchant_count=5,
            customers_per_merchant=10,
            seed=44,
        )
    )

    assert first == second


# ============================================================
# INVALID INPUTS
# ============================================================


@pytest.mark.parametrize(
    "count",
    [
        0,
        -1,
    ],
)
def test_invalid_merchant_count_is_rejected(
    count,
):
    with pytest.raises(
        ValueError,
        match=(
            "count must be greater than zero"
        ),
    ):
        generate_merchants(
            count
        )


@pytest.mark.parametrize(
    "count",
    [
        0,
        -5,
    ],
)
def test_invalid_customer_count_is_rejected(
    count,
):
    merchant = generate_merchants(
        1
    )[0]

    with pytest.raises(
        ValueError,
        match=(
            "count must be greater than zero"
        ),
    ):
        generate_customers_for_merchant(
            merchant,
            count,
        )


def test_naive_reference_time_is_rejected():
    merchant = generate_merchants(
        1
    )[0]

    naive_time = datetime(
        2026,
        8,
        23,
        12,
        0,
    )

    with pytest.raises(
        ValueError,
        match=(
            "reference_time must be timezone-aware"
        ),
    ):
        generate_customers_for_merchant(
            merchant,
            10,
            reference_time=naive_time,
        )