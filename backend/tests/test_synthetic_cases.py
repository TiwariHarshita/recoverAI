from datetime import (
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal

import pytest

from app.domain.enums import (
    CaseType,
    FailureClass,
    InvoiceStatus,
    PaymentStatus,
    RecoveryCaseStatus,
)

from app.services.candidate_actions import (
    generate_candidate_actions,
)
from app.services.diagnosis import (
    diagnose_case,
)

from simulator import (
    generate_case_for_customer,
    generate_recovery_cases,
    generate_synthetic_population,
)


REFERENCE_TIME = datetime(
    2026,
    8,
    23,
    0,
    0,
    tzinfo=timezone.utc,
)


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def population():

    return generate_synthetic_population(
        merchant_count=8,
        customers_per_merchant=25,
        seed=100,
        reference_time=REFERENCE_TIME,
    )


@pytest.fixture
def merchant_customer(
    population,
):

    merchant = (
        population.merchants[0]
    )

    customer = next(
        customer
        for customer
        in population.customers
        if (
            customer.merchant_id
            == merchant.id
        )
    )

    return (
        merchant,
        customer,
    )


# ============================================================
# REPRODUCIBILITY
# ============================================================


def test_single_case_generation_is_reproducible(
    merchant_customer,
):

    merchant, customer = (
        merchant_customer
    )

    first = generate_case_for_customer(
        merchant,
        customer,
        seed=99,
        case_index=1,
        reference_time=REFERENCE_TIME,
        case_type=(
            CaseType.PAYMENT_FAILURE
        ),
    )

    second = generate_case_for_customer(
        merchant,
        customer,
        seed=99,
        case_index=1,
        reference_time=REFERENCE_TIME,
        case_type=(
            CaseType.PAYMENT_FAILURE
        ),
    )

    assert first == second


def test_batch_generation_is_reproducible(
    population,
):

    first = generate_recovery_cases(
        population,
        250,
        seed=777,
    )

    second = generate_recovery_cases(
        population,
        250,
        seed=777,
    )

    assert first == second


def test_different_seed_changes_case_batch(
    population,
):

    first = generate_recovery_cases(
        population,
        100,
        seed=1,
    )

    second = generate_recovery_cases(
        population,
        100,
        seed=2,
    )

    assert first != second


def test_generated_case_ids_are_unique(
    population,
):

    batch = generate_recovery_cases(
        population,
        500,
        seed=404,
    )

    case_ids = [
        scenario.case.id
        for scenario
        in batch.scenarios
    ]

    assert (
        len(case_ids)
        == len(
            set(case_ids)
        )
    )


# ============================================================
# POPULATION RELATIONSHIPS
# ============================================================


def test_all_generated_cases_reference_real_population_entities(
    population,
):

    batch = generate_recovery_cases(
        population,
        300,
        seed=222,
    )

    merchant_ids = {
        merchant.id
        for merchant
        in population.merchants
    }

    customer_ids = {
        customer.id
        for customer
        in population.customers
    }

    for scenario in batch.scenarios:

        assert (
            scenario.case.merchant_id
            in merchant_ids
        )

        assert (
            scenario.case.customer_id
            in customer_ids
        )


# ============================================================
# NO DIAGNOSIS LEAKAGE
# ============================================================


def test_raw_cases_do_not_leak_normalized_failure_class(
    population,
):

    batch = generate_recovery_cases(
        population,
        400,
        seed=333,
    )

    assert all(
        (
            scenario.case.failure_class
            == FailureClass.UNKNOWN
        )
        for scenario
        in batch.scenarios
    )


# ============================================================
# DIAGNOSIS INTEGRATION
# ============================================================


def test_diagnosis_matches_simulator_ground_truth(
    population,
):

    batch = generate_recovery_cases(
        population,
        1000,
        seed=515,
    )

    for scenario in batch.scenarios:

        diagnosis = diagnose_case(
            scenario.case
        )

        assert (
            diagnosis.failure_class
            == scenario.expected_failure_class
        )


# ============================================================
# CANDIDATE GENERATOR INTEGRATION
# ============================================================


def test_generated_cases_feed_candidate_action_generator(
    population,
):

    batch = generate_recovery_cases(
        population,
        300,
        seed=616,
    )

    for scenario in batch.scenarios:

        diagnosis = diagnose_case(
            scenario.case
        )

        candidates = (
            generate_candidate_actions(
                scenario.case,
                diagnosis,
            )
        )

        assert (
            candidates.case_id
            == scenario.case.id
        )

        assert candidates.actions

        assert all(
            action.case_id
            == scenario.case.id
            for action
            in candidates.actions
        )


# ============================================================
# FORCE EACH CASE TYPE
# ============================================================


@pytest.mark.parametrize(
    "case_type",
    list(CaseType),
)
def test_can_force_each_supported_case_type(
    merchant_customer,
    case_type,
):

    merchant, customer = (
        merchant_customer
    )

    scenario = (
        generate_case_for_customer(
            merchant,
            customer,
            seed=11,
            case_index=1,
            reference_time=(
                REFERENCE_TIME
            ),
            case_type=case_type,
        )
    )

    assert (
        scenario.case.case_type
        == case_type
    )


# ============================================================
# PAYMENT FAILURE
# ============================================================


def test_payment_failure_builds_failed_payment(
    merchant_customer,
):

    merchant, customer = (
        merchant_customer
    )

    scenario = (
        generate_case_for_customer(
            merchant,
            customer,
            seed=121,
            reference_time=(
                REFERENCE_TIME
            ),
            case_type=(
                CaseType.PAYMENT_FAILURE
            ),
        )
    )

    assert (
        scenario.payment
        is not None
    )

    assert (
        scenario.subscription
        is None
    )

    assert (
        scenario.invoice
        is None
    )

    assert (
        scenario.payment.status
        == PaymentStatus.FAILED
    )

    assert (
        scenario.case.payment_id
        == scenario.payment.id
    )

    assert (
        scenario.case.payment_method
        == scenario.payment.method
    )

    assert (
        scenario.case.amount_at_risk
        == scenario.payment.amount
    )

    assert (
        scenario.case.error_reason
        == scenario.payment.error_reason
    )

    assert (
        scenario.case.attempt_count
        == scenario.payment.attempt_number
    )


# ============================================================
# SUBSCRIPTION FAILURE
# ============================================================


def test_subscription_failure_builds_subscription_and_payment(
    merchant_customer,
):

    merchant, customer = (
        merchant_customer
    )

    scenario = (
        generate_case_for_customer(
            merchant,
            customer,
            seed=131,
            reference_time=(
                REFERENCE_TIME
            ),
            case_type=(
                CaseType.SUBSCRIPTION_FAILURE
            ),
        )
    )

    assert (
        scenario.payment
        is not None
    )

    assert (
        scenario.subscription
        is not None
    )

    assert (
        scenario.invoice
        is None
    )

    assert (
        scenario.case.payment_id
        == scenario.payment.id
    )

    assert (
        scenario.case.subscription_id
        == scenario.subscription.id
    )

    assert (
        scenario.subscription.merchant_id
        == merchant.id
    )

    assert (
        scenario.subscription.customer_id
        == customer.id
    )


# ============================================================
# CHECKOUT ABANDONMENT
# ============================================================


def test_checkout_abandonment_has_checkout_id_without_payment_entity(
    merchant_customer,
):

    merchant, customer = (
        merchant_customer
    )

    scenario = (
        generate_case_for_customer(
            merchant,
            customer,
            seed=141,
            reference_time=(
                REFERENCE_TIME
            ),
            case_type=(
                CaseType.CHECKOUT_ABANDONMENT
            ),
        )
    )

    assert (
        scenario.case.checkout_id
        is not None
    )

    assert (
        scenario.case.payment_id
        is None
    )

    assert scenario.payment is None

    assert scenario.subscription is None

    assert scenario.invoice is None

    assert (
        scenario.expected_failure_class
        == FailureClass.CHECKOUT_ABANDONMENT
    )


# ============================================================
# OVERDUE INVOICE
# ============================================================


def test_overdue_invoice_links_amount_at_risk_to_outstanding_balance(
    merchant_customer,
):

    merchant, customer = (
        merchant_customer
    )

    scenario = (
        generate_case_for_customer(
            merchant,
            customer,
            seed=151,
            reference_time=(
                REFERENCE_TIME
            ),
            case_type=(
                CaseType.OVERDUE_INVOICE
            ),
        )
    )

    assert (
        scenario.invoice
        is not None
    )

    assert scenario.payment is None

    assert scenario.subscription is None

    invoice = scenario.invoice

    assert (
        scenario.case.invoice_id
        == invoice.id
    )

    assert (
        invoice.status
        in {
            InvoiceStatus.OVERDUE,
            InvoiceStatus.PARTIALLY_PAID,
        }
    )

    assert (
        scenario.case.amount_at_risk
        == (
            invoice.amount_due
            - invoice.amount_paid
        )
    )

    assert (
        invoice.days_overdue
        >= 1
    )

    assert (
        invoice.due_at
        < REFERENCE_TIME
    )


# ============================================================
# INITIAL CASE STATE
# ============================================================


def test_new_cases_start_with_no_recoverai_side_effects(
    population,
):

    batch = generate_recovery_cases(
        population,
        300,
        seed=717,
    )

    for scenario in batch.scenarios:

        case = scenario.case

        assert (
            case.status
            == RecoveryCaseStatus.OPEN
        )

        assert (
            case.recovery_retry_count
            == 0
        )

        assert (
            case.previous_contacts
            == 0
        )

        assert (
            case.recovered_amount
            == Decimal("0")
        )

        assert (
            case.selected_action_id
            is None
        )


# ============================================================
# RECOVERY-WINDOW VALIDITY
# ============================================================


def test_cases_are_generated_inside_merchant_recovery_window(
    population,
):

    merchants_by_id = {
        merchant.id: merchant
        for merchant
        in population.merchants
    }

    batch = generate_recovery_cases(
        population,
        500,
        seed=818,
    )

    for scenario in batch.scenarios:

        merchant = merchants_by_id[
            scenario.case.merchant_id
        ]

        age = (
            batch.reference_time
            - scenario.case.created_at
        )

        assert (
            age >= timedelta(0)
        )

        assert (
            age
            < timedelta(
                days=(
                    merchant.policy
                    .max_recovery_window_days
                )
            )
        )


# ============================================================
# AMOUNT VALIDITY
# ============================================================


def test_generated_amounts_are_positive(
    population,
):

    batch = generate_recovery_cases(
        population,
        300,
        seed=919,
    )

    assert all(
        (
            scenario.case.amount_at_risk
            > Decimal("0")
        )
        for scenario
        in batch.scenarios
    )


# ============================================================
# REFERENCE TIME
# ============================================================


def test_case_generation_uses_population_reference_time(
    population,
):

    batch = generate_recovery_cases(
        population,
        10,
        seed=1001,
    )

    assert (
        batch.reference_time
        == population.reference_time
    )


def test_explicit_reference_time_override_is_used(
    population,
):

    override = (
        REFERENCE_TIME
        + timedelta(
            days=3
        )
    )

    batch = generate_recovery_cases(
        population,
        10,
        seed=1002,
        reference_time=override,
    )

    assert (
        batch.reference_time
        == override
    )


# ============================================================
# INVALID RELATIONSHIPS
# ============================================================


def test_mismatched_customer_and_merchant_is_rejected(
    population,
):

    first_merchant = (
        population.merchants[0]
    )

    customer = next(
        customer
        for customer
        in population.customers
        if (
            customer.merchant_id
            != first_merchant.id
        )
    )

    with pytest.raises(
        ValueError,
        match="Customer does not belong",
    ):

        generate_case_for_customer(
            first_merchant,
            customer,
            seed=1,
            reference_time=(
                REFERENCE_TIME
            ),
        )


# ============================================================
# INVALID COUNTS
# ============================================================


@pytest.mark.parametrize(
    "count",
    [
        0,
        -1,
    ],
)
def test_invalid_case_count_is_rejected(
    population,
    count,
):

    with pytest.raises(
        ValueError,
        match=(
            "count must be greater than zero"
        ),
    ):

        generate_recovery_cases(
            population,
            count,
        )


# ============================================================
# INVALID TIME
# ============================================================


def test_naive_reference_time_is_rejected(
    merchant_customer,
):

    merchant, customer = (
        merchant_customer
    )

    naive = datetime(
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

        generate_case_for_customer(
            merchant,
            customer,
            reference_time=naive,
        )