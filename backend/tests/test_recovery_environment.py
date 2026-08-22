from datetime import (
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal

import pytest

from app.domain.actions import (
    RecoveryAction,
)
from app.domain.enums import (
    ActionStatus,
    CaseType,
    CommunicationChannel,
    PolicyDecision,
    RecoveryActionType,
    RecoveryCaseStatus,
)
from app.policy import (
    MerchantPolicyEngine,
    PolicyContext,
)

from app.services.candidate_actions import (
    generate_candidate_actions,
)
from app.services.diagnosis import (
    diagnose_case,
)

from simulator import (
    RecoveryEnvironment,
    RecoveryOutcomeType,
    generate_case_for_customer,
    generate_synthetic_population,
)


REFERENCE_TIME = datetime(
    2026,
    8,
    23,
    12,
    0,
    tzinfo=timezone.utc,
)


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def population():

    return generate_synthetic_population(
        merchant_count=6,
        customers_per_merchant=30,
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


@pytest.fixture
def scenario(
    merchant_customer,
):

    merchant, customer = (
        merchant_customer
    )

    return generate_case_for_customer(
        merchant,
        customer,
        seed=200,
        case_index=1,
        reference_time=REFERENCE_TIME,
        case_type=(
            CaseType.PAYMENT_FAILURE
        ),
    )


def make_action(
    scenario,
    action_type,
    channel=CommunicationChannel.NONE,
):
    return RecoveryAction(
        case_id=scenario.case.id,
        action_type=action_type,
        channel=channel,
        reason="environment test action",
    )


def find_successful_rollout(
    environment,
    scenario,
    merchant,
    customer,
    action,
):
    """
    Find a deterministic rollout that succeeds.

    Used so tests do not depend on one particular hash draw.
    """

    for rollout_index in range(
        1000
    ):
        result = (
            environment.simulate_action(
                scenario,
                merchant,
                customer,
                action,
                now=REFERENCE_TIME,
                rollout_index=rollout_index,
            )
        )

        if result.success:
            return result

    raise AssertionError(
        "Could not find successful rollout."
    )


def find_failed_rollout(
    environment,
    scenario,
    merchant,
    customer,
    action,
):
    """
    Find a deterministic rollout that produces no recovery.
    """

    for rollout_index in range(
        1000
    ):
        result = (
            environment.simulate_action(
                scenario,
                merchant,
                customer,
                action,
                now=REFERENCE_TIME,
                rollout_index=rollout_index,
            )
        )

        if not result.success:
            return result

    raise AssertionError(
        "Could not find failed rollout."
    )


# ============================================================
# DETERMINISM
# ============================================================


def test_probability_is_deterministic(
    merchant_customer,
    scenario,
):

    merchant, customer = (
        merchant_customer
    )

    action = make_action(
        scenario,
        RecoveryActionType.CREATE_PAYMENT_LINK,
        CommunicationChannel.SMS,
    )

    environment = RecoveryEnvironment(
        seed=42
    )

    first = environment.recovery_probability(
        scenario,
        merchant,
        customer,
        action,
        now=REFERENCE_TIME,
    )

    second = environment.recovery_probability(
        scenario,
        merchant,
        customer,
        action,
        now=REFERENCE_TIME,
    )

    assert first == second


def test_same_rollout_is_reproducible(
    merchant_customer,
    scenario,
):

    merchant, customer = (
        merchant_customer
    )

    action = make_action(
        scenario,
        RecoveryActionType.CREATE_PAYMENT_LINK,
        CommunicationChannel.SMS,
    )

    first = RecoveryEnvironment(
        seed=42
    ).simulate_action(
        scenario,
        merchant,
        customer,
        action,
        now=REFERENCE_TIME,
        rollout_index=10,
    )

    second = RecoveryEnvironment(
        seed=42
    ).simulate_action(
        scenario,
        merchant,
        customer,
        action,
        now=REFERENCE_TIME,
        rollout_index=10,
    )

    assert (
        first.random_draw
        == second.random_draw
    )

    assert (
        first.outcome
        == second.outcome
    )

    assert (
        first.recovered_amount_this_action
        == second.recovered_amount_this_action
    )

    assert (
        first.recovery_delay_hours
        == second.recovery_delay_hours
    )


def test_different_rollout_changes_random_draw(
    merchant_customer,
    scenario,
):

    merchant, customer = (
        merchant_customer
    )

    action = make_action(
        scenario,
        RecoveryActionType.CREATE_PAYMENT_LINK,
        CommunicationChannel.SMS,
    )

    environment = RecoveryEnvironment(
        seed=42
    )

    first = environment.simulate_action(
        scenario,
        merchant,
        customer,
        action,
        now=REFERENCE_TIME,
        rollout_index=1,
    )

    second = environment.simulate_action(
        scenario,
        merchant,
        customer,
        action,
        now=REFERENCE_TIME,
        rollout_index=2,
    )

    assert (
        first.random_draw
        != second.random_draw
    )


# ============================================================
# PROBABILITY VALIDITY
# ============================================================


def test_probabilities_are_bounded(
    population,
):

    environment = RecoveryEnvironment(
        seed=55
    )

    checked = 0

    for merchant in population.merchants:

        customer = next(
            customer
            for customer
            in population.customers
            if (
                customer.merchant_id
                == merchant.id
            )
        )

        scenario = (
            generate_case_for_customer(
                merchant,
                customer,
                seed=300 + checked,
                reference_time=(
                    REFERENCE_TIME
                ),
            )
        )

        diagnosis = diagnose_case(
            scenario.case
        )

        candidates = (
            generate_candidate_actions(
                scenario.case,
                diagnosis,
            )
        )

        for action in candidates.actions:

            probability = (
                environment
                .recovery_probability(
                    scenario,
                    merchant,
                    customer,
                    action,
                    now=REFERENCE_TIME,
                )
            )

            assert (
                0.0
                <= probability
                <= 1.0
            )

            checked += 1

    assert checked > 0


# ============================================================
# STOP
# ============================================================


def test_stop_never_recovers_money(
    merchant_customer,
    scenario,
):

    merchant, customer = (
        merchant_customer
    )

    action = make_action(
        scenario,
        RecoveryActionType.STOP,
    )

    result = RecoveryEnvironment(
        seed=1
    ).simulate_action(
        scenario,
        merchant,
        customer,
        action,
        now=REFERENCE_TIME,
    )

    assert (
        result.latent_recovery_probability
        == 0.0
    )

    assert (
        result.success
        is False
    )

    assert (
        result.outcome
        == RecoveryOutcomeType.STOPPED
    )

    assert (
        result.case_after.status
        == RecoveryCaseStatus.STOPPED
    )

    assert (
        result.case_after.closed_at
        == REFERENCE_TIME
    )


# ============================================================
# FULL RECOVERY
# ============================================================


def test_successful_normal_action_fully_recovers_case(
    merchant_customer,
    scenario,
):

    merchant, customer = (
        merchant_customer
    )

    action = make_action(
        scenario,
        RecoveryActionType.CREATE_PAYMENT_LINK,
        CommunicationChannel.SMS,
    )

    environment = RecoveryEnvironment(
        seed=100
    )

    result = find_successful_rollout(
        environment,
        scenario,
        merchant,
        customer,
        action,
    )

    assert (
        result.outcome
        == RecoveryOutcomeType.FULL_RECOVERY
    )

    assert result.success

    assert result.fully_recovered

    assert (
        result.recovered_amount_this_action
        == scenario.case.amount_at_risk
    )

    assert (
        result.remaining_amount_after
        == Decimal("0")
    )

    assert (
        result.case_after.status
        == RecoveryCaseStatus.RECOVERED
    )

    assert (
        result.case_after.recovered_amount
        == scenario.case.amount_at_risk
    )

    assert (
        result.recovered_at
        is not None
    )


# ============================================================
# FAILED RECOVERY
# ============================================================


def test_failed_action_does_not_recover_money(
    merchant_customer,
    scenario,
):

    merchant, customer = (
        merchant_customer
    )

    action = make_action(
        scenario,
        RecoveryActionType.CREATE_PAYMENT_LINK,
        CommunicationChannel.SMS,
    )

    environment = RecoveryEnvironment(
        seed=101
    )

    result = find_failed_rollout(
        environment,
        scenario,
        merchant,
        customer,
        action,
    )

    assert (
        result.success
        is False
    )

    assert (
        result.recovered_amount_this_action
        == Decimal("0")
    )

    assert (
        result.case_after.recovered_amount
        == scenario.case.recovered_amount
    )


# ============================================================
# RETRY COUNTER
# ============================================================


def test_retry_action_increments_recoverai_retry_count(
    merchant_customer,
    scenario,
):

    merchant, customer = (
        merchant_customer
    )

    original = (
        scenario.case.recovery_retry_count
    )

    action = make_action(
        scenario,
        RecoveryActionType.DELAYED_RETRY,
    )

    result = RecoveryEnvironment(
        seed=200
    ).simulate_action(
        scenario,
        merchant,
        customer,
        action,
        now=REFERENCE_TIME,
    )

    assert (
        result.case_after.recovery_retry_count
        == original + 1
    )


def test_customer_contact_does_not_increment_retry_count(
    merchant_customer,
    scenario,
):

    merchant, customer = (
        merchant_customer
    )

    original = (
        scenario.case.recovery_retry_count
    )

    action = make_action(
        scenario,
        RecoveryActionType.SEND_REMINDER,
        CommunicationChannel.SMS,
    )

    result = RecoveryEnvironment(
        seed=201
    ).simulate_action(
        scenario,
        merchant,
        customer,
        action,
        now=REFERENCE_TIME,
    )

    assert (
        result.case_after.recovery_retry_count
        == original
    )


# ============================================================
# CONTACT COUNTER
# ============================================================


def test_contact_action_increments_previous_contacts(
    merchant_customer,
    scenario,
):

    merchant, customer = (
        merchant_customer
    )

    original = (
        scenario.case.previous_contacts
    )

    action = make_action(
        scenario,
        RecoveryActionType.SEND_REMINDER,
        CommunicationChannel.SMS,
    )

    result = RecoveryEnvironment(
        seed=300
    ).simulate_action(
        scenario,
        merchant,
        customer,
        action,
        now=REFERENCE_TIME,
    )

    assert (
        result.case_after.previous_contacts
        == original + 1
    )


def test_non_contact_retry_does_not_increment_contacts(
    merchant_customer,
    scenario,
):

    merchant, customer = (
        merchant_customer
    )

    original = (
        scenario.case.previous_contacts
    )

    action = make_action(
        scenario,
        RecoveryActionType.DELAYED_RETRY,
    )

    result = RecoveryEnvironment(
        seed=301
    ).simulate_action(
        scenario,
        merchant,
        customer,
        action,
        now=REFERENCE_TIME,
    )

    assert (
        result.case_after.previous_contacts
        == original
    )


# ============================================================
# PARTIAL PAYMENT
# ============================================================


def test_partial_payment_can_produce_partial_recovery(
    population,
):

    merchant = next(
        merchant
        for merchant
        in population.merchants
        if (
            merchant.policy
            .allow_partial_payments
        )
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

    scenario = (
        generate_case_for_customer(
            merchant,
            customer,
            seed=400,
            case_index=1,
            reference_time=(
                REFERENCE_TIME
            ),
            case_type=(
                CaseType.OVERDUE_INVOICE
            ),
        )
    )

    action = make_action(
        scenario,
        RecoveryActionType.OFFER_PARTIAL_PAYMENT,
        CommunicationChannel.EMAIL,
    )

    result = find_successful_rollout(
        RecoveryEnvironment(
            seed=401
        ),
        scenario,
        merchant,
        customer,
        action,
    )

    assert (
        result.outcome
        == RecoveryOutcomeType.PARTIAL_RECOVERY
    )

    assert (
        Decimal("0")
        < result.recovered_amount_this_action
        < scenario.case.amount_at_risk
    )

    assert (
        result.remaining_amount_after
        > Decimal("0")
    )

    assert (
        result.case_after.status
        == RecoveryCaseStatus.WAITING_CUSTOMER
    )


# ============================================================
# ACTION STATE
# ============================================================


def test_simulated_action_becomes_executed(
    merchant_customer,
    scenario,
):

    merchant, customer = (
        merchant_customer
    )

    action = make_action(
        scenario,
        RecoveryActionType.DELAYED_RETRY,
    )

    result = RecoveryEnvironment(
        seed=500
    ).simulate_action(
        scenario,
        merchant,
        customer,
        action,
        now=REFERENCE_TIME,
    )

    assert (
        result.action_after.status
        == ActionStatus.EXECUTED
    )

    assert (
        result.action_after.executed_at
        == REFERENCE_TIME
    )


# ============================================================
# INPUT IMMUTABILITY
# ============================================================


def test_environment_does_not_mutate_input_case_or_action(
    merchant_customer,
    scenario,
):

    merchant, customer = (
        merchant_customer
    )

    action = make_action(
        scenario,
        RecoveryActionType.DELAYED_RETRY,
    )

    original_case = (
        scenario.case.model_copy(
            deep=True
        )
    )

    original_action = (
        action.model_copy(
            deep=True
        )
    )

    RecoveryEnvironment(
        seed=600
    ).simulate_action(
        scenario,
        merchant,
        customer,
        action,
        now=REFERENCE_TIME,
    )

    assert (
        scenario.case
        == original_case
    )

    assert (
        action
        == original_action
    )


# ============================================================
# CUSTOMER HISTORY SIGNAL
# ============================================================


def test_stronger_customer_history_increases_probability(
    merchant_customer,
    scenario,
):

    merchant, customer = (
        merchant_customer
    )

    weak_customer = (
        customer.model_copy(
            update={
                "historical_payment_success_rate": 0.30,
                "previous_recovery_attempts": 4,
                "previous_recovery_successes": 0,
            }
        )
    )

    strong_customer = (
        customer.model_copy(
            update={
                "historical_payment_success_rate": 0.98,
                "previous_recovery_attempts": 4,
                "previous_recovery_successes": 4,
            }
        )
    )

    action = make_action(
        scenario,
        RecoveryActionType.CREATE_PAYMENT_LINK,
        CommunicationChannel.SMS,
    )

    environment = RecoveryEnvironment(
        seed=700
    )

    weak_probability = (
        environment.recovery_probability(
            scenario,
            merchant,
            weak_customer,
            action,
            now=REFERENCE_TIME,
        )
    )

    strong_probability = (
        environment.recovery_probability(
            scenario,
            merchant,
            strong_customer,
            action,
            now=REFERENCE_TIME,
        )
    )

    assert (
        strong_probability
        > weak_probability
    )


# ============================================================
# CHANNEL SIGNAL
# ============================================================


def test_matching_customer_channel_improves_probability(
    merchant_customer,
    scenario,
):

    merchant, customer = (
        merchant_customer
    )

    customer = customer.model_copy(
        update={
            "preferred_channel": (
                CommunicationChannel.SMS
            )
        }
    )

    sms_action = make_action(
        scenario,
        RecoveryActionType.SEND_REMINDER,
        CommunicationChannel.SMS,
    )

    email_action = make_action(
        scenario,
        RecoveryActionType.SEND_REMINDER,
        CommunicationChannel.EMAIL,
    )

    environment = RecoveryEnvironment(
        seed=800
    )

    sms_probability = (
        environment.recovery_probability(
            scenario,
            merchant,
            customer,
            sms_action,
            now=REFERENCE_TIME,
        )
    )

    email_probability = (
        environment.recovery_probability(
            scenario,
            merchant,
            customer,
            email_action,
            now=REFERENCE_TIME,
        )
    )

    assert (
        sms_probability
        > email_probability
    )


# ============================================================
# PIPELINE INTEGRATION
# ============================================================


def test_case_to_diagnosis_to_candidates_to_policy_to_environment(
    merchant_customer,
):

    merchant, customer = (
        merchant_customer
    )

    # Avoid do-not-contact for this normal happy-path integration test.
    customer = customer.model_copy(
        update={
            "do_not_contact": False
        }
    )

    scenario = (
        generate_case_for_customer(
            merchant,
            customer,
            seed=900,
            case_index=1,
            reference_time=(
                REFERENCE_TIME
            ),
            case_type=(
                CaseType.PAYMENT_FAILURE
            ),
        )
    )

    diagnosis = diagnose_case(
        scenario.case
    )

    candidates = (
        generate_candidate_actions(
            scenario.case,
            diagnosis,
        )
    )

    policy_engine = (
        MerchantPolicyEngine()
    )

    policy_context = (
        PolicyContext(
            now=REFERENCE_TIME,

            customer_do_not_contact=(
                customer.do_not_contact
            ),

            action_history=[],
        )
    )

    allowed_actions = []

    for action in candidates.actions:

        evaluation = (
            policy_engine.evaluate(
                scenario.case,
                action,
                merchant.policy,
                policy_context,
            )
        )

        if (
            evaluation.decision
            == PolicyDecision.ALLOWED
        ):
            allowed_actions.append(
                action
            )

    assert allowed_actions

    selected_action = (
        allowed_actions[0]
    )

    result = (
        RecoveryEnvironment(
            seed=901
        ).simulate_action(
            scenario,
            merchant,
            customer,
            selected_action,
            now=REFERENCE_TIME,
        )
    )

    assert (
        result.case_id
        == scenario.case.id
    )

    assert (
        result.action_id
        == selected_action.id
    )


# ============================================================
# VALIDATION
# ============================================================


def test_wrong_action_case_is_rejected(
    merchant_customer,
    scenario,
):

    merchant, customer = (
        merchant_customer
    )

    action = RecoveryAction(
        case_id="rc_wrong",
        action_type=(
            RecoveryActionType.WAIT
        ),
    )

    with pytest.raises(
        ValueError,
        match="does not belong",
    ):
        RecoveryEnvironment().simulate_action(
            scenario,
            merchant,
            customer,
            action,
            now=REFERENCE_TIME,
        )


def test_naive_now_is_rejected(
    merchant_customer,
    scenario,
):

    merchant, customer = (
        merchant_customer
    )

    action = make_action(
        scenario,
        RecoveryActionType.WAIT,
    )

    naive_now = datetime(
        2026,
        8,
        23,
        12,
        0,
    )

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        RecoveryEnvironment().simulate_action(
            scenario,
            merchant,
            customer,
            action,
            now=naive_now,
        )


def test_execution_before_case_creation_is_rejected(
    merchant_customer,
    scenario,
):

    merchant, customer = (
        merchant_customer
    )

    action = make_action(
        scenario,
        RecoveryActionType.WAIT,
    )

    before_case = (
        scenario.case.created_at
        - timedelta(
            seconds=1
        )
    )

    with pytest.raises(
        ValueError,
        match="before the RecoveryCase",
    ):
        RecoveryEnvironment().simulate_action(
            scenario,
            merchant,
            customer,
            action,
            now=before_case,
        )


def test_already_executed_action_is_rejected(
    merchant_customer,
    scenario,
):

    merchant, customer = (
        merchant_customer
    )

    action = RecoveryAction(
        case_id=scenario.case.id,

        action_type=(
            RecoveryActionType.WAIT
        ),

        status=(
            ActionStatus.EXECUTED
        ),

        executed_at=(
            REFERENCE_TIME
        ),
    )

    with pytest.raises(
        ValueError,
        match="not executable",
    ):
        RecoveryEnvironment().simulate_action(
            scenario,
            merchant,
            customer,
            action,
            now=REFERENCE_TIME,
        )


def test_terminal_case_is_rejected(
    merchant_customer,
    scenario,
):

    merchant, customer = (
        merchant_customer
    )

    terminal_scenario = (
        scenario.model_copy(
            update={
                "case": (
                    scenario.case.model_copy(
                        update={
                            "status": (
                                RecoveryCaseStatus.RECOVERED
                            )
                        }
                    )
                )
            }
        )
    )

    action = make_action(
        terminal_scenario,
        RecoveryActionType.WAIT,
    )

    with pytest.raises(
        ValueError,
        match="terminal RecoveryCase",
    ):
        RecoveryEnvironment().simulate_action(
            terminal_scenario,
            merchant,
            customer,
            action,
            now=REFERENCE_TIME,
        )


def test_negative_rollout_index_is_rejected(
    merchant_customer,
    scenario,
):

    merchant, customer = (
        merchant_customer
    )

    action = make_action(
        scenario,
        RecoveryActionType.WAIT,
    )

    with pytest.raises(
        ValueError,
        match="rollout_index",
    ):
        RecoveryEnvironment().simulate_action(
            scenario,
            merchant,
            customer,
            action,
            now=REFERENCE_TIME,
            rollout_index=-1,
        )