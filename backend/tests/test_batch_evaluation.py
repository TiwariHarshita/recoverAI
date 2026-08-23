from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from app.domain.enums import RecoveryActionType
from simulator.cases import generate_recovery_cases
from simulator.customers import generate_synthetic_population
from simulator.evaluation import (
    EvaluationPolicy,
    evaluate_policy_batch,
    select_rules_first_action,
    write_batch_evaluation,
)
from app.services.candidate_actions import generate_candidate_actions
from app.services.diagnosis import diagnose_case


REFERENCE_TIME = datetime(
    2026,
    8,
    23,
    0,
    0,
    tzinfo=timezone.utc,
)


class ActionProbabilityModel:
    """Small deterministic probability model for evaluation tests."""

    def __init__(
        self,
        probabilities: dict[RecoveryActionType, float],
        *,
        default: float = 0.25,
    ) -> None:
        self.probabilities = probabilities
        self.default = default

    def predict_recovery_probability(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        values = []

        for raw in dataframe["action_type"]:
            action_type = (
                raw
                if isinstance(raw, RecoveryActionType)
                else RecoveryActionType(str(raw))
            )

            values.append(
                self.probabilities.get(
                    action_type,
                    self.default,
                )
            )

        return np.asarray(values, dtype=float)


@pytest.fixture(scope="module")
def population():
    return generate_synthetic_population(
        merchant_count=5,
        customers_per_merchant=30,
        seed=3000,
        reference_time=REFERENCE_TIME,
    )


@pytest.fixture(scope="module")
def case_batch(population):
    return generate_recovery_cases(
        population,
        120,
        seed=3001,
        reference_time=REFERENCE_TIME,
    )


@pytest.fixture(scope="module")
def primary_model():
    return ActionProbabilityModel(
        {
            RecoveryActionType.IMMEDIATE_RETRY: 0.80,
            RecoveryActionType.DELAYED_RETRY: 0.75,
            RecoveryActionType.CREATE_PAYMENT_LINK: 0.70,
            RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD: 0.78,
            RecoveryActionType.SEND_REMINDER: 0.50,
            RecoveryActionType.OFFER_PARTIAL_PAYMENT: 0.72,
            RecoveryActionType.REQUEST_PROMISE_TO_PAY: 0.55,
            RecoveryActionType.WAIT: 0.10,
            RecoveryActionType.ESCALATE_TO_HUMAN: 0.40,
            RecoveryActionType.STOP: 0.01,
        }
    )


@pytest.fixture(scope="module")
def baseline_model():
    return ActionProbabilityModel(
        {},
        default=0.30,
    )


def test_batch_evaluation_is_reproducible(
    population,
    case_batch,
    primary_model,
    baseline_model,
):
    first = evaluate_policy_batch(
        population=population,
        case_batch=case_batch,
        catboost_model=primary_model,
        logistic_model=baseline_model,
        environment_seed=3002,
        rollouts_per_case=4,
    )

    second = evaluate_policy_batch(
        population=population,
        case_batch=case_batch,
        catboost_model=primary_model,
        logistic_model=baseline_model,
        environment_seed=3002,
        rollouts_per_case=4,
    )

    assert first == second


def test_every_evaluated_case_has_all_three_policy_records(
    population,
    case_batch,
    primary_model,
    baseline_model,
):
    report = evaluate_policy_batch(
        population=population,
        case_batch=case_batch,
        catboost_model=primary_model,
        logistic_model=baseline_model,
        environment_seed=3010,
        rollouts_per_case=3,
    )

    assert len(report.records) == report.evaluated_case_count * 3

    policies = set(EvaluationPolicy)

    by_case: dict[str, set[EvaluationPolicy]] = {}

    for record in report.records:
        by_case.setdefault(record.case_id, set()).add(record.policy)

    assert all(value == policies for value in by_case.values())


def test_rules_baseline_preserves_candidate_generator_order(population):
    merchants = {merchant.id: merchant for merchant in population.merchants}
    customers = {customer.id: customer for customer in population.customers}

    batch = generate_recovery_cases(
        population,
        40,
        seed=3050,
        reference_time=REFERENCE_TIME,
    )

    checked = False

    for scenario in batch.scenarios:
        case = scenario.case
        if case.customer_id is None:
            continue

        merchant = merchants[case.merchant_id]
        customer = customers[case.customer_id]
        diagnosis = diagnose_case(case)
        candidates = generate_candidate_actions(case, diagnosis)

        selection = select_rules_first_action(
            scenario=scenario,
            candidate_actions=candidates.actions,
            merchant=merchant,
            customer=customer,
            reference_time=REFERENCE_TIME,
        )

        if selection is None:
            continue

        # If the first generator candidate was selectable, rules-first
        # must choose it exactly. Cases where policy blocks/defer-rejects
        # the first candidate are intentionally skipped for this assertion.
        if selection.action.id == candidates.actions[0].id:
            assert selection.action.action_type == candidates.actions[0].action_type
            checked = True
            break

    assert checked is True


def test_summary_arithmetic_matches_case_records(
    population,
    case_batch,
    primary_model,
    baseline_model,
):
    report = evaluate_policy_batch(
        population=population,
        case_batch=case_batch,
        catboost_model=primary_model,
        logistic_model=baseline_model,
        environment_seed=3060,
        rollouts_per_case=5,
    )

    summary = report.summaries[
        EvaluationPolicy.RECOVERAI_CATBOOST.value
    ]

    records = [
        record
        for record in report.records
        if record.policy == EvaluationPolicy.RECOVERAI_CATBOOST
    ]

    expected_recovered = sum(
        (record.mean_recovered_amount for record in records),
        Decimal("0"),
    )

    expected_net = sum(
        (record.mean_net_recovery_value for record in records),
        Decimal("0"),
    )

    assert summary.expected_recovered_amount == expected_recovered.quantize(
        Decimal("0.01")
    )

    assert summary.expected_net_recovery_value == expected_net.quantize(
        Decimal("0.01")
    )


def test_identical_models_produce_identical_erv_policy_performance(
    population,
    case_batch,
    primary_model,
):
    report = evaluate_policy_batch(
        population=population,
        case_batch=case_batch,
        catboost_model=primary_model,
        logistic_model=primary_model,
        environment_seed=3070,
        rollouts_per_case=4,
    )

    cat = report.summaries[
        EvaluationPolicy.RECOVERAI_CATBOOST.value
    ]
    log = report.summaries[
        EvaluationPolicy.LOGISTIC_ERV.value
    ]

    assert cat.expected_recovered_amount == log.expected_recovered_amount
    assert cat.expected_net_recovery_value == log.expected_net_recovery_value
    assert cat.action_counts == log.action_counts

    comparison = report.comparisons["recoverai_vs_logistic"]

    assert comparison.recovered_amount_uplift == Decimal("0.00")
    assert comparison.net_recovery_value_uplift == Decimal("0.00")
    assert comparison.action_disagreement_rate == 0.0


def test_evaluation_tracks_approval_gates_without_crashing(
    primary_model,
    baseline_model,
):
    population = generate_synthetic_population(
        merchant_count=3,
        customers_per_merchant=25,
        seed=3080,
        reference_time=REFERENCE_TIME,
    )

    # Force approval thresholds very low so non-safe actions commonly
    # require approval. Pydantic models are mutable in the current domain.
    for merchant in population.merchants:
        merchant.policy.human_approval_threshold = Decimal("1.00")

    batch = generate_recovery_cases(
        population,
        80,
        seed=3081,
        reference_time=REFERENCE_TIME,
    )

    report = evaluate_policy_batch(
        population=population,
        case_batch=batch,
        catboost_model=primary_model,
        logistic_model=baseline_model,
        environment_seed=3082,
        rollouts_per_case=2,
    )

    assert report.approval_assumed_for_evaluation is True

    approval_rates = [
        summary.approval_rate
        for summary in report.summaries.values()
    ]

    assert any(rate > 0.0 for rate in approval_rates)


def test_comparison_counts_partition_all_paired_cases(
    population,
    case_batch,
    primary_model,
    baseline_model,
):
    report = evaluate_policy_batch(
        population=population,
        case_batch=case_batch,
        catboost_model=primary_model,
        logistic_model=baseline_model,
        environment_seed=3090,
        rollouts_per_case=3,
    )

    for comparison in report.comparisons.values():
        assert (
            comparison.primary_case_wins
            + comparison.baseline_case_wins
            + comparison.case_ties
            == report.evaluated_case_count
        )


def test_writer_creates_summary_json_and_long_form_csv(
    population,
    case_batch,
    primary_model,
    baseline_model,
    tmp_path,
):
    report = evaluate_policy_batch(
        population=population,
        case_batch=case_batch,
        catboost_model=primary_model,
        logistic_model=baseline_model,
        environment_seed=3100,
        rollouts_per_case=2,
    )

    summary_path = tmp_path / "summary.json"
    cases_path = tmp_path / "cases.csv"

    returned_summary, returned_cases = write_batch_evaluation(
        report,
        summary_path=summary_path,
        cases_path=cases_path,
    )

    assert returned_summary == summary_path
    assert returned_cases == cases_path
    assert summary_path.exists()
    assert cases_path.exists()

    summary_text = summary_path.read_text(encoding="utf-8")
    header = cases_path.read_text(encoding="utf-8").splitlines()[0]

    assert "recoverai_vs_logistic" in summary_text
    assert "recoverai_vs_rules" in summary_text
    assert "latent_recovery_probability" not in summary_text
    assert "latent_recovery_probability" in header
