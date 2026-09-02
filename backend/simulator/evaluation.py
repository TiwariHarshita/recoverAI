from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from app.domain.action_scoring import (
    RecoveryEconomicsConfig,
    RecoverySourceContext,
    SelectionOutcome,
)
from app.domain.actions import RecoveryAction
from app.domain.customer import Customer
from app.domain.enums import ActionStatus, PolicyDecision, RecoveryActionType
from app.domain.merchant import Merchant
from app.policy import MerchantPolicyEngine
from app.policy.models import PolicyContext, PolicyEvaluation
from app.services.action_selector import (
    DEFAULT_RECOVERY_ECONOMICS,
    RecoveryProbabilityModel,
    select_best_recovery_action,
)
from app.services.candidate_actions import generate_candidate_actions
from app.services.diagnosis import diagnose_case
from simulator.cases import SyntheticCaseBatch, SyntheticRecoveryScenario
from simulator.customers import SyntheticPopulation
from simulator.environment import RecoveryEnvironment
from simulator.merchants import SyntheticMerchant


MONEY_QUANT = Decimal("0.01")


class EvaluationPolicy(str, Enum):
    RECOVERAI_CATBOOST = "recoverai_catboost_erv"
    LOGISTIC_ERV = "logistic_erv"
    RULES_FIRST = "rules_first"


class ComparisonSignal(str, Enum):
    CLEAR_IMPROVEMENT = "clear_improvement"
    INCONCLUSIVE = "inconclusive"
    CLEAR_REGRESSION = "clear_regression"


class PolicyCaseEvaluation(BaseModel):
    """One policy evaluated on one fresh synthetic recovery case."""

    case_id: str
    merchant_id: str
    customer_id: str
    failure_class: str
    amount_at_risk: Decimal = Field(ge=0)

    policy: EvaluationPolicy
    selected_action_type: RecoveryActionType | None = None
    selection_outcome: SelectionOutcome
    execute_at: datetime | None = None

    predicted_recovery_probability: float | None = Field(default=None, ge=0, le=1)
    predicted_expected_recovery_value: Decimal | None = None

    # Simulator-only diagnostic. Never used as a production feature.
    latent_recovery_probability: float | None = Field(default=None, ge=0, le=1)

    rollout_count: int = Field(ge=0)
    recovery_success_rate: float = Field(ge=0, le=1)
    full_recovery_rate: float = Field(ge=0, le=1)
    mean_recovered_amount: Decimal = Field(ge=0)
    economic_cost: Decimal = Field(ge=0)
    mean_net_recovery_value: Decimal


class PolicyBatchSummary(BaseModel):
    policy: EvaluationPolicy
    cases: int = Field(ge=0)
    executed_cases: int = Field(ge=0)

    total_amount_at_risk: Decimal = Field(ge=0)
    expected_recovered_amount: Decimal = Field(ge=0)
    expected_net_recovery_value: Decimal

    recovered_amount_rate: float = Field(ge=0)
    mean_recovery_success_rate: float = Field(ge=0, le=1)
    mean_full_recovery_rate: float = Field(ge=0, le=1)
    mean_selected_latent_probability: float = Field(ge=0, le=1)

    execute_rate: float = Field(ge=0, le=1)
    schedule_rate: float = Field(ge=0, le=1)
    approval_rate: float = Field(ge=0, le=1)
    no_eligible_action_rate: float = Field(ge=0, le=1)

    action_counts: dict[str, int] = Field(default_factory=dict)


class PolicyComparison(BaseModel):
    primary_policy: EvaluationPolicy
    baseline_policy: EvaluationPolicy

    recovered_amount_uplift: Decimal
    relative_recovered_amount_uplift_pct: float
    net_recovery_value_uplift: Decimal
    relative_net_value_uplift_pct: float
    recovered_amount_rate_delta_pp: float
    recovery_success_rate_delta_pp: float
    action_disagreement_rate: float = Field(ge=0, le=1)

    mean_recovered_amount_uplift_per_case: Decimal
    recovered_amount_uplift_ci95_low_per_case: Decimal
    recovered_amount_uplift_ci95_high_per_case: Decimal

    mean_net_value_uplift_per_case: Decimal
    net_value_uplift_ci95_low_per_case: Decimal
    net_value_uplift_ci95_high_per_case: Decimal

    comparison_signal: ComparisonSignal

    primary_case_wins: int = Field(ge=0)
    baseline_case_wins: int = Field(ge=0)
    case_ties: int = Field(ge=0)


class BatchEvaluationReport(BaseModel):
    population_seed: int
    case_seed: int
    environment_seed: int
    reference_time: datetime

    requested_case_count: int = Field(gt=0)
    evaluated_case_count: int = Field(ge=0)
    skipped_case_count: int = Field(ge=0)
    rollouts_per_case: int = Field(gt=0)
    bootstrap_samples: int = Field(gt=0)
    bootstrap_seed: int

    # Recommendation quality is evaluated as if required human approval
    # were granted. Approval rates remain explicit in every summary.
    approval_assumed_for_evaluation: bool = True

    summaries: dict[str, PolicyBatchSummary]
    comparisons: dict[str, PolicyComparison]
    records: list[PolicyCaseEvaluation] = Field(default_factory=list)


class _RulesSelection(BaseModel):
    action: RecoveryAction
    initial_evaluation: PolicyEvaluation
    execution_evaluation: PolicyEvaluation
    execute_at: datetime
    outcome: SelectionOutcome


def _money(value: Decimal) -> Decimal:
    return value.quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)


def _merchant_profile(merchant: SyntheticMerchant) -> Merchant:
    return Merchant(
        merchant_id=merchant.id,
        archetype=merchant.archetype.value,
        average_order_value=merchant.average_order_value,
    )


def _source_context(scenario: SyntheticRecoveryScenario) -> RecoverySourceContext:
    return RecoverySourceContext(
        bank=scenario.payment.bank if scenario.payment else None,
        payment_attempt_number=(
            scenario.payment.attempt_number if scenario.payment else None
        ),
        subscription_retry_count=(
            scenario.subscription.retry_count if scenario.subscription else None
        ),
        mandate_active=(
            scenario.subscription.mandate_active if scenario.subscription else None
        ),
        invoice_days_overdue=(
            scenario.invoice.days_overdue if scenario.invoice else None
        ),
    )


def _policy_context(*, now: datetime, customer: Customer) -> PolicyContext:
    return PolicyContext(
        now=now,
        customer_do_not_contact=customer.do_not_contact,
        action_history=[],
    )


def _selection_outcome(
    initial: PolicyEvaluation,
    execution: PolicyEvaluation,
) -> SelectionOutcome:
    if execution.decision == PolicyDecision.REQUIRES_APPROVAL:
        return SelectionOutcome.REQUIRE_APPROVAL
    if initial.decision == PolicyDecision.DEFERRED:
        return SelectionOutcome.SCHEDULE
    return SelectionOutcome.EXECUTE


def select_rules_first_action(
    *,
    scenario: SyntheticRecoveryScenario,
    candidate_actions: list[RecoveryAction],
    merchant: SyntheticMerchant,
    customer: Customer,
    reference_time: datetime,
    policy_engine: MerchantPolicyEngine | None = None,
) -> _RulesSelection | None:
    """
    No-ML baseline: keep Candidate Action Generator order and choose the
    first action that can legally proceed.

    Deferred actions are rechecked at eligible_at so the baseline cannot
    bypass the same time-based policy guardrails enforced by Layer 11.
    """

    engine = policy_engine or MerchantPolicyEngine()
    context = _policy_context(now=reference_time, customer=customer)
    evaluations = engine.evaluate_candidates(
        recovery_case=scenario.case,
        actions=candidate_actions,
        policy=merchant.policy,
        context=context,
    )

    for action, initial in zip(candidate_actions, evaluations, strict=True):
        if initial.decision == PolicyDecision.BLOCKED:
            continue

        if initial.decision in {
            PolicyDecision.ALLOWED,
            PolicyDecision.REQUIRES_APPROVAL,
        }:
            return _RulesSelection(
                action=action,
                initial_evaluation=initial,
                execution_evaluation=initial,
                execute_at=reference_time,
                outcome=_selection_outcome(initial, initial),
            )

        if initial.decision != PolicyDecision.DEFERRED or initial.eligible_at is None:
            continue

        future_context = context.model_copy(
            update={"now": initial.eligible_at},
            deep=True,
        )
        execution = engine.evaluate(
            recovery_case=scenario.case,
            action=action,
            policy=merchant.policy,
            context=future_context,
        )

        if execution.decision not in {
            PolicyDecision.ALLOWED,
            PolicyDecision.REQUIRES_APPROVAL,
        }:
            continue

        return _RulesSelection(
            action=action,
            initial_evaluation=initial,
            execution_evaluation=execution,
            execute_at=initial.eligible_at,
            outcome=_selection_outcome(initial, execution),
        )

    return None


def _economic_cost(
    *,
    scenario: SyntheticRecoveryScenario,
    action: RecoveryAction,
    execute_at: datetime,
    reference_time: datetime,
    economics: RecoveryEconomicsConfig,
) -> Decimal:
    rule = economics.action_rules[action.action_type]
    delay_hours = max(
        Decimal("0"),
        Decimal(str((execute_at - reference_time).total_seconds() / 3600.0)),
    )
    amount = scenario.case.amount_at_risk

    return _money(
        rule.direct_cost
        + amount * rule.friction_rate
        + amount * economics.delay_penalty_rate_per_hour * delay_hours
    )


def _evaluate_selected_action(
    *,
    policy: EvaluationPolicy,
    scenario: SyntheticRecoveryScenario,
    merchant: SyntheticMerchant,
    customer: Customer,
    action: RecoveryAction | None,
    selection_outcome: SelectionOutcome,
    execute_at: datetime | None,
    predicted_probability: float | None,
    predicted_erv: Decimal | None,
    environment: RecoveryEnvironment,
    economics: RecoveryEconomicsConfig,
    reference_time: datetime,
    rollouts_per_case: int,
) -> PolicyCaseEvaluation:
    case = scenario.case

    if action is None or execute_at is None:
        return PolicyCaseEvaluation(
            case_id=case.id,
            merchant_id=merchant.id,
            customer_id=customer.id,
            failure_class=scenario.expected_failure_class.value,
            amount_at_risk=case.amount_at_risk,
            policy=policy,
            selected_action_type=None,
            selection_outcome=SelectionOutcome.NO_ELIGIBLE_ACTION,
            rollout_count=0,
            recovery_success_rate=0.0,
            full_recovery_rate=0.0,
            mean_recovered_amount=Decimal("0.00"),
            economic_cost=Decimal("0.00"),
            mean_net_recovery_value=Decimal("0.00"),
        )

    # The environment cannot execute REQUIRES_APPROVAL directly. For this
    # offline recommendation-quality benchmark only, treat it as approved.
    execution_action = action.model_copy(deep=True)
    if execution_action.status == ActionStatus.REQUIRES_APPROVAL:
        execution_action.status = ActionStatus.APPROVED

    latent_probability = environment.recovery_probability(
        scenario=scenario,
        merchant=merchant,
        customer=customer,
        action=execution_action,
        now=execute_at,
    )

    successes = 0
    full_recoveries = 0
    recovered_total = Decimal("0")

    for rollout_index in range(rollouts_per_case):
        result = environment.simulate_action(
            scenario=scenario,
            merchant=merchant,
            customer=customer,
            action=execution_action,
            now=execute_at,
            rollout_index=rollout_index,
        )
        successes += int(result.success)
        full_recoveries += int(result.fully_recovered)
        recovered_total += result.recovered_amount_this_action

    mean_recovered = _money(recovered_total / Decimal(rollouts_per_case))
    cost = _economic_cost(
        scenario=scenario,
        action=execution_action,
        execute_at=execute_at,
        reference_time=reference_time,
        economics=economics,
    )

    return PolicyCaseEvaluation(
        case_id=case.id,
        merchant_id=merchant.id,
        customer_id=customer.id,
        failure_class=scenario.expected_failure_class.value,
        amount_at_risk=case.amount_at_risk,
        policy=policy,
        selected_action_type=execution_action.action_type,
        selection_outcome=selection_outcome,
        execute_at=execute_at,
        predicted_recovery_probability=predicted_probability,
        predicted_expected_recovery_value=predicted_erv,
        latent_recovery_probability=latent_probability,
        rollout_count=rollouts_per_case,
        recovery_success_rate=successes / rollouts_per_case,
        full_recovery_rate=full_recoveries / rollouts_per_case,
        mean_recovered_amount=mean_recovered,
        economic_cost=cost,
        mean_net_recovery_value=_money(mean_recovered - cost),
    )


def _summary(
    policy: EvaluationPolicy,
    records: list[PolicyCaseEvaluation],
) -> PolicyBatchSummary:
    selected = [record for record in records if record.policy == policy]
    cases = len(selected)

    if cases == 0:
        return PolicyBatchSummary(
            policy=policy,
            cases=0,
            executed_cases=0,
            total_amount_at_risk=Decimal("0.00"),
            expected_recovered_amount=Decimal("0.00"),
            expected_net_recovery_value=Decimal("0.00"),
            recovered_amount_rate=0.0,
            mean_recovery_success_rate=0.0,
            mean_full_recovery_rate=0.0,
            mean_selected_latent_probability=0.0,
            execute_rate=0.0,
            schedule_rate=0.0,
            approval_rate=0.0,
            no_eligible_action_rate=0.0,
            action_counts={},
        )

    total_amount = sum((r.amount_at_risk for r in selected), Decimal("0"))
    recovered_amount = sum((r.mean_recovered_amount for r in selected), Decimal("0"))
    net_value = sum((r.mean_net_recovery_value for r in selected), Decimal("0"))
    executed = [r for r in selected if r.selected_action_type is not None]
    latent = [
        r.latent_recovery_probability
        for r in executed
        if r.latent_recovery_probability is not None
    ]
    outcomes = Counter(r.selection_outcome for r in selected)
    actions = Counter(
        r.selected_action_type.value
        for r in executed
        if r.selected_action_type is not None
    )

    return PolicyBatchSummary(
        policy=policy,
        cases=cases,
        executed_cases=len(executed),
        total_amount_at_risk=_money(total_amount),
        expected_recovered_amount=_money(recovered_amount),
        expected_net_recovery_value=_money(net_value),
        recovered_amount_rate=float(recovered_amount / total_amount) if total_amount > 0 else 0.0,
        mean_recovery_success_rate=sum(r.recovery_success_rate for r in selected) / cases,
        mean_full_recovery_rate=sum(r.full_recovery_rate for r in selected) / cases,
        mean_selected_latent_probability=sum(latent) / len(latent) if latent else 0.0,
        execute_rate=outcomes[SelectionOutcome.EXECUTE] / cases,
        schedule_rate=outcomes[SelectionOutcome.SCHEDULE] / cases,
        approval_rate=outcomes[SelectionOutcome.REQUIRE_APPROVAL] / cases,
        no_eligible_action_rate=outcomes[SelectionOutcome.NO_ELIGIBLE_ACTION] / cases,
        action_counts=dict(sorted(actions.items())),
    )


def _relative_pct(primary: Decimal, baseline: Decimal) -> float:
    if baseline == 0:
        return 0.0
    return float(((primary - baseline) / abs(baseline)) * Decimal("100"))


def _paired_bootstrap_mean_ci(
    differences: list[Decimal],
    *,
    samples: int,
    seed: int,
) -> tuple[Decimal, Decimal, Decimal]:
    """Deterministic paired bootstrap 95% CI for mean per-case uplift."""

    if samples <= 0:
        raise ValueError("bootstrap samples must be greater than zero.")
    if not differences:
        zero = Decimal("0.00")
        return zero, zero, zero

    values = np.asarray([float(value) for value in differences], dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)

    for index in range(samples):
        means[index] = float(
            rng.choice(values, size=len(values), replace=True).mean()
        )

    return (
        _money(Decimal(str(float(values.mean())))),
        _money(Decimal(str(float(np.quantile(means, 0.025))))),
        _money(Decimal(str(float(np.quantile(means, 0.975))))),
    )


def _comparison_signal(low: Decimal, high: Decimal) -> ComparisonSignal:
    if low > 0:
        return ComparisonSignal.CLEAR_IMPROVEMENT
    if high < 0:
        return ComparisonSignal.CLEAR_REGRESSION
    return ComparisonSignal.INCONCLUSIVE


def _comparison(
    *,
    primary: EvaluationPolicy,
    baseline: EvaluationPolicy,
    summaries: dict[str, PolicyBatchSummary],
    records: list[PolicyCaseEvaluation],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> PolicyComparison:
    primary_summary = summaries[primary.value]
    baseline_summary = summaries[baseline.value]

    by_case: dict[str, dict[EvaluationPolicy, PolicyCaseEvaluation]] = defaultdict(dict)
    for record in records:
        if record.policy in {primary, baseline}:
            by_case[record.case_id][record.policy] = record

    disagreements = 0
    primary_wins = 0
    baseline_wins = 0
    ties = 0
    recovered_differences: list[Decimal] = []
    net_differences: list[Decimal] = []

    for pair in by_case.values():
        if primary not in pair or baseline not in pair:
            continue

        a = pair[primary]
        b = pair[baseline]
        recovered_differences.append(a.mean_recovered_amount - b.mean_recovered_amount)
        net_differences.append(a.mean_net_recovery_value - b.mean_net_recovery_value)

        disagreements += int(a.selected_action_type != b.selected_action_type)
        if a.mean_net_recovery_value > b.mean_net_recovery_value:
            primary_wins += 1
        elif a.mean_net_recovery_value < b.mean_net_recovery_value:
            baseline_wins += 1
        else:
            ties += 1

    paired_cases = len(recovered_differences)
    recovered_mean, recovered_low, recovered_high = _paired_bootstrap_mean_ci(
        recovered_differences,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    net_mean, net_low, net_high = _paired_bootstrap_mean_ci(
        net_differences,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )

    recovered_uplift = _money(
        primary_summary.expected_recovered_amount
        - baseline_summary.expected_recovered_amount
    )
    net_uplift = _money(
        primary_summary.expected_net_recovery_value
        - baseline_summary.expected_net_recovery_value
    )

    return PolicyComparison(
        primary_policy=primary,
        baseline_policy=baseline,
        recovered_amount_uplift=recovered_uplift,
        relative_recovered_amount_uplift_pct=_relative_pct(
            primary_summary.expected_recovered_amount,
            baseline_summary.expected_recovered_amount,
        ),
        net_recovery_value_uplift=net_uplift,
        relative_net_value_uplift_pct=_relative_pct(
            primary_summary.expected_net_recovery_value,
            baseline_summary.expected_net_recovery_value,
        ),
        recovered_amount_rate_delta_pp=(
            primary_summary.recovered_amount_rate
            - baseline_summary.recovered_amount_rate
        ) * 100.0,
        recovery_success_rate_delta_pp=(
            primary_summary.mean_recovery_success_rate
            - baseline_summary.mean_recovery_success_rate
        ) * 100.0,
        action_disagreement_rate=(
            disagreements / paired_cases if paired_cases > 0 else 0.0
        ),
        mean_recovered_amount_uplift_per_case=recovered_mean,
        recovered_amount_uplift_ci95_low_per_case=recovered_low,
        recovered_amount_uplift_ci95_high_per_case=recovered_high,
        mean_net_value_uplift_per_case=net_mean,
        net_value_uplift_ci95_low_per_case=net_low,
        net_value_uplift_ci95_high_per_case=net_high,
        comparison_signal=_comparison_signal(net_low, net_high),
        primary_case_wins=primary_wins,
        baseline_case_wins=baseline_wins,
        case_ties=ties,
    )


def _model_policy_record(
    *,
    policy: EvaluationPolicy,
    model: RecoveryProbabilityModel,
    scenario: SyntheticRecoveryScenario,
    merchant: SyntheticMerchant,
    customer: Customer,
    diagnosis,
    candidate_actions: list[RecoveryAction],
    reference_time: datetime,
    environment: RecoveryEnvironment,
    economics: RecoveryEconomicsConfig,
    rollouts_per_case: int,
) -> PolicyCaseEvaluation:
    selection = select_best_recovery_action(
        recovery_case=scenario.case,
        customer=customer,
        diagnosis=diagnosis,
        candidate_actions=candidate_actions,
        merchant=_merchant_profile(merchant),
        merchant_policy=merchant.policy,
        policy_context=_policy_context(now=reference_time, customer=customer),
        source_context=_source_context(scenario),
        probability_model=model,
        economics=economics,
    )

    action = selection.selected_action
    execute_at = None
    if action is not None:
        execute_at = action.scheduled_for or reference_time

    return _evaluate_selected_action(
        policy=policy,
        scenario=scenario,
        merchant=merchant,
        customer=customer,
        action=action,
        selection_outcome=selection.outcome,
        execute_at=execute_at,
        predicted_probability=(
            selection.selected_score.predicted_recovery_probability
            if selection.selected_score else None
        ),
        predicted_erv=(
            selection.selected_score.expected_recovery_value
            if selection.selected_score else None
        ),
        environment=environment,
        economics=economics,
        reference_time=reference_time,
        rollouts_per_case=rollouts_per_case,
    )


def evaluate_policy_batch(
    *,
    population: SyntheticPopulation,
    case_batch: SyntheticCaseBatch,
    catboost_model: RecoveryProbabilityModel,
    logistic_model: RecoveryProbabilityModel,
    environment_seed: int = 2202,
    rollouts_per_case: int = 20,
    economics: RecoveryEconomicsConfig = DEFAULT_RECOVERY_ECONOMICS,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 2203,
) -> BatchEvaluationReport:
    """
    Evaluate three policies on the same fresh cases:

    1. RecoverAI CatBoost + ERV
    2. Logistic Regression + the same ERV selector
    3. Rules-first candidate ordering with no ML

    Hidden simulator probability is observed only after each policy has
    selected an action. It cannot influence selection.
    """

    if rollouts_per_case <= 0:
        raise ValueError("rollouts_per_case must be greater than zero.")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be greater than zero.")
    if case_batch.reference_time.tzinfo is None:
        raise ValueError("case_batch.reference_time must be timezone-aware.")

    merchants = {merchant.id: merchant for merchant in population.merchants}
    customers = {customer.id: customer for customer in population.customers}
    environment = RecoveryEnvironment(seed=environment_seed)
    records: list[PolicyCaseEvaluation] = []
    skipped = 0

    for scenario in case_batch.scenarios:
        case = scenario.case
        if (
            case.customer_id is None
            or case.merchant_id not in merchants
            or case.customer_id not in customers
        ):
            skipped += 1
            continue

        merchant = merchants[case.merchant_id]
        customer = customers[case.customer_id]
        diagnosis = diagnose_case(case)
        candidate_set = generate_candidate_actions(case, diagnosis)

        records.append(
            _model_policy_record(
                policy=EvaluationPolicy.RECOVERAI_CATBOOST,
                model=catboost_model,
                scenario=scenario,
                merchant=merchant,
                customer=customer,
                diagnosis=diagnosis,
                candidate_actions=candidate_set.actions,
                reference_time=case_batch.reference_time,
                environment=environment,
                economics=economics,
                rollouts_per_case=rollouts_per_case,
            )
        )
        records.append(
            _model_policy_record(
                policy=EvaluationPolicy.LOGISTIC_ERV,
                model=logistic_model,
                scenario=scenario,
                merchant=merchant,
                customer=customer,
                diagnosis=diagnosis,
                candidate_actions=candidate_set.actions,
                reference_time=case_batch.reference_time,
                environment=environment,
                economics=economics,
                rollouts_per_case=rollouts_per_case,
            )
        )

        rules = select_rules_first_action(
            scenario=scenario,
            candidate_actions=candidate_set.actions,
            merchant=merchant,
            customer=customer,
            reference_time=case_batch.reference_time,
        )
        records.append(
            _evaluate_selected_action(
                policy=EvaluationPolicy.RULES_FIRST,
                scenario=scenario,
                merchant=merchant,
                customer=customer,
                action=rules.action if rules else None,
                selection_outcome=(
                    rules.outcome if rules else SelectionOutcome.NO_ELIGIBLE_ACTION
                ),
                execute_at=rules.execute_at if rules else None,
                predicted_probability=None,
                predicted_erv=None,
                environment=environment,
                economics=economics,
                reference_time=case_batch.reference_time,
                rollouts_per_case=rollouts_per_case,
            )
        )

    summaries = {
        policy.value: _summary(policy, records)
        for policy in EvaluationPolicy
    }
    comparisons = {
        "recoverai_vs_logistic": _comparison(
            primary=EvaluationPolicy.RECOVERAI_CATBOOST,
            baseline=EvaluationPolicy.LOGISTIC_ERV,
            summaries=summaries,
            records=records,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
        "recoverai_vs_rules": _comparison(
            primary=EvaluationPolicy.RECOVERAI_CATBOOST,
            baseline=EvaluationPolicy.RULES_FIRST,
            summaries=summaries,
            records=records,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 10,
        ),
    }

    return BatchEvaluationReport(
        population_seed=population.seed,
        case_seed=case_batch.seed,
        environment_seed=environment_seed,
        reference_time=case_batch.reference_time,
        requested_case_count=len(case_batch.scenarios),
        evaluated_case_count=len(case_batch.scenarios) - skipped,
        skipped_case_count=skipped,
        rollouts_per_case=rollouts_per_case,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        approval_assumed_for_evaluation=True,
        summaries=summaries,
        comparisons=comparisons,
        records=records,
    )


def write_batch_evaluation(
    report: BatchEvaluationReport,
    *,
    summary_path: str | Path,
    cases_path: str | Path,
) -> tuple[Path, Path]:
    """Write compact JSON summary plus long-form per-policy case CSV."""

    summary_output = Path(summary_path)
    cases_output = Path(cases_path)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    cases_output.parent.mkdir(parents=True, exist_ok=True)

    with summary_output.open("w", encoding="utf-8") as handle:
        json.dump(
            report.model_dump(mode="json", exclude={"records"}),
            handle,
            indent=2,
        )
        handle.write("\n")

    fieldnames = list(PolicyCaseEvaluation.model_fields.keys())
    with cases_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in report.records:
            writer.writerow(record.model_dump(mode="json"))

    return summary_output, cases_output
