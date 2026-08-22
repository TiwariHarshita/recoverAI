from __future__ import annotations

import csv
import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, Field

from app.domain.actions import RecoveryAction
from app.domain.customer import Customer
from app.domain.enums import (
    CaseType,
    CommunicationChannel,
    DiagnosisCertainty,
    FailureClass,
    PaymentMethod,
    PolicyDecision,
    RecoveryActionType,
)
from app.policy import (
    MerchantPolicyEngine,
    PolicyContext,
)
from app.policy.models import PolicyEvaluation
from app.services.candidate_actions import (
    generate_candidate_actions,
)
from app.services.diagnosis import diagnose_case

from simulator.assumptions import MerchantArchetype
from simulator.cases import (
    SyntheticRecoveryScenario,
    generate_recovery_cases,
)
from simulator.customers import SyntheticPopulation
from simulator.environment import (
    RecoveryEnvironment,
    RecoveryOutcomeType,
    SimulationResult,
)
from simulator.merchants import SyntheticMerchant


class HistoricalRecoveryRecord(BaseModel):
    """
    One observable historical recovery attempt.

    This represents information that would be available to RecoverAI
    before an action executes, plus the outcome observed afterwards.

    Hidden simulator ground truth is intentionally excluded.
    """

    # ========================================================
    # AUDIT IDENTIFIERS
    # ========================================================

    # These should not become ML features later.
    history_id: str

    case_id: str

    merchant_id: str

    customer_id: str

    # ========================================================
    # MERCHANT CONTEXT
    # ========================================================

    merchant_archetype: MerchantArchetype

    merchant_average_order_value: float = Field(
        ge=0
    )

    amount_to_average_order_ratio: float = Field(
        ge=0
    )

    # ========================================================
    # CASE / PAYMENT CONTEXT
    # ========================================================

    case_type: CaseType

    amount_at_risk: float = Field(
        ge=0
    )

    currency: str

    payment_method: PaymentMethod | None = None

    bank: str | None = None

    # Raw payment/provider facts.
    error_code: str | None = None

    error_source: str | None = None

    error_step: str | None = None

    error_reason: str | None = None

    payment_attempt_number: int | None = Field(
        default=None,
        ge=1,
    )

    subscription_retry_count: int | None = Field(
        default=None,
        ge=0,
    )

    mandate_active: bool | None = None

    invoice_days_overdue: int | None = Field(
        default=None,
        ge=0,
    )

    attempt_count: int = Field(
        ge=0
    )

    recovery_retry_count: int = Field(
        ge=0
    )

    previous_contacts: int = Field(
        ge=0
    )

    case_age_hours: float = Field(
        ge=0
    )

    # ========================================================
    # DIAGNOSIS OUTPUT
    # ========================================================

    failure_class: FailureClass

    diagnosis_certainty: DiagnosisCertainty

    temporary_failure: bool

    retry_same_method_reasonable: bool

    requires_new_payment_method: bool

    customer_action_required: bool

    merchant_action_required: bool

    # ========================================================
    # CUSTOMER HISTORY
    # ========================================================

    customer_lifetime_value: float = Field(
        ge=0
    )

    historical_payment_success_rate: float = Field(
        ge=0,
        le=1,
    )

    successful_payments: int = Field(
        ge=0
    )

    failed_payments: int = Field(
        ge=0
    )

    previous_recovery_attempts: int = Field(
        ge=0
    )

    previous_recovery_successes: int = Field(
        ge=0
    )

    previous_recovery_success_rate: float = Field(
        ge=0,
        le=1,
    )

    preferred_payment_method: (
        PaymentMethod
        | None
    ) = None

    preferred_channel: (
        CommunicationChannel
        | None
    ) = None

    language_preference: str

    customer_do_not_contact: bool

    customer_tenure_days: float = Field(
        ge=0
    )

    # ========================================================
    # ACTION TAKEN
    # ========================================================

    action_type: RecoveryActionType

    channel: CommunicationChannel

    action_amount: float | None = Field(
        default=None,
        ge=0,
    )

    # Number of policy-executable candidate actions
    # available when the historical action was chosen.
    eligible_action_count: int = Field(
        ge=1
    )

    # The historical behaviour policy chooses uniformly
    # from eligible actions.
    behavior_selection_probability: float = Field(
        gt=0,
        le=1,
    )

    # ========================================================
    # POLICY OBSERVATION
    # ========================================================

    policy_decision_at_selection: PolicyDecision

    was_deferred: bool

    action_delay_hours: float = Field(
        ge=0
    )

    # ========================================================
    # SUPERVISED LABELS / OUTCOMES
    # ========================================================

    # Primary binary ML target.
    #
    # 1 = the action recovered some money.
    # This includes both full and partial recovery.
    recovered: int = Field(
        ge=0,
        le=1,
    )

    fully_recovered: int = Field(
        ge=0,
        le=1,
    )

    outcome: RecoveryOutcomeType

    recovered_amount: float = Field(
        ge=0
    )

    recovered_fraction: float = Field(
        ge=0,
        le=1,
    )

    recovery_delay_hours: float | None = Field(
        default=None,
        ge=0,
    )

    remaining_amount_after: float = Field(
        ge=0
    )


class HistoricalRecoveryDataset(BaseModel):
    """
    Reproducible historical training dataset.
    """

    population_seed: int

    case_seed: int

    environment_seed: int

    reference_time: datetime

    requested_case_count: int = Field(
        gt=0
    )

    record_count: int = Field(
        ge=0
    )

    skipped_case_count: int = Field(
        ge=0
    )

    records: list[
        HistoricalRecoveryRecord
    ] = Field(
        default_factory=list
    )


class _ExecutableCandidate(BaseModel):
    """
    Internal representation of a candidate that can actually
    be executed under merchant policy.
    """

    action: RecoveryAction

    initial_evaluation: PolicyEvaluation

    execute_at: datetime


# ============================================================
# DETERMINISTIC HISTORICAL BEHAVIOUR POLICY
# ============================================================


def _stable_index(
    *,
    seed: int,
    case_id: str,
    count: int,
) -> int:
    """
    Pick one candidate deterministically.

    We intentionally do not use RecoveryAction.id because action IDs
    currently use UUID4 and would make historical generation
    non-reproducible.
    """

    if count <= 0:
        raise ValueError(
            "count must be greater than zero."
        )

    raw = (
        f"{seed}|"
        f"{case_id}|"
        "historical_action_selection"
    ).encode(
        "utf-8"
    )

    digest = sha256(
        raw
    ).digest()

    integer = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return (
        integer
        % count
    )


def _history_id(
    *,
    case_id: str,
    action: RecoveryAction,
    case_seed: int,
    environment_seed: int,
) -> str:
    """
    Stable historical-row identifier.
    """

    raw = (
        f"{case_id}|"
        f"{action.action_type.value}|"
        f"{action.channel.value}|"
        f"{case_seed}|"
        f"{environment_seed}"
    ).encode(
        "utf-8"
    )

    token = sha256(
        raw
    ).hexdigest()[:20]

    return (
        f"hist_{token}"
    )


# ============================================================
# POLICY HELPERS
# ============================================================


def _policy_context(
    *,
    now: datetime,
    customer: Customer,
) -> PolicyContext:

    return PolicyContext(
        now=now,

        customer_do_not_contact=(
            customer.do_not_contact
        ),

        action_history=[],
    )


def _executable_candidates(
    *,
    scenario: SyntheticRecoveryScenario,
    merchant: SyntheticMerchant,
    customer: Customer,
    reference_time: datetime,
    actions: list[RecoveryAction],
    policy_engine: MerchantPolicyEngine,
) -> list[_ExecutableCandidate]:
    """
    Determine which candidate actions could actually have appeared
    in historical executed-action data.

    ALLOWED actions can execute immediately.

    DEFERRED actions are re-checked at eligible_at before being
    considered executable. This prevents quiet-hour deferral from
    accidentally bypassing another guardrail such as the recovery
    window.

    BLOCKED and REQUIRES_APPROVAL actions do not receive synthetic
    outcome labels because they were not automatically executed.
    """

    executable: list[
        _ExecutableCandidate
    ] = []

    initial_context = (
        _policy_context(
            now=reference_time,
            customer=customer,
        )
    )

    evaluations = (
        policy_engine.evaluate_candidates(
            recovery_case=(
                scenario.case
            ),

            actions=actions,

            policy=merchant.policy,

            context=initial_context,
        )
    )

    for (
        action,
        evaluation,
    ) in zip(
        actions,
        evaluations,
        strict=True,
    ):

        # ----------------------------------------------------
        # Immediately allowed
        # ----------------------------------------------------

        if (
            evaluation.decision
            == PolicyDecision.ALLOWED
        ):

            executable.append(
                _ExecutableCandidate(
                    action=action,

                    initial_evaluation=(
                        evaluation
                    ),

                    execute_at=(
                        reference_time
                    ),
                )
            )

            continue

        # ----------------------------------------------------
        # Not executable by automatic historical policy
        # ----------------------------------------------------

        if (
            evaluation.decision
            != PolicyDecision.DEFERRED
            or evaluation.eligible_at
            is None
        ):
            continue

        # ----------------------------------------------------
        # Re-evaluate deferred action at the time it becomes
        # eligible.
        # ----------------------------------------------------

        deferred_context = (
            _policy_context(
                now=(
                    evaluation
                    .eligible_at
                ),

                customer=customer,
            )
        )

        later_evaluation = (
            policy_engine.evaluate(
                recovery_case=(
                    scenario.case
                ),

                action=action,

                policy=(
                    merchant.policy
                ),

                context=(
                    deferred_context
                ),
            )
        )

        if (
            later_evaluation.decision
            == PolicyDecision.ALLOWED
        ):

            executable.append(
                _ExecutableCandidate(
                    action=action,

                    initial_evaluation=(
                        evaluation
                    ),

                    execute_at=(
                        evaluation
                        .eligible_at
                    ),
                )
            )

    return executable


# ============================================================
# FEATURE HELPERS
# ============================================================


def _previous_recovery_success_rate(
    customer: Customer,
) -> float:

    if (
        customer.previous_recovery_attempts
        == 0
    ):
        return 0.0

    return (
        customer
        .previous_recovery_successes
        / customer
        .previous_recovery_attempts
    )


def _safe_nonnegative_hours(
    start: datetime,
    end: datetime,
) -> float:

    return max(
        0.0,
        (
            end
            - start
        ).total_seconds()
        / 3600.0,
    )


# ============================================================
# HISTORICAL RECORD BUILDER
# ============================================================


def _build_record(
    *,
    scenario: SyntheticRecoveryScenario,
    merchant: SyntheticMerchant,
    customer: Customer,
    diagnosis,
    selected: _ExecutableCandidate,
    result: SimulationResult,
    eligible_action_count: int,
    case_seed: int,
    environment_seed: int,
    reference_time: datetime,
) -> HistoricalRecoveryRecord:

    case = (
        scenario.case
    )

    action = (
        selected.action
    )

    execute_at = (
        selected.execute_at
    )

    average_order_value = float(
        merchant.average_order_value
    )

    amount_at_risk = float(
        case.amount_at_risk
    )

    # --------------------------------------------------------
    # Relative transaction size
    # --------------------------------------------------------

    if (
        average_order_value
        > 0
    ):

        amount_ratio = (
            amount_at_risk
            / average_order_value
        )

    else:

        amount_ratio = 0.0

    # --------------------------------------------------------
    # Amount recovered as a fraction of the original risk
    # --------------------------------------------------------

    if (
        amount_at_risk
        > 0
    ):

        recovered_fraction = min(
            1.0,

            float(
                result
                .recovered_amount_this_action
            )
            / amount_at_risk,
        )

    else:

        recovered_fraction = 0.0

    return HistoricalRecoveryRecord(

        # ====================================================
        # AUDIT IDS
        # ====================================================

        history_id=(
            _history_id(
                case_id=case.id,
                action=action,
                case_seed=case_seed,
                environment_seed=(
                    environment_seed
                ),
            )
        ),

        case_id=case.id,

        merchant_id=merchant.id,

        customer_id=customer.id,

        # ====================================================
        # MERCHANT
        # ====================================================

        merchant_archetype=(
            merchant.archetype
        ),

        merchant_average_order_value=(
            average_order_value
        ),

        amount_to_average_order_ratio=(
            amount_ratio
        ),

        # ====================================================
        # CASE
        # ====================================================

        case_type=(
            case.case_type
        ),

        amount_at_risk=(
            amount_at_risk
        ),

        currency=(
            case.currency
        ),

        payment_method=(
            case.payment_method
        ),

        bank=(
            scenario.payment.bank
            if (
                scenario.payment
                is not None
            )
            else None
        ),

        error_code=(
            case.error_code
        ),

        error_source=(
            case.error_source
        ),

        error_step=(
            case.error_step
        ),

        error_reason=(
            case.error_reason
        ),

        payment_attempt_number=(
            scenario.payment
            .attempt_number
            if (
                scenario.payment
                is not None
            )
            else None
        ),

        subscription_retry_count=(
            scenario.subscription
            .retry_count
            if (
                scenario.subscription
                is not None
            )
            else None
        ),

        mandate_active=(
            scenario.subscription
            .mandate_active
            if (
                scenario.subscription
                is not None
            )
            else None
        ),

        invoice_days_overdue=(
            scenario.invoice
            .days_overdue
            if (
                scenario.invoice
                is not None
            )
            else None
        ),

        attempt_count=(
            case.attempt_count
        ),

        recovery_retry_count=(
            case.recovery_retry_count
        ),

        previous_contacts=(
            case.previous_contacts
        ),

        case_age_hours=(
            _safe_nonnegative_hours(
                case.created_at,
                execute_at,
            )
        ),

        # ====================================================
        # DIAGNOSIS
        # ====================================================

        failure_class=(
            diagnosis.failure_class
        ),

        diagnosis_certainty=(
            diagnosis.certainty
        ),

        temporary_failure=(
            diagnosis
            .temporary_failure
        ),

        retry_same_method_reasonable=(
            diagnosis
            .retry_same_method_reasonable
        ),

        requires_new_payment_method=(
            diagnosis
            .requires_new_payment_method
        ),

        customer_action_required=(
            diagnosis
            .customer_action_required
        ),

        merchant_action_required=(
            diagnosis
            .merchant_action_required
        ),

        # ====================================================
        # CUSTOMER
        # ====================================================

        customer_lifetime_value=float(
            customer.lifetime_value
        ),

        historical_payment_success_rate=(
            customer
            .historical_payment_success_rate
        ),

        successful_payments=(
            customer.successful_payments
        ),

        failed_payments=(
            customer.failed_payments
        ),

        previous_recovery_attempts=(
            customer
            .previous_recovery_attempts
        ),

        previous_recovery_successes=(
            customer
            .previous_recovery_successes
        ),

        previous_recovery_success_rate=(
            _previous_recovery_success_rate(
                customer
            )
        ),

        preferred_payment_method=(
            customer
            .preferred_payment_method
        ),

        preferred_channel=(
            customer.preferred_channel
        ),

        language_preference=(
            customer
            .language_preference
        ),

        customer_do_not_contact=(
            customer.do_not_contact
        ),

        customer_tenure_days=max(
            0.0,

            (
                execute_at
                - customer.created_at
            ).total_seconds()
            / 86400.0,
        ),

        # ====================================================
        # ACTION
        # ====================================================

        action_type=(
            action.action_type
        ),

        channel=(
            action.channel
        ),

        action_amount=(
            float(
                action.amount
            )
            if (
                action.amount
                is not None
            )
            else None
        ),

        eligible_action_count=(
            eligible_action_count
        ),

        behavior_selection_probability=(
            1.0
            / eligible_action_count
        ),

        # ====================================================
        # POLICY
        # ====================================================

        policy_decision_at_selection=(
            selected
            .initial_evaluation
            .decision
        ),

        was_deferred=(
            selected
            .initial_evaluation
            .decision
            == PolicyDecision.DEFERRED
        ),

        action_delay_hours=(
            _safe_nonnegative_hours(
                reference_time,
                execute_at,
            )
        ),

        # ====================================================
        # LABEL / OUTCOME
        # ====================================================

        recovered=int(
            result.success
        ),

        fully_recovered=int(
            result.fully_recovered
        ),

        outcome=(
            result.outcome
        ),

        recovered_amount=float(
            result
            .recovered_amount_this_action
        ),

        recovered_fraction=(
            recovered_fraction
        ),

        recovery_delay_hours=(
            result
            .recovery_delay_hours
        ),

        remaining_amount_after=float(
            result
            .remaining_amount_after
        ),
    )


# ============================================================
# PUBLIC DATASET GENERATOR
# ============================================================


def generate_historical_dataset(
    population: SyntheticPopulation,
    case_count: int,
    *,
    case_seed: int = 42,
    environment_seed: int = 43,
    reference_time: datetime | None = None,
) -> HistoricalRecoveryDataset:
    """
    Generate policy-aware synthetic recovery history.

    Pipeline:

        RecoveryCase
            -> Diagnosis Engine
            -> Candidate Action Generator
            -> Merchant Policy Engine
            -> historical behaviour policy
            -> RecoveryEnvironment
            -> observable training example

    One action is selected for each case that has at least one
    automatically executable candidate.

    Cases for which no candidate can legally execute are skipped.
    That is intentional because real supervised history contains no
    recovery outcome for an action that was never executed.
    """

    if (
        case_count
        <= 0
    ):
        raise ValueError(
            (
                "case_count must be "
                "greater than zero."
            )
        )

    resolved_reference_time = (
        reference_time
        if (
            reference_time
            is not None
        )
        else population.reference_time
    )

    # --------------------------------------------------------
    # Generate raw historical recovery cases.
    # --------------------------------------------------------

    case_batch = (
        generate_recovery_cases(
            population,
            case_count,
            seed=case_seed,
            reference_time=(
                resolved_reference_time
            ),
        )
    )

    merchants_by_id = {
        merchant.id: merchant
        for merchant
        in population.merchants
    }

    customers_by_id = {
        customer.id: customer
        for customer
        in population.customers
    }

    policy_engine = (
        MerchantPolicyEngine()
    )

    environment = (
        RecoveryEnvironment(
            seed=environment_seed
        )
    )

    records: list[
        HistoricalRecoveryRecord
    ] = []

    skipped_case_count = 0

    # ========================================================
    # GENERATE ONE OBSERVED ACTION PER CASE
    # ========================================================

    for scenario in (
        case_batch.scenarios
    ):

        case = (
            scenario.case
        )

        if (
            case.customer_id
            is None
        ):

            skipped_case_count += 1

            continue

        merchant = (
            merchants_by_id[
                case.merchant_id
            ]
        )

        customer = (
            customers_by_id[
                case.customer_id
            ]
        )

        # ----------------------------------------------------
        # Diagnosis
        # ----------------------------------------------------

        diagnosis = (
            diagnose_case(
                case
            )
        )

        # ----------------------------------------------------
        # Candidate generation
        # ----------------------------------------------------

        candidates = (
            generate_candidate_actions(
                case,
                diagnosis,
            )
        )

        # ----------------------------------------------------
        # Policy
        # ----------------------------------------------------

        executable = (
            _executable_candidates(
                scenario=scenario,

                merchant=merchant,

                customer=customer,

                reference_time=(
                    resolved_reference_time
                ),

                actions=(
                    candidates.actions
                ),

                policy_engine=(
                    policy_engine
                ),
            )
        )

        if not executable:

            skipped_case_count += 1

            continue

        # ----------------------------------------------------
        # Historical behaviour policy
        #
        # We deliberately use deterministic uniform exploration
        # rather than an ML model here.
        #
        # The ML model has not been trained yet.
        # ----------------------------------------------------

        selected = (
            executable[
                _stable_index(
                    seed=case_seed,

                    case_id=(
                        case.id
                    ),

                    count=len(
                        executable
                    ),
                )
            ]
        )

        # ----------------------------------------------------
        # Synthetic real-world outcome
        # ----------------------------------------------------

        result = (
            environment
            .simulate_action(
                scenario=(
                    scenario
                ),

                merchant=(
                    merchant
                ),

                customer=(
                    customer
                ),

                action=(
                    selected.action
                ),

                now=(
                    selected
                    .execute_at
                ),

                rollout_index=0,
            )
        )

        # ----------------------------------------------------
        # Observable historical row
        # ----------------------------------------------------

        record = (
            _build_record(
                scenario=scenario,

                merchant=merchant,

                customer=customer,

                diagnosis=diagnosis,

                selected=selected,

                result=result,

                eligible_action_count=len(
                    executable
                ),

                case_seed=(
                    case_seed
                ),

                environment_seed=(
                    environment_seed
                ),

                reference_time=(
                    resolved_reference_time
                ),
            )
        )

        records.append(
            record
        )

    return HistoricalRecoveryDataset(

        population_seed=(
            population.seed
        ),

        case_seed=(
            case_seed
        ),

        environment_seed=(
            environment_seed
        ),

        reference_time=(
            resolved_reference_time
        ),

        requested_case_count=(
            case_count
        ),

        record_count=len(
            records
        ),

        skipped_case_count=(
            skipped_case_count
        ),

        records=records,
    )


# ============================================================
# CSV OUTPUT
# ============================================================


def write_historical_dataset_csv(
    dataset: HistoricalRecoveryDataset,
    path: str | Path,
    *,
    write_metadata: bool = True,
) -> Path:
    """
    Write the historical dataset as a model-ready CSV.

    A metadata JSON sidecar is written by default so we can reproduce
    the dataset later and explicitly document leakage exclusions.
    """

    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        HistoricalRecoveryRecord
        .model_fields
        .keys()
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = (
            csv.DictWriter(
                handle,
                fieldnames=(
                    fieldnames
                ),
            )
        )

        writer.writeheader()

        for record in (
            dataset.records
        ):

            writer.writerow(
                record.model_dump(
                    mode="json"
                )
            )

    # --------------------------------------------------------
    # Reproducibility metadata
    # --------------------------------------------------------

    if write_metadata:

        metadata_path = (
            output_path
            .with_suffix(
                ".meta.json"
            )
        )

        metadata = {

            "population_seed": (
                dataset
                .population_seed
            ),

            "case_seed": (
                dataset.case_seed
            ),

            "environment_seed": (
                dataset
                .environment_seed
            ),

            "reference_time": (
                dataset
                .reference_time
                .isoformat()
            ),

            "requested_case_count": (
                dataset
                .requested_case_count
            ),

            "record_count": (
                dataset
                .record_count
            ),

            "skipped_case_count": (
                dataset
                .skipped_case_count
            ),

            "primary_target": (
                "recovered"
            ),

            "leakage_excluded": [
                "expected_failure_class",
                "latent_recovery_probability",
                "random_draw",
            ],
        }

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                metadata,
                handle,
                indent=2,
            )

            handle.write(
                "\n"
            )

    return output_path