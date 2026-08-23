from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import (
    Decimal,
    ROUND_HALF_UP,
)
from typing import Protocol

import numpy as np
import pandas as pd

from app.domain.action_scoring import (
    ActionEconomicRule,
    ActionScore,
    ActionSelectionResult,
    ExcludedAction,
    MerchantScoringProfile,
    RecoveryEconomicsConfig,
    RecoverySourceContext,
    SelectionOutcome,
)
from app.domain.actions import RecoveryAction
from app.domain.customer import Customer
from app.domain.diagnosis import DiagnosisResult
from app.domain.enums import (
    ActionStatus,
    PolicyDecision,
    RecoveryActionType,
)
from app.domain.policies import MerchantPolicy
from app.domain.recovery_case import RecoveryCase
from app.ml.inference_features import (
    build_action_feature_frame,
    build_action_feature_row,
)
from app.policy import MerchantPolicyEngine
from app.policy.models import (
    PolicyContext,
    PolicyEvaluation,
)


MONEY_QUANT = Decimal(
    "0.01"
)


class RecoveryProbabilityModel(
    Protocol
):
    """
    Minimal model contract needed by action selection.

    Both LogisticRecoveryBaseline and CatBoostRecoveryModel
    already satisfy this interface.
    """

    def predict_recovery_probability(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        ...


# ============================================================
# DEFAULT PRODUCT ECONOMICS
# ============================================================

"""
These values exist to make economic ranking meaningful in the
hackathon/demo environment.

They are NOT claims about actual Razorpay pricing.

Later these can become merchant-configurable database settings.
"""


DEFAULT_RECOVERY_ECONOMICS = (
    RecoveryEconomicsConfig(

        action_rules={

            RecoveryActionType.IMMEDIATE_RETRY: (
                ActionEconomicRule(
                    direct_cost=Decimal(
                        "0.25"
                    ),
                    friction_rate=Decimal(
                        "0.0005"
                    ),
                )
            ),

            RecoveryActionType.DELAYED_RETRY: (
                ActionEconomicRule(
                    direct_cost=Decimal(
                        "0.25"
                    ),
                    friction_rate=Decimal(
                        "0.0002"
                    ),
                )
            ),

            RecoveryActionType.CREATE_PAYMENT_LINK: (
                ActionEconomicRule(
                    direct_cost=Decimal(
                        "1.00"
                    ),
                    friction_rate=Decimal(
                        "0.0010"
                    ),
                )
            ),

            RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD: (
                ActionEconomicRule(
                    direct_cost=Decimal(
                        "1.50"
                    ),
                    friction_rate=Decimal(
                        "0.0015"
                    ),
                )
            ),

            RecoveryActionType.SEND_REMINDER: (
                ActionEconomicRule(
                    direct_cost=Decimal(
                        "0.50"
                    ),
                    friction_rate=Decimal(
                        "0.0008"
                    ),
                )
            ),

            RecoveryActionType.OFFER_PARTIAL_PAYMENT: (
                ActionEconomicRule(
                    direct_cost=Decimal(
                        "1.00"
                    ),
                    friction_rate=Decimal(
                        "0.0020"
                    ),

                    # Binary ML target means:
                    # "Did we recover anything?"
                    #
                    # A successful partial-payment action should not
                    # be valued as if it always recovers 100%.
                    expected_recovery_fraction=Decimal(
                        "0.50"
                    ),
                )
            ),

            RecoveryActionType.REQUEST_PROMISE_TO_PAY: (
                ActionEconomicRule(
                    direct_cost=Decimal(
                        "1.00"
                    ),
                    friction_rate=Decimal(
                        "0.0010"
                    ),
                )
            ),

            RecoveryActionType.WAIT: (
                ActionEconomicRule(
                    direct_cost=Decimal(
                        "0"
                    ),
                    friction_rate=Decimal(
                        "0"
                    ),
                )
            ),

            RecoveryActionType.ESCALATE_TO_HUMAN: (
                ActionEconomicRule(
                    direct_cost=Decimal(
                        "150.00"
                    ),
                    friction_rate=Decimal(
                        "0.0025"
                    ),
                )
            ),

            RecoveryActionType.STOP: (
                ActionEconomicRule(
                    direct_cost=Decimal(
                        "0"
                    ),
                    friction_rate=Decimal(
                        "0"
                    ),
                    expected_recovery_fraction=Decimal(
                        "0"
                    ),
                )
            ),
        },

        delay_penalty_rate_per_hour=Decimal(
            "0.00001"
        ),
    )
)


# ============================================================
# INTERNAL TYPES
# ============================================================


@dataclass(
    frozen=True
)
class _ScoreableCandidate:

    action: RecoveryAction

    initial_evaluation: (
        PolicyEvaluation
    )

    execution_evaluation: (
        PolicyEvaluation
    )

    execute_at: datetime

    original_index: int


@dataclass(
    frozen=True
)
class _UnrankedScore:

    action: RecoveryAction

    initial_evaluation: (
        PolicyEvaluation
    )

    execution_evaluation: (
        PolicyEvaluation
    )

    execute_at: datetime

    original_index: int

    probability: float

    recoverable_amount: Decimal

    expected_gross_recovery: Decimal

    direct_action_cost: Decimal

    friction_cost: Decimal

    delay_cost: Decimal

    expected_recovery_value: Decimal


# ============================================================
# MONEY HELPERS
# ============================================================


def _money(
    value: Decimal,
) -> Decimal:

    return value.quantize(
        MONEY_QUANT,
        rounding=ROUND_HALF_UP,
    )


# ============================================================
# POLICY RESOLUTION
# ============================================================


def _resolve_candidate_policy(
    *,
    recovery_case: RecoveryCase,
    action: RecoveryAction,
    evaluation: PolicyEvaluation,
    policy: MerchantPolicy,
    context: PolicyContext,
    policy_engine: MerchantPolicyEngine,
    original_index: int,
) -> tuple[
    _ScoreableCandidate | None,
    PolicyEvaluation,
]:
    """
    Resolve whether an action may proceed to ML scoring.

    BLOCKED actions stop here.

    DEFERRED actions must pass another deterministic policy check
    at eligible_at. This prevents quiet-hours deferral from
    bypassing later recovery-window or approval guardrails.
    """

    if (
        evaluation.decision
        == PolicyDecision.BLOCKED
    ):
        return (
            None,
            evaluation,
        )

    # Immediate ALLOWED / REQUIRES_APPROVAL action.
    if (
        evaluation.decision
        != PolicyDecision.DEFERRED
    ):
        return (
            _ScoreableCandidate(
                action=action,

                initial_evaluation=(
                    evaluation
                ),

                execution_evaluation=(
                    evaluation
                ),

                execute_at=(
                    context.now
                ),

                original_index=(
                    original_index
                ),
            ),

            evaluation,
        )

    # --------------------------------------------------------
    # Deferred action
    # --------------------------------------------------------

    if (
        evaluation.eligible_at
        is None
    ):
        raise ValueError(
            (
                "Deferred policy evaluation "
                "must include eligible_at."
            )
        )

    future_context = (
        context.model_copy(

            update={
                "now": (
                    evaluation.eligible_at
                )
            },

            deep=True,
        )
    )

    execution_evaluation = (
        policy_engine.evaluate(

            recovery_case=(
                recovery_case
            ),

            action=action,

            policy=policy,

            context=(
                future_context
            ),
        )
    )

    # Fail closed if future execution is no longer safe.
    if (
        execution_evaluation.decision
        in {
            PolicyDecision.BLOCKED,
            PolicyDecision.DEFERRED,
        }
    ):
        return (
            None,
            execution_evaluation,
        )

    return (
        _ScoreableCandidate(

            action=action,

            initial_evaluation=(
                evaluation
            ),

            execution_evaluation=(
                execution_evaluation
            ),

            execute_at=(
                evaluation.eligible_at
            ),

            original_index=(
                original_index
            ),
        ),

        execution_evaluation,
    )


# ============================================================
# INPUT VALIDATION
# ============================================================


def _validate_relationships(
    *,
    recovery_case: RecoveryCase,
    customer: Customer,
    merchant: MerchantScoringProfile,
    policy: MerchantPolicy,
    context: PolicyContext,
) -> None:

    if (
        recovery_case.customer_id
        is None
    ):
        raise ValueError(
            (
                "RecoveryCase.customer_id is "
                "required for action scoring."
            )
        )

    if (
        recovery_case.customer_id
        != customer.id
    ):
        raise ValueError(
            (
                "Customer does not match "
                "RecoveryCase.customer_id."
            )
        )

    if (
        customer.merchant_id
        != recovery_case.merchant_id
    ):
        raise ValueError(
            (
                "Customer does not belong to "
                "the RecoveryCase merchant."
            )
        )

    if (
        merchant.merchant_id
        != recovery_case.merchant_id
    ):
        raise ValueError(
            (
                "Merchant scoring profile does "
                "not belong to the RecoveryCase merchant."
            )
        )

    if (
        policy.merchant_id
        != recovery_case.merchant_id
    ):
        raise ValueError(
            (
                "Merchant policy does not belong "
                "to the RecoveryCase merchant."
            )
        )

    if (
        context.now.tzinfo
        is None
    ):
        raise ValueError(
            (
                "PolicyContext.now must be "
                "timezone-aware."
            )
        )


def _validate_probabilities(
    probabilities: np.ndarray,
    *,
    expected_count: int,
) -> np.ndarray:

    values = np.asarray(
        probabilities,
        dtype=float,
    ).reshape(
        -1
    )

    if (
        len(values)
        != expected_count
    ):
        raise ValueError(
            (
                "Recovery model returned an unexpected "
                "number of probabilities: "
                f"expected {expected_count}, "
                f"got {len(values)}."
            )
        )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise ValueError(
            (
                "Recovery model returned "
                "non-finite probabilities."
            )
        )

    if (
        np.any(
            values < 0.0
        )
        or np.any(
            values > 1.0
        )
    ):
        raise ValueError(
            (
                "Recovery model probabilities "
                "must be between 0 and 1."
            )
        )

    return values


# ============================================================
# ECONOMIC SCORING
# ============================================================


def _economic_score(
    *,
    recovery_case: RecoveryCase,
    candidate: _ScoreableCandidate,
    probability: float,
    economics: RecoveryEconomicsConfig,
    selection_time: datetime,
) -> _UnrankedScore:

    rule = (
        economics.action_rules[
            candidate.action.action_type
        ]
    )

    amount_at_risk = (
        recovery_case.amount_at_risk
    )

    # --------------------------------------------------------
    # Expected collectible amount
    # --------------------------------------------------------

    recoverable_amount = _money(
        amount_at_risk
        * rule.expected_recovery_fraction
    )

    expected_gross_recovery = _money(
        Decimal(
            str(
                probability
            )
        )
        * recoverable_amount
    )

    # --------------------------------------------------------
    # Direct action cost
    # --------------------------------------------------------

    direct_action_cost = _money(
        rule.direct_cost
    )

    # --------------------------------------------------------
    # Customer friction / business cost
    # --------------------------------------------------------

    friction_cost = _money(
        amount_at_risk
        * rule.friction_rate
    )

    # --------------------------------------------------------
    # Delay / opportunity cost
    # --------------------------------------------------------

    delay_hours = max(
        Decimal(
            "0"
        ),

        Decimal(
            str(
                (
                    candidate.execute_at
                    - selection_time
                ).total_seconds()
                / 3600.0
            )
        ),
    )

    delay_cost = _money(
        amount_at_risk
        * economics.delay_penalty_rate_per_hour
        * delay_hours
    )

    # --------------------------------------------------------
    # Expected Recovery Value
    # --------------------------------------------------------

    expected_recovery_value = _money(
        expected_gross_recovery
        - direct_action_cost
        - friction_cost
        - delay_cost
    )

    return _UnrankedScore(

        action=(
            candidate.action
        ),

        initial_evaluation=(
            candidate.initial_evaluation
        ),

        execution_evaluation=(
            candidate.execution_evaluation
        ),

        execute_at=(
            candidate.execute_at
        ),

        original_index=(
            candidate.original_index
        ),

        probability=(
            probability
        ),

        recoverable_amount=(
            recoverable_amount
        ),

        expected_gross_recovery=(
            expected_gross_recovery
        ),

        direct_action_cost=(
            direct_action_cost
        ),

        friction_cost=(
            friction_cost
        ),

        delay_cost=(
            delay_cost
        ),

        expected_recovery_value=(
            expected_recovery_value
        ),
    )


# ============================================================
# RANKING
# ============================================================


def _rank_scores(
    scores: list[
        _UnrankedScore
    ],
) -> list[
    _UnrankedScore
]:
    """
    Stable deterministic ranking.

    Priority:

    1. Higher expected recovery value
    2. Higher recovery probability
    3. Lower total economic cost
    4. Original Candidate Generator order
    """

    def key(
        item: _UnrankedScore,
    ):

        total_cost = (
            item.direct_action_cost
            + item.friction_cost
            + item.delay_cost
        )

        return (
            -item.expected_recovery_value,

            -Decimal(
                str(
                    item.probability
                )
            ),

            total_cost,

            item.original_index,
        )

    return sorted(
        scores,
        key=key,
    )


# ============================================================
# PUBLIC SCORE CONVERSION
# ============================================================


def _public_score(
    item: _UnrankedScore,
    *,
    rank: int,
) -> ActionScore:

    total_cost = (
        item.direct_action_cost
        + item.friction_cost
        + item.delay_cost
    )

    explanation = (
        f"P(recovery)={item.probability:.4f}; "
        f"expected gross recovery="
        f"{item.expected_gross_recovery}; "
        f"economic costs="
        f"{_money(total_cost)}; "
        f"expected recovery value="
        f"{item.expected_recovery_value}."
    )

    return ActionScore(

        rank=rank,

        action_id=(
            item.action.id
        ),

        action_type=(
            item.action.action_type
        ),

        policy_decision_at_selection=(
            item.initial_evaluation.decision
        ),

        policy_reason_at_selection=(
            item.initial_evaluation.reason.value
        ),

        policy_explanation_at_selection=(
            item.initial_evaluation.explanation
        ),

        execution_policy_decision=(
            item.execution_evaluation.decision
        ),

        execution_policy_reason=(
            item.execution_evaluation.reason.value
        ),

        execution_policy_explanation=(
            item.execution_evaluation.explanation
        ),

        eligible_at=(
            item.initial_evaluation.eligible_at
        ),

        predicted_recovery_probability=(
            item.probability
        ),

        recoverable_amount=(
            item.recoverable_amount
        ),

        expected_gross_recovery=(
            item.expected_gross_recovery
        ),

        direct_action_cost=(
            item.direct_action_cost
        ),

        friction_cost=(
            item.friction_cost
        ),

        delay_cost=(
            item.delay_cost
        ),

        expected_recovery_value=(
            item.expected_recovery_value
        ),

        explanation=(
            explanation
        ),
    )


# ============================================================
# SELECTED ACTION ENRICHMENT
# ============================================================


def _selected_action(
    item: _UnrankedScore,
) -> RecoveryAction:

    action = (
        item.action.model_copy(
            deep=True
        )
    )

    action.predicted_recovery_probability = (
        item.probability
    )

    action.expected_recovery_value = (
        item.expected_recovery_value
    )

    # --------------------------------------------------------
    # Approval gate
    # --------------------------------------------------------

    if (
        item.execution_evaluation.decision
        == PolicyDecision.REQUIRES_APPROVAL
    ):

        action.status = (
            ActionStatus.REQUIRES_APPROVAL
        )

        if (
            item.initial_evaluation.decision
            == PolicyDecision.DEFERRED
        ):
            action.scheduled_for = (
                item.execute_at
            )

    # --------------------------------------------------------
    # Future scheduling
    # --------------------------------------------------------

    elif (
        item.initial_evaluation.decision
        == PolicyDecision.DEFERRED
    ):

        action.status = (
            ActionStatus.SCHEDULED
        )

        action.scheduled_for = (
            item.execute_at
        )

    return action


def _selection_outcome(
    item: _UnrankedScore,
) -> SelectionOutcome:

    if (
        item.execution_evaluation.decision
        == PolicyDecision.REQUIRES_APPROVAL
    ):
        return (
            SelectionOutcome.REQUIRE_APPROVAL
        )

    if (
        item.initial_evaluation.decision
        == PolicyDecision.DEFERRED
    ):
        return (
            SelectionOutcome.SCHEDULE
        )

    return (
        SelectionOutcome.EXECUTE
    )


# ============================================================
# PUBLIC ACTION SELECTOR
# ============================================================


def select_best_recovery_action(
    *,
    recovery_case: RecoveryCase,
    customer: Customer,
    diagnosis: DiagnosisResult,
    candidate_actions: list[
        RecoveryAction
    ],
    merchant: MerchantScoringProfile,
    merchant_policy: MerchantPolicy,
    policy_context: PolicyContext,
    source_context: RecoverySourceContext,
    probability_model: RecoveryProbabilityModel,
    economics: RecoveryEconomicsConfig = (
        DEFAULT_RECOVERY_ECONOMICS
    ),
    policy_engine: MerchantPolicyEngine | None = None,
) -> ActionSelectionResult:
    """
    Policy-gate, ML-score, economically rank and select
    one recovery action.

    Authority order:

        deterministic policy
                ↓
        ML recovery probability
                ↓
        deterministic economics
                ↓
        final ranked action

    BLOCKED actions never reach ML.

    ALLOWED, DEFERRED and REQUIRES_APPROVAL actions may
    be economically evaluated, but the returned outcome keeps
    all execution gates explicit.
    """

    _validate_relationships(

        recovery_case=(
            recovery_case
        ),

        customer=(
            customer
        ),

        merchant=(
            merchant
        ),

        policy=(
            merchant_policy
        ),

        context=(
            policy_context
        ),
    )

    if not candidate_actions:

        return ActionSelectionResult(

            case_id=(
                recovery_case.id
            ),

            outcome=(
                SelectionOutcome
                .NO_ELIGIBLE_ACTION
            ),

            explanation=(
                "No candidate recovery "
                "actions were supplied."
            ),
        )

    resolved_policy_engine = (
        policy_engine
        if policy_engine is not None
        else MerchantPolicyEngine()
    )

    # ========================================================
    # POLICY EVALUATION
    # ========================================================

    evaluations = (
        resolved_policy_engine
        .evaluate_candidates(

            recovery_case=(
                recovery_case
            ),

            actions=(
                candidate_actions
            ),

            policy=(
                merchant_policy
            ),

            context=(
                policy_context
            ),
        )
    )

    scoreable: list[
        _ScoreableCandidate
    ] = []

    excluded: list[
        ExcludedAction
    ] = []

    for (
        index,
        (
            action,
            evaluation,
        ),
    ) in enumerate(

        zip(
            candidate_actions,
            evaluations,
            strict=True,
        )
    ):

        (
            candidate,
            effective_evaluation,
        ) = _resolve_candidate_policy(

            recovery_case=(
                recovery_case
            ),

            action=(
                action
            ),

            evaluation=(
                evaluation
            ),

            policy=(
                merchant_policy
            ),

            context=(
                policy_context
            ),

            policy_engine=(
                resolved_policy_engine
            ),

            original_index=(
                index
            ),
        )

        if candidate is None:

            excluded.append(

                ExcludedAction(

                    action_id=(
                        action.id
                    ),

                    action_type=(
                        action.action_type
                    ),

                    policy_decision=(
                        effective_evaluation.decision
                    ),

                    policy_reason=(
                        effective_evaluation.reason.value
                    ),

                    policy_explanation=(
                        effective_evaluation.explanation
                    ),
                )
            )

            continue

        scoreable.append(
            candidate
        )

    # ========================================================
    # NO LEGAL ACTIONS
    # ========================================================

    if not scoreable:

        return ActionSelectionResult(

            case_id=(
                recovery_case.id
            ),

            outcome=(
                SelectionOutcome
                .NO_ELIGIBLE_ACTION
            ),

            excluded_actions=(
                excluded
            ),

            explanation=(
                "Merchant policy blocked every "
                "candidate action before ML scoring."
            ),
        )

    eligible_action_count = (
        len(
            scoreable
        )
    )

    # ========================================================
    # BUILD LIVE ML FEATURES
    # ========================================================

    feature_rows = [

        build_action_feature_row(

            recovery_case=(
                recovery_case
            ),

            customer=(
                customer
            ),

            diagnosis=(
                diagnosis
            ),

            action=(
                candidate.action
            ),

            merchant=(
                merchant
            ),

            source=(
                source_context
            ),

            # Historical training data stores the policy result
            # from the moment the action was first considered.
            policy_decision=(
                candidate
                .initial_evaluation
                .decision
            ),

            eligible_action_count=(
                eligible_action_count
            ),

            selection_time=(
                policy_context.now
            ),

            execute_at=(
                candidate.execute_at
            ),
        )

        for candidate
        in scoreable
    ]

    feature_frame = (
        build_action_feature_frame(
            feature_rows
        )
    )

    # ========================================================
    # ML PREDICTIONS
    # ========================================================

    probabilities = (
        _validate_probabilities(

            probability_model
            .predict_recovery_probability(
                feature_frame
            ),

            expected_count=(
                eligible_action_count
            ),
        )
    )

    # ========================================================
    # ECONOMIC SCORING
    # ========================================================

    unranked = [

        _economic_score(

            recovery_case=(
                recovery_case
            ),

            candidate=(
                candidate
            ),

            probability=float(
                probability
            ),

            economics=(
                economics
            ),

            selection_time=(
                policy_context.now
            ),
        )

        for (
            candidate,
            probability,
        )
        in zip(
            scoreable,
            probabilities,
            strict=True,
        )
    ]

    # ========================================================
    # DETERMINISTIC RANKING
    # ========================================================

    ranked = (
        _rank_scores(
            unranked
        )
    )

    public_scores = [

        _public_score(

            item,

            rank=(
                rank
            ),
        )

        for (
            rank,
            item,
        )
        in enumerate(
            ranked,
            start=1,
        )
    ]

    winner = (
        ranked[0]
    )

    selected = (
        _selected_action(
            winner
        )
    )

    outcome = (
        _selection_outcome(
            winner
        )
    )

    winner_score = (
        public_scores[0]
    )

    explanation = (
        f"Selected "
        f"{winner.action.action_type.value} "
        f"because it had the highest expected "
        f"recovery value "
        f"({winner.expected_recovery_value}) "
        f"among {eligible_action_count} "
        f"policy-eligible actions. "
        f"Policy at selection: "
        f"{winner.initial_evaluation.decision.value}; "
        f"policy at execution: "
        f"{winner.execution_evaluation.decision.value}."
    )

    return ActionSelectionResult(

        case_id=(
            recovery_case.id
        ),

        outcome=(
            outcome
        ),

        selected_action=(
            selected
        ),

        selected_score=(
            winner_score
        ),

        scored_actions=(
            public_scores
        ),

        excluded_actions=(
            excluded
        ),

        explanation=(
            explanation
        ),
    )