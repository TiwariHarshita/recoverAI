from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from app.domain.action_scoring import (
    ActionEconomicRule,
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
    CaseType,
    CommunicationChannel,
    DiagnosisCertainty,
    FailureClass,
    PolicyDecision,
    RecoveryActionType,
)
from app.domain.policies import MerchantPolicy
from app.domain.recovery_case import RecoveryCase
from app.ml.dataset import MODEL_FEATURES
from app.ml.inference_features import (
    build_action_feature_row,
)
from app.policy.models import PolicyContext
from app.services.action_selector import (
    DEFAULT_RECOVERY_ECONOMICS,
    select_best_recovery_action,
)


NOW = datetime(
    2026,
    8,
    23,
    5,
    0,
    tzinfo=timezone.utc,
)


class FixedProbabilityModel:
    """
    Deterministic model used only by unit tests.
    """

    def __init__(
        self,
        probabilities: dict[
            RecoveryActionType,
            float,
        ],
    ) -> None:

        self.probabilities = (
            probabilities
        )

        self.frames: list[
            pd.DataFrame
        ] = []

    def predict_recovery_probability(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:

        self.frames.append(
            dataframe.copy()
        )

        values: list[
            float
        ] = []

        for raw in (
            dataframe[
                "action_type"
            ]
        ):

            if isinstance(
                raw,
                RecoveryActionType,
            ):
                action_type = raw

            else:
                action_type = (
                    RecoveryActionType(
                        str(raw)
                    )
                )

            values.append(
                self.probabilities[
                    action_type
                ]
            )

        return np.asarray(
            values,
            dtype=float,
        )


def make_case(
    *,
    amount: Decimal = Decimal(
        "1000.00"
    ),
) -> RecoveryCase:

    return RecoveryCase(

        id="case_1",

        merchant_id="merchant_1",

        customer_id="customer_1",

        case_type=(
            CaseType.PAYMENT_FAILURE
        ),

        amount_at_risk=(
            amount
        ),

        currency="INR",

        error_code="BAD_REQUEST_ERROR",

        error_source="customer",

        error_step=(
            "payment_authentication"
        ),

        error_reason="invalid_otp",

        attempt_count=1,

        created_at=datetime(
            2026,
            8,
            23,
            2,
            0,
            tzinfo=timezone.utc,
        ),
    )


def make_customer() -> Customer:

    return Customer(

        id="customer_1",

        merchant_id="merchant_1",

        created_at=datetime(
            2025,
            8,
            23,
            tzinfo=timezone.utc,
        ),

        lifetime_value=Decimal(
            "50000.00"
        ),

        successful_payments=12,

        failed_payments=3,

        historical_payment_success_rate=(
            0.80
        ),

        previous_recovery_attempts=4,

        previous_recovery_successes=2,

        preferred_payment_method=None,

        preferred_channel=(
            CommunicationChannel.SMS
        ),

        language_preference="en",

        do_not_contact=False,
    )


def make_diagnosis() -> DiagnosisResult:

    return DiagnosisResult(

        failure_class=(
            FailureClass
            .AUTHENTICATION_FAILURE
        ),

        certainty=(
            DiagnosisCertainty.EXACT
        ),

        summary=(
            "Authentication failed."
        ),

        temporary_failure=True,

        retry_same_method_reasonable=True,

        requires_new_payment_method=False,

        customer_action_required=True,

        merchant_action_required=False,
    )


def make_merchant() -> MerchantScoringProfile:

    return MerchantScoringProfile(

        merchant_id="merchant_1",

        archetype="ecommerce",

        average_order_value=Decimal(
            "800.00"
        ),
    )


def make_policy(
    *,
    allowed_actions: (
        set[
            RecoveryActionType
        ]
        | None
    ) = None,
    approval_threshold: Decimal = Decimal(
        "25000"
    ),
) -> MerchantPolicy:

    return MerchantPolicy(

        merchant_id="merchant_1",

        human_approval_threshold=(
            approval_threshold
        ),

        allowed_actions=(
            allowed_actions
            if allowed_actions is not None
            else set(
                RecoveryActionType
            )
        ),
    )


def make_context(
    *,
    now: datetime = NOW,
) -> PolicyContext:

    return PolicyContext(

        now=now,

        customer_do_not_contact=False,

        action_history=[],
    )


def make_source() -> RecoverySourceContext:

    return RecoverySourceContext(

        bank="HDFC",

        payment_attempt_number=1,
    )


def make_action(
    action_type: RecoveryActionType,
    *,
    channel: CommunicationChannel = (
        CommunicationChannel.NONE
    ),
) -> RecoveryAction:

    return RecoveryAction(

        id=(
            f"action_"
            f"{action_type.value}"
        ),

        case_id="case_1",

        action_type=(
            action_type
        ),

        channel=(
            channel
        ),
    )


def test_live_feature_row_matches_frozen_ml_contract():

    case = (
        make_case()
    )

    customer = (
        make_customer()
    )

    diagnosis = (
        make_diagnosis()
    )

    action = (
        make_action(

            RecoveryActionType
            .CREATE_PAYMENT_LINK,

            channel=(
                CommunicationChannel.SMS
            ),
        )
    )

    row = (
        build_action_feature_row(

            recovery_case=case,

            customer=customer,

            diagnosis=diagnosis,

            action=action,

            merchant=(
                make_merchant()
            ),

            source=(
                make_source()
            ),

            policy_decision=(
                PolicyDecision.ALLOWED
            ),

            eligible_action_count=2,

            selection_time=NOW,

            execute_at=NOW,
        )
    )

    assert (
        list(row)
        == list(
            MODEL_FEATURES
        )
    )

    assert (
        set(row)
        == set(
            MODEL_FEATURES
        )
    )

    assert (
        "recovered"
        not in row
    )

    assert (
        "recovered_amount"
        not in row
    )

    assert (
        "latent_recovery_probability"
        not in row
    )

    assert (
        row[
            "amount_to_average_order_ratio"
        ]
        == pytest.approx(
            1.25
        )
    )

    assert (
        row[
            "previous_recovery_success_rate"
        ]
        == pytest.approx(
            0.5
        )
    )


def test_policy_blocked_actions_never_reach_probability_model():

    wait = (
        make_action(
            RecoveryActionType.WAIT
        )
    )

    link = (
        make_action(

            RecoveryActionType
            .CREATE_PAYMENT_LINK,

            channel=(
                CommunicationChannel.SMS
            ),
        )
    )

    model = (
        FixedProbabilityModel(
            {
                RecoveryActionType.WAIT: (
                    0.20
                ),
            }
        )
    )

    result = (
        select_best_recovery_action(

            recovery_case=(
                make_case()
            ),

            customer=(
                make_customer()
            ),

            diagnosis=(
                make_diagnosis()
            ),

            candidate_actions=[
                link,
                wait,
            ],

            merchant=(
                make_merchant()
            ),

            merchant_policy=(
                make_policy(
                    allowed_actions={
                        RecoveryActionType.WAIT,
                    }
                )
            ),

            policy_context=(
                make_context()
            ),

            source_context=(
                make_source()
            ),

            probability_model=(
                model
            ),
        )
    )

    assert (
        len(
            model.frames
        )
        == 1
    )

    assert (
        len(
            model.frames[0]
        )
        == 1
    )

    assert (
        model.frames[
            0
        ].iloc[
            0
        ][
            "action_type"
        ]
        == RecoveryActionType.WAIT
    )

    assert (
        len(
            result.excluded_actions
        )
        == 1
    )

    assert (
        result
        .excluded_actions[
            0
        ]
        .action_type
        == RecoveryActionType
        .CREATE_PAYMENT_LINK
    )

    assert (
        result
        .excluded_actions[
            0
        ]
        .policy_decision
        == PolicyDecision.BLOCKED
    )


def test_selector_optimizes_expected_value_not_probability_alone():

    link = (
        make_action(

            RecoveryActionType
            .CREATE_PAYMENT_LINK,

            channel=(
                CommunicationChannel.SMS
            ),
        )
    )

    wait = (
        make_action(
            RecoveryActionType.WAIT
        )
    )

    model = (
        FixedProbabilityModel(
            {
                RecoveryActionType
                .CREATE_PAYMENT_LINK: 0.90,

                RecoveryActionType.WAIT: (
                    0.50
                ),
            }
        )
    )

    rules = dict(
        DEFAULT_RECOVERY_ECONOMICS
        .action_rules
    )

    rules[
        RecoveryActionType
        .CREATE_PAYMENT_LINK
    ] = ActionEconomicRule(

        direct_cost=Decimal(
            "700.00"
        ),

        friction_rate=Decimal(
            "0"
        ),
    )

    economics = (
        RecoveryEconomicsConfig(

            action_rules=(
                rules
            ),

            delay_penalty_rate_per_hour=Decimal(
                "0"
            ),
        )
    )

    result = (
        select_best_recovery_action(

            recovery_case=(
                make_case()
            ),

            customer=(
                make_customer()
            ),

            diagnosis=(
                make_diagnosis()
            ),

            candidate_actions=[
                link,
                wait,
            ],

            merchant=(
                make_merchant()
            ),

            merchant_policy=(
                make_policy()
            ),

            policy_context=(
                make_context()
            ),

            source_context=(
                make_source()
            ),

            probability_model=(
                model
            ),

            economics=(
                economics
            ),
        )
    )

    assert (
        result.selected_action
        is not None
    )

    assert (
        result
        .selected_action
        .action_type
        == RecoveryActionType.WAIT
    )

    assert (
        result
        .scored_actions[
            0
        ]
        .action_type
        == RecoveryActionType.WAIT
    )

    assert (
        result
        .scored_actions[
            0
        ]
        .expected_recovery_value
        == Decimal(
            "500.00"
        )
    )


def test_high_value_action_can_win_but_remains_approval_gated():

    link = (
        make_action(

            RecoveryActionType
            .CREATE_PAYMENT_LINK,

            channel=(
                CommunicationChannel.SMS
            ),
        )
    )

    model = (
        FixedProbabilityModel(
            {
                RecoveryActionType
                .CREATE_PAYMENT_LINK: 0.80,
            }
        )
    )

    result = (
        select_best_recovery_action(

            recovery_case=(
                make_case(
                    amount=Decimal(
                        "50000.00"
                    )
                )
            ),

            customer=(
                make_customer()
            ),

            diagnosis=(
                make_diagnosis()
            ),

            candidate_actions=[
                link
            ],

            merchant=(
                make_merchant()
            ),

            merchant_policy=(
                make_policy(
                    approval_threshold=Decimal(
                        "25000"
                    )
                )
            ),

            policy_context=(
                make_context()
            ),

            source_context=(
                make_source()
            ),

            probability_model=(
                model
            ),
        )
    )

    assert (
        result.outcome
        == SelectionOutcome
        .REQUIRE_APPROVAL
    )

    assert (
        result.selected_action
        is not None
    )

    assert (
        result
        .selected_action
        .status
        == ActionStatus
        .REQUIRES_APPROVAL
    )

    assert (
        result.selected_score
        is not None
    )

    assert (
        result
        .selected_score
        .execution_policy_decision
        == PolicyDecision
        .REQUIRES_APPROVAL
    )


def test_deferred_winner_is_returned_as_scheduled():

    # 16:30 UTC = 22:00 Asia/Kolkata.
    # Default quiet hours are active.
    quiet_now = datetime(
        2026,
        8,
        23,
        16,
        30,
        tzinfo=timezone.utc,
    )

    case = (
        make_case()
    )

    case.created_at = datetime(
        2026,
        8,
        23,
        14,
        0,
        tzinfo=timezone.utc,
    )

    link = (
        make_action(

            RecoveryActionType
            .CREATE_PAYMENT_LINK,

            channel=(
                CommunicationChannel.SMS
            ),
        )
    )

    wait = (
        make_action(
            RecoveryActionType.WAIT
        )
    )

    model = (
        FixedProbabilityModel(
            {
                RecoveryActionType
                .CREATE_PAYMENT_LINK: 0.99,

                RecoveryActionType.WAIT: (
                    0.01
                ),
            }
        )
    )

    result = (
        select_best_recovery_action(

            recovery_case=case,

            customer=(
                make_customer()
            ),

            diagnosis=(
                make_diagnosis()
            ),

            candidate_actions=[
                link,
                wait,
            ],

            merchant=(
                make_merchant()
            ),

            merchant_policy=(
                make_policy()
            ),

            policy_context=(
                make_context(
                    now=quiet_now
                )
            ),

            source_context=(
                make_source()
            ),

            probability_model=(
                model
            ),
        )
    )

    assert (
        result.outcome
        == SelectionOutcome.SCHEDULE
    )

    assert (
        result.selected_action
        is not None
    )

    assert (
        result
        .selected_action
        .action_type
        == RecoveryActionType
        .CREATE_PAYMENT_LINK
    )

    assert (
        result
        .selected_action
        .status
        == ActionStatus.SCHEDULED
    )

    assert (
        result
        .selected_action
        .scheduled_for
        is not None
    )

    assert (
        result.selected_score
        is not None
    )

    assert (
        result
        .selected_score
        .policy_decision_at_selection
        == PolicyDecision.DEFERRED
    )

    assert (
        result
        .selected_score
        .execution_policy_decision
        == PolicyDecision.ALLOWED
    )

    assert (
        result
        .selected_score
        .delay_cost
        > Decimal(
            "0"
        )
    )


def test_selected_action_is_enriched_with_probability_and_erv():

    retry = (
        make_action(
            RecoveryActionType
            .IMMEDIATE_RETRY
        )
    )

    model = (
        FixedProbabilityModel(
            {
                RecoveryActionType
                .IMMEDIATE_RETRY: 0.75,
            }
        )
    )

    result = (
        select_best_recovery_action(

            recovery_case=(
                make_case()
            ),

            customer=(
                make_customer()
            ),

            diagnosis=(
                make_diagnosis()
            ),

            candidate_actions=[
                retry
            ],

            merchant=(
                make_merchant()
            ),

            merchant_policy=(
                make_policy()
            ),

            policy_context=(
                make_context()
            ),

            source_context=(
                make_source()
            ),

            probability_model=(
                model
            ),
        )
    )

    assert (
        result.outcome
        == SelectionOutcome.EXECUTE
    )

    assert (
        result.selected_action
        is not None
    )

    assert (
        result.selected_score
        is not None
    )

    assert (
        result
        .selected_action
        .predicted_recovery_probability
        == pytest.approx(
            0.75
        )
    )

    assert (
        result
        .selected_action
        .expected_recovery_value
        == result
        .selected_score
        .expected_recovery_value
    )

    assert (
        result
        .selected_score
        .expected_gross_recovery
        == Decimal(
            "750.00"
        )
    )


def test_ranking_tie_break_preserves_candidate_generator_order():

    stop = (
        make_action(
            RecoveryActionType.STOP
        )
    )

    wait = (
        make_action(
            RecoveryActionType.WAIT
        )
    )

    model = (
        FixedProbabilityModel(
            {
                RecoveryActionType.STOP: 0.0,
                RecoveryActionType.WAIT: 0.0,
            }
        )
    )

    result = (
        select_best_recovery_action(

            recovery_case=(
                make_case()
            ),

            customer=(
                make_customer()
            ),

            diagnosis=(
                make_diagnosis()
            ),

            candidate_actions=[
                stop,
                wait,
            ],

            merchant=(
                make_merchant()
            ),

            merchant_policy=(
                make_policy()
            ),

            policy_context=(
                make_context()
            ),

            source_context=(
                make_source()
            ),

            probability_model=(
                model
            ),
        )
    )

    assert (
        result.selected_action
        is not None
    )

    assert (
        result
        .selected_action
        .action_type
        == RecoveryActionType.STOP
    )

    assert (
        result
        .scored_actions[
            0
        ]
        .action_type
        == RecoveryActionType.STOP
    )

    assert (
        result
        .scored_actions[
            1
        ]
        .action_type
        == RecoveryActionType.WAIT
    )


def test_no_candidates_returns_no_eligible_action_without_calling_model():

    model = (
        FixedProbabilityModel(
            {}
        )
    )

    result = (
        select_best_recovery_action(

            recovery_case=(
                make_case()
            ),

            customer=(
                make_customer()
            ),

            diagnosis=(
                make_diagnosis()
            ),

            candidate_actions=[],

            merchant=(
                make_merchant()
            ),

            merchant_policy=(
                make_policy()
            ),

            policy_context=(
                make_context()
            ),

            source_context=(
                make_source()
            ),

            probability_model=(
                model
            ),
        )
    )

    assert (
        result.outcome
        == SelectionOutcome
        .NO_ELIGIBLE_ACTION
    )

    assert (
        result.selected_action
        is None
    )

    assert (
        result.scored_actions
        == []
    )

    assert (
        model.frames
        == []
    )


def test_invalid_model_probability_fails_closed():

    retry = (
        make_action(
            RecoveryActionType
            .IMMEDIATE_RETRY
        )
    )

    model = (
        FixedProbabilityModel(
            {
                RecoveryActionType
                .IMMEDIATE_RETRY: 1.50,
            }
        )
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):

        select_best_recovery_action(

            recovery_case=(
                make_case()
            ),

            customer=(
                make_customer()
            ),

            diagnosis=(
                make_diagnosis()
            ),

            candidate_actions=[
                retry
            ],

            merchant=(
                make_merchant()
            ),

            merchant_policy=(
                make_policy()
            ),

            policy_context=(
                make_context()
            ),

            source_context=(
                make_source()
            ),

            probability_model=(
                model
            ),
        )