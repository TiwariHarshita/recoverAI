from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from .actions import RecoveryAction
from .enums import PolicyDecision, RecoveryActionType
from .merchant import Merchant


class SelectionOutcome(str, Enum):
    """
    What must happen next for the selected action.
    """

    EXECUTE = "execute"
    SCHEDULE = "schedule"
    REQUIRE_APPROVAL = "require_approval"
    NO_ELIGIBLE_ACTION = "no_eligible_action"


# Backward-compatible public name. There is one production merchant model;
# scoring no longer defines a competing merchant representation.
MerchantScoringProfile = Merchant


class RecoverySourceContext(BaseModel):
    """
    Provider/source facts that are not stored directly
    on RecoveryCase.
    """

    bank: str | None = None

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


class ActionEconomicRule(BaseModel):
    """
    Deterministic economics for one recovery action.

    These values are merchant/product configuration.
    They are not Razorpay fee claims.
    """

    direct_cost: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    friction_rate: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        le=1,
    )

    expected_recovery_fraction: Decimal = Field(
        default=Decimal("1"),
        ge=0,
        le=1,
    )


class RecoveryEconomicsConfig(BaseModel):
    """
    Merchant-overridable economics used for action ranking.
    """

    action_rules: dict[
        RecoveryActionType,
        ActionEconomicRule,
    ]

    # Opportunity/time penalty while an action is deferred.
    #
    # 0.00001 means 0.001% of amount-at-risk per deferred hour.
    delay_penalty_rate_per_hour: Decimal = Field(
        default=Decimal("0.00001"),
        ge=0,
    )

    @model_validator(
        mode="after"
    )
    def require_every_action_type(
        self,
    ) -> "RecoveryEconomicsConfig":

        missing = (
            set(RecoveryActionType)
            - set(self.action_rules)
        )

        if missing:
            raise ValueError(
                (
                    "Economics config is missing action rules for: "
                    f"{sorted(action.value for action in missing)}"
                )
            )

        return self


class ActionScore(BaseModel):
    """
    Explainable economic score for one policy-eligible action.
    """

    rank: int = Field(
        ge=1
    )

    action_id: str

    action_type: RecoveryActionType

    # Initial policy result when RecoverAI considered the action.
    policy_decision_at_selection: PolicyDecision

    policy_reason_at_selection: str

    policy_explanation_at_selection: str

    # Deferred actions are re-evaluated at eligible_at.
    # For immediate actions this equals the initial decision.
    execution_policy_decision: PolicyDecision

    execution_policy_reason: str

    execution_policy_explanation: str

    eligible_at: datetime | None = None

    predicted_recovery_probability: float = Field(
        ge=0,
        le=1,
    )

    recoverable_amount: Decimal = Field(
        ge=0
    )

    expected_gross_recovery: Decimal = Field(
        ge=0
    )

    direct_action_cost: Decimal = Field(
        ge=0
    )

    friction_cost: Decimal = Field(
        ge=0
    )

    delay_cost: Decimal = Field(
        ge=0
    )

    expected_recovery_value: Decimal

    explanation: str


class ExcludedAction(BaseModel):
    """
    Candidate rejected by deterministic policy before ML scoring.
    """

    action_id: str

    action_type: RecoveryActionType

    policy_decision: PolicyDecision

    policy_reason: str

    policy_explanation: str


class ActionSelectionResult(BaseModel):
    """
    Final deterministic action-selection result.
    """

    case_id: str

    outcome: SelectionOutcome

    selected_action: RecoveryAction | None = None

    selected_score: ActionScore | None = None

    scored_actions: list[
        ActionScore
    ] = Field(
        default_factory=list
    )

    excluded_actions: list[
        ExcludedAction
    ] = Field(
        default_factory=list
    )

    explanation: str
