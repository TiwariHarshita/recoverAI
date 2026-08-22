from __future__ import annotations

from datetime import datetime, time
from enum import Enum

from pydantic import BaseModel, Field

from app.domain.enums import RecoveryActionType


class PolicyDecisionStatus(str, Enum):
    """
    Final result of evaluating one candidate action
    against merchant policy.
    """

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    DEFER = "DEFER"


class PolicyReason(str, Enum):
    """
    Machine-readable reason codes.

    These codes are intentionally stable because they will later
    appear in:
    - audit events
    - API responses
    - dashboard explanations
    - LangGraph state
    - experiment results
    """

    ACTION_ALLOWED = "ACTION_ALLOWED"

    ACTION_DISABLED = "ACTION_DISABLED"
    CHANNEL_DISABLED = "CHANNEL_DISABLED"

    CUSTOMER_OPTED_OUT = "CUSTOMER_OPTED_OUT"
    PAYMENT_ALREADY_RECOVERED = "PAYMENT_ALREADY_RECOVERED"
    DISPUTE_ACTIVE = "DISPUTE_ACTIVE"

    ACTIVE_PROMISE_TO_PAY = "ACTIVE_PROMISE_TO_PAY"

    CONTACT_LIMIT_REACHED = "CONTACT_LIMIT_REACHED"
    RETRY_LIMIT_REACHED = "RETRY_LIMIT_REACHED"
    RECOVERY_WINDOW_EXPIRED = "RECOVERY_WINDOW_EXPIRED"

    QUIET_HOURS = "QUIET_HOURS"

    VALUE_REQUIRES_APPROVAL = "VALUE_REQUIRES_APPROVAL"

    INVALID_POLICY_CONTEXT = "INVALID_POLICY_CONTEXT"


class RecoveryActionHistoryItem(BaseModel):
    """
    One historical recovery action.

    We store the timestamp because contact limits are rolling-window
    limits, not lifetime counters.
    """

    action: RecoveryActionType
    occurred_at: datetime


class MerchantPolicy(BaseModel):
    """
    Merchant-controlled recovery guardrails.

    The defaults below are demo/product defaults only.
    They are configuration, not universal compliance rules.
    """

    merchant_id: str

    enabled_actions: set[RecoveryActionType]

    sms_enabled: bool = True
    email_enabled: bool = True

    # Maximum automated customer contacts inside rolling window.
    max_automated_contacts: int = Field(
        default=3,
        ge=0,
    )

    contact_window_days: int = Field(
        default=7,
        ge=1,
    )

    # RecoverAI-triggered payment retries.
    max_recovery_retries: int = Field(
        default=2,
        ge=0,
    )

    # Merchant-local quiet hours.
    quiet_hours_start: time = time(21, 0)
    quiet_hours_end: time = time(8, 0)

    # Consumer default. B2B can override later.
    recovery_window_days: int = Field(
        default=7,
        ge=1,
    )

    # IMPORTANT:
    # This must use the same canonical money unit as RecoveryCase.amount_at_risk.
    approval_amount_threshold: int = Field(
        default=25_000,
        ge=0,
    )

    timezone: str = "Asia/Kolkata"


class PolicyContext(BaseModel):
    """
    Runtime state required to safely evaluate policy.

    It contains information that is not necessarily intrinsic
    to RecoveryCase itself.
    """

    now: datetime

    customer_opted_out: bool = False

    dispute_active: bool = False

    payment_already_recovered: bool = False

    active_promise_to_pay: bool = False
    promise_due_at: datetime | None = None

    action_history: list[RecoveryActionHistoryItem] = Field(
        default_factory=list
    )


class PolicyDecision(BaseModel):
    """
    Explainable policy decision for one action.
    """

    action: RecoveryActionType

    status: PolicyDecisionStatus

    reason: PolicyReason

    explanation: str

    # Used for DEFER decisions such as quiet hours/PTP.
    eligible_at: datetime | None = None