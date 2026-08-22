from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.domain.actions import RecoveryAction
from app.domain.enums import (
    CommunicationChannel,
    PolicyDecision,
    RecoveryActionType,
)


class PolicyReason(str, Enum):
    ACTION_ALLOWED = "action_allowed"

    ACTION_DISABLED = "action_disabled"

    CHANNEL_DISABLED = "channel_disabled"

    PARTIAL_PAYMENT_DISABLED = "partial_payment_disabled"

    CUSTOMER_DO_NOT_CONTACT = "customer_do_not_contact"

    CASE_ALREADY_RECOVERED = "case_already_recovered"

    DISPUTE_ACTIVE = "dispute_active"

    ACTIVE_PROMISE_TO_PAY = "active_promise_to_pay"

    CONTACT_LIMIT_REACHED = "contact_limit_reached"

    RETRY_LIMIT_REACHED = "retry_limit_reached"

    RECOVERY_WINDOW_EXPIRED = "recovery_window_expired"

    QUIET_HOURS = "quiet_hours"

    HUMAN_APPROVAL_REQUIRED = "human_approval_required"


class PolicyContext(BaseModel):
    """
    Runtime facts needed by the policy engine that are not
    permanent MerchantPolicy configuration.
    """

    now: datetime

    customer_do_not_contact: bool = False

    dispute_active: bool = False

    active_promise_to_pay: bool = False

    promise_due_at: datetime | None = None

    # Previous RecoveryActions for rolling-window policy checks.
    action_history: list[RecoveryAction] = Field(
        default_factory=list
    )


class PolicyEvaluation(BaseModel):
    """
    Explainable result of evaluating one candidate RecoveryAction.
    """

    action_id: str

    action_type: RecoveryActionType

    channel: CommunicationChannel

    decision: PolicyDecision

    reason: PolicyReason

    explanation: str

    # Used when decision == DEFERRED.
    eligible_at: datetime | None = None