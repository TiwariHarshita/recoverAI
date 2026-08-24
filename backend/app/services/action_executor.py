from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from app.domain.actions import RecoveryAction
from app.domain.enums import ActionStatus, RecoveryActionType, RecoveryCaseStatus
from app.domain.recovery_case import RecoveryCase
from app.integrations.razorpay import (
    PaymentLinkRequest,
    RazorpayClient,
    RazorpayIntegrationError,
)

if TYPE_CHECKING:
    from app.db.repositories import RecoveryActionRepository, RecoveryCaseRepository


_IMPLEMENTED_ACTIONS = {
    RecoveryActionType.CREATE_PAYMENT_LINK,
    RecoveryActionType.DELAYED_RETRY,
    RecoveryActionType.WAIT,
    RecoveryActionType.ESCALATE_TO_HUMAN,
    RecoveryActionType.STOP,
}

_DEFERRED_ACTIONS = {
    RecoveryActionType.IMMEDIATE_RETRY,
    RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD,
    RecoveryActionType.SEND_REMINDER,
    RecoveryActionType.REQUEST_PROMISE_TO_PAY,
    RecoveryActionType.OFFER_PARTIAL_PAYMENT,
}

_TERMINAL_ACTION_STATUSES = {
    ActionStatus.BLOCKED,
    ActionStatus.FAILED,
    ActionStatus.CANCELLED,
}


class RecoveryActionExecutor:
    """Execute an already-selected recovery action.

    Layer 18 deliberately does not diagnose, rank, re-check policy, approve,
    orchestrate, audit, or interpret outcomes. It only translates a decision
    made upstream into the supported execution-side effect/state transition.

    In the current selector contract, a policy-allowed selected action can
    remain ``PROPOSED`` while approval-gated actions are explicitly marked
    ``REQUIRES_APPROVAL``. Therefore a ``PROPOSED`` action is executable only
    when ``RecoveryCase.selected_action_id`` proves it was selected upstream.
    """

    def __init__(
        self,
        *,
        case_repository: RecoveryCaseRepository,
        action_repository: RecoveryActionRepository,
        razorpay_client: RazorpayClient | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.case_repository = case_repository
        self.action_repository = action_repository
        self.razorpay_client = razorpay_client
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def execute(
        self,
        recovery_case: RecoveryCase,
        action: RecoveryAction,
    ) -> RecoveryAction:
        self._validate_identity(recovery_case, action)

        if action.action_type in _DEFERRED_ACTIONS:
            raise ValueError(
                f"{action.action_type.value} is intentionally deferred beyond canonical Layer 18."
            )
        if action.action_type not in _IMPLEMENTED_ACTIONS:
            raise ValueError(f"Unsupported recovery action: {action.action_type.value}")

        if action.action_type == RecoveryActionType.DELAYED_RETRY:
            if action.status == ActionStatus.SCHEDULED:
                return action
        elif action.status == ActionStatus.EXECUTED:
            return action

        self._validate_execution_gate(recovery_case, action)

        if action.action_type == RecoveryActionType.CREATE_PAYMENT_LINK:
            return self._create_payment_link(recovery_case, action)
        if action.action_type == RecoveryActionType.DELAYED_RETRY:
            return self._schedule_delayed_retry(recovery_case, action)
        if action.action_type == RecoveryActionType.WAIT:
            return self._record_wait(recovery_case, action)
        if action.action_type == RecoveryActionType.ESCALATE_TO_HUMAN:
            return self._record_escalation(recovery_case, action)
        if action.action_type == RecoveryActionType.STOP:
            return self._record_stop(recovery_case, action)

        raise AssertionError("Layer 18 action dispatch is incomplete")

    @staticmethod
    def _validate_identity(recovery_case: RecoveryCase, action: RecoveryAction) -> None:
        if action.case_id != recovery_case.id:
            raise ValueError("RecoveryAction does not belong to the supplied RecoveryCase.")
        if recovery_case.selected_action_id != action.id:
            raise ValueError(
                "RecoveryAction is not the action selected upstream for this RecoveryCase."
            )

    @staticmethod
    def _validate_execution_gate(
        recovery_case: RecoveryCase,
        action: RecoveryAction,
    ) -> None:
        if action.status == ActionStatus.REQUIRES_APPROVAL:
            raise ValueError("RecoveryAction still requires authoritative human approval.")
        if action.status in _TERMINAL_ACTION_STATUSES:
            raise ValueError(
                f"RecoveryAction with status {action.status.value} cannot be executed."
            )
        if action.status == ActionStatus.SCHEDULED:
            raise ValueError(
                "Scheduled actions are not executed by Layer 18 before orchestration releases them."
            )
        if action.status not in {ActionStatus.PROPOSED, ActionStatus.APPROVED}:
            raise ValueError(
                f"RecoveryAction with status {action.status.value} is not executable."
            )
        if recovery_case.status in {
            RecoveryCaseStatus.RECOVERED,
            RecoveryCaseStatus.STOPPED,
        }:
            raise ValueError(
                f"RecoveryCase with status {recovery_case.status.value} cannot execute a new action."
            )

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Layer 18 execution clock must return a timezone-aware datetime.")
        return value

    @staticmethod
    def _amount_to_subunits(amount: Decimal) -> int:
        if amount <= 0:
            raise ValueError("Payment Link amount must be greater than zero.")
        subunits = amount * Decimal("100")
        if subunits != subunits.to_integral_value():
            raise ValueError("Payment Link amount cannot contain fractions below one subunit.")
        return int(subunits)

    def _payment_link_request(
        self,
        recovery_case: RecoveryCase,
        action: RecoveryAction,
    ) -> PaymentLinkRequest:
        amount = action.amount if action.amount is not None else recovery_case.amount_at_risk
        return PaymentLinkRequest(
            amount_subunits=self._amount_to_subunits(amount),
            currency=recovery_case.currency.upper(),
            description=f"RecoverAI recovery for case {recovery_case.id}",
            reference_id=action.id,
            notify_sms=False,
            notify_email=False,
            reminder_enable=False,
            notes={
                "recoverai_case_id": recovery_case.id,
                "recoverai_action_id": action.id,
            },
        )

    def _client(self) -> RazorpayClient:
        if self.razorpay_client is None:
            self.razorpay_client = RazorpayClient()
        return self.razorpay_client

    def _create_payment_link(
        self,
        recovery_case: RecoveryCase,
        action: RecoveryAction,
    ) -> RecoveryAction:
        request = self._payment_link_request(recovery_case, action)
        now = self._now()

        try:
            payload = self._client().create_payment_link(request)
            payment_link_id = str(payload.get("id") or "").strip()
            if not payment_link_id.startswith("plink_"):
                raise RazorpayIntegrationError(
                    "Razorpay Payment Link response did not contain a valid plink_ id."
                )
        except RazorpayIntegrationError as exc:
            action.status = ActionStatus.FAILED
            action.metadata = {
                **action.metadata,
                "execution_error_type": type(exc).__name__,
                "execution_error": str(exc),
            }
            recovery_case.updated_at = now
            self._persist(recovery_case, action)
            raise

        action.status = ActionStatus.EXECUTED
        action.executed_at = now
        action.metadata = {
            **action.metadata,
            "razorpay_payment_link_id": payment_link_id,
            "razorpay_payment_link_short_url": payload.get("short_url"),
            "razorpay_payment_link_status": payload.get("status"),
        }
        recovery_case.status = RecoveryCaseStatus.ACTION_EXECUTED
        recovery_case.updated_at = now
        self._persist(recovery_case, action)
        return action

    def _schedule_delayed_retry(
        self,
        recovery_case: RecoveryCase,
        action: RecoveryAction,
    ) -> RecoveryAction:
        if action.scheduled_for is None:
            raise ValueError(
                "DELAYED_RETRY requires scheduled_for to be decided before Layer 18."
            )
        if action.scheduled_for.tzinfo is None or action.scheduled_for.utcoffset() is None:
            raise ValueError("DELAYED_RETRY scheduled_for must be timezone-aware.")

        now = self._now()
        action.status = ActionStatus.SCHEDULED
        action.executed_at = None
        action.metadata = {
            **action.metadata,
            "execution_mode": "scheduled_state_only",
        }
        recovery_case.status = RecoveryCaseStatus.ACTION_SCHEDULED
        recovery_case.updated_at = now
        self._persist(recovery_case, action)
        return action

    def _record_wait(
        self,
        recovery_case: RecoveryCase,
        action: RecoveryAction,
    ) -> RecoveryAction:
        now = self._now()
        action.status = ActionStatus.EXECUTED
        action.executed_at = now
        action.metadata = {
            **action.metadata,
            "execution_mode": "no_external_side_effect",
        }
        recovery_case.status = RecoveryCaseStatus.WAITING_CUSTOMER
        recovery_case.updated_at = now
        self._persist(recovery_case, action)
        return action

    def _record_escalation(
        self,
        recovery_case: RecoveryCase,
        action: RecoveryAction,
    ) -> RecoveryAction:
        now = self._now()
        action.status = ActionStatus.EXECUTED
        action.executed_at = now
        action.metadata = {
            **action.metadata,
            "execution_mode": "human_escalation_state",
        }
        recovery_case.status = RecoveryCaseStatus.ESCALATED
        recovery_case.updated_at = now
        self._persist(recovery_case, action)
        return action

    def _record_stop(
        self,
        recovery_case: RecoveryCase,
        action: RecoveryAction,
    ) -> RecoveryAction:
        now = self._now()
        action.status = ActionStatus.EXECUTED
        action.executed_at = now
        action.metadata = {
            **action.metadata,
            "execution_mode": "stop_state",
        }
        recovery_case.status = RecoveryCaseStatus.STOPPED
        recovery_case.updated_at = now
        recovery_case.closed_at = now
        self._persist(recovery_case, action)
        return action

    def _persist(self, recovery_case: RecoveryCase, action: RecoveryAction) -> None:
        self.case_repository.save(recovery_case)
        self.action_repository.save(action)
