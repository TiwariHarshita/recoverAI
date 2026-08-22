from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.domain.actions import RecoveryAction
from app.domain.enums import (
    ActionStatus,
    CommunicationChannel,
    PolicyDecision,
    RecoveryActionType,
    RecoveryCaseStatus,
)
from app.domain.policies import MerchantPolicy
from app.domain.recovery_case import RecoveryCase
from app.policy.models import (
    PolicyContext,
    PolicyEvaluation,
    PolicyReason,
)


class MerchantPolicyEngine:
    """
    Deterministic merchant guardrail layer.

    This engine does NOT:
    - diagnose failures
    - generate candidate actions
    - rank candidate actions
    - call ML
    - calculate expected recovery value
    - generate customer messages
    - execute Razorpay actions

    It only decides whether a proposed RecoveryAction is
    permitted under merchant policy and current case state.
    """

    CUSTOMER_CONTACT_ACTIONS = {
        RecoveryActionType.CREATE_PAYMENT_LINK,
        RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD,
        RecoveryActionType.SEND_REMINDER,
        RecoveryActionType.OFFER_PARTIAL_PAYMENT,
        RecoveryActionType.REQUEST_PROMISE_TO_PAY,
    }

    RETRY_ACTIONS = {
        RecoveryActionType.IMMEDIATE_RETRY,
        RecoveryActionType.DELAYED_RETRY,
    }

    SAFE_ACTIONS = {
        RecoveryActionType.WAIT,
        RecoveryActionType.ESCALATE_TO_HUMAN,
        RecoveryActionType.STOP,
    }

    def evaluate(
        self,
        recovery_case: RecoveryCase,
        action: RecoveryAction,
        policy: MerchantPolicy,
        context: PolicyContext,
    ) -> PolicyEvaluation:
        """
        Evaluate exactly one candidate RecoveryAction.
        """

        self._validate_inputs(
            recovery_case=recovery_case,
            action=action,
            policy=policy,
            context=context,
        )

        # =====================================================
        # 1. STOP must always remain a safe exit
        # =====================================================

        if action.action_type == RecoveryActionType.STOP:
            return self._allowed(
                action,
                "Stopping automated recovery is always permitted.",
            )

        # =====================================================
        # 2. Fully recovered case
        # =====================================================

        if self._case_is_recovered(recovery_case):
            return self._blocked(
                action,
                PolicyReason.CASE_ALREADY_RECOVERED,
                (
                    "The case is already fully recovered. "
                    "Further recovery actions are blocked."
                ),
            )

        # =====================================================
        # 3. Active dispute
        # =====================================================

        if context.dispute_active:
            if action.action_type in self.SAFE_ACTIONS:
                return self._allowed(
                    action,
                    (
                        "An active dispute exists. Only safe "
                        "non-collection handling is permitted."
                    ),
                )

            return self._blocked(
                action,
                PolicyReason.DISPUTE_ACTIVE,
                (
                    "Recovery action is blocked while an "
                    "active dispute exists."
                ),
            )

        # =====================================================
        # 4. Active promise-to-pay
        # =====================================================

        if (
            context.active_promise_to_pay
            and context.promise_due_at is not None
            and context.now < context.promise_due_at
        ):
            if action.action_type in self.SAFE_ACTIONS:
                return self._allowed(
                    action,
                    (
                        "A promise-to-pay is active. Safe "
                        "non-collection handling remains permitted."
                    ),
                )

            return self._deferred(
                action,
                PolicyReason.ACTIVE_PROMISE_TO_PAY,
                (
                    "Recovery activity is deferred until the "
                    "active promise-to-pay becomes due."
                ),
                context.promise_due_at,
            )

        # =====================================================
        # 5. Recovery window
        # =====================================================

        recovery_deadline = (
            recovery_case.created_at
            + timedelta(
                days=policy.max_recovery_window_days
            )
        )

        if context.now >= recovery_deadline:
            if action.action_type in self.SAFE_ACTIONS:
                return self._allowed(
                    action,
                    (
                        "The automated recovery window has expired. "
                        "Safe handling remains permitted."
                    ),
                )

            return self._blocked(
                action,
                PolicyReason.RECOVERY_WINDOW_EXPIRED,
                (
                    "The merchant-configured automated recovery "
                    "window has expired."
                ),
            )

        # =====================================================
        # 6. Merchant-approved action types
        # =====================================================

        if action.action_type not in policy.allowed_actions:
            return self._blocked(
                action,
                PolicyReason.ACTION_DISABLED,
                (
                    f"{action.action_type.value} is disabled "
                    "by merchant policy."
                ),
            )

        # =====================================================
        # 7. Partial-payment permission
        # =====================================================

        if (
            action.action_type
            == RecoveryActionType.OFFER_PARTIAL_PAYMENT
            and not policy.allow_partial_payments
        ):
            return self._blocked(
                action,
                PolicyReason.PARTIAL_PAYMENT_DISABLED,
                (
                    "Partial-payment recovery is disabled "
                    "by merchant policy."
                ),
            )

        # =====================================================
        # 8. Merchant-approved communication channels
        # =====================================================

        if action.channel != CommunicationChannel.NONE:
            if action.channel not in policy.allowed_channels:
                return self._blocked(
                    action,
                    PolicyReason.CHANNEL_DISABLED,
                    (
                        f"{action.channel.value} is not an "
                        "allowed recovery channel."
                    ),
                )

            if (
                action.channel == CommunicationChannel.VOICE
                and not policy.allow_voice_calls
            ):
                return self._blocked(
                    action,
                    PolicyReason.CHANNEL_DISABLED,
                    (
                        "Voice recovery calls are disabled "
                        "by merchant policy."
                    ),
                )

        # =====================================================
        # 9. RecoverAI retry ceiling
        # =====================================================

        if (
            action.action_type in self.RETRY_ACTIONS
            and recovery_case.recovery_retry_count
            >= policy.max_payment_retries
        ):
            return self._blocked(
                action,
                PolicyReason.RETRY_LIMIT_REACHED,
                (
                    "Maximum RecoverAI retry count reached: "
                    f"{recovery_case.recovery_retry_count}/"
                    f"{policy.max_payment_retries}."
                ),
            )

        # =====================================================
        # 10. Customer-contact guardrails
        # =====================================================

        if self._is_customer_contact(action):

            # -------------------------------------------------
            # Explicit customer do-not-contact
            # -------------------------------------------------

            if context.customer_do_not_contact:
                return self._blocked(
                    action,
                    PolicyReason.CUSTOMER_DO_NOT_CONTACT,
                    (
                        "Customer is marked do-not-contact, "
                        "so automated outreach is blocked."
                    ),
                )

            # -------------------------------------------------
            # Rolling contact limit
            # -------------------------------------------------

            recent_contacts = self._count_recent_contacts(
                recovery_case=recovery_case,
                policy=policy,
                context=context,
            )

            if recent_contacts >= policy.max_contacts_per_case:
                return self._blocked(
                    action,
                    PolicyReason.CONTACT_LIMIT_REACHED,
                    (
                        "Maximum automated contact limit reached: "
                        f"{recent_contacts}/"
                        f"{policy.max_contacts_per_case} "
                        f"within {policy.contact_window_days} days."
                    ),
                )

            # -------------------------------------------------
            # Quiet hours
            # -------------------------------------------------

            local_now = self._merchant_local_time(
                context.now,
                policy.timezone,
            )

            if self._inside_quiet_hours(
                local_now,
                policy,
            ):
                return self._deferred(
                    action,
                    PolicyReason.QUIET_HOURS,
                    (
                        "Customer outreach is inside "
                        "merchant-configured quiet hours."
                    ),
                    self._next_allowed_contact_time(
                        local_now,
                        policy,
                    ),
                )

        # =====================================================
        # 11. Human approval threshold
        # =====================================================

        if (
            action.action_type not in self.SAFE_ACTIONS
            and recovery_case.amount_at_risk
            > policy.human_approval_threshold
        ):
            return self._approval_required(
                action,
                (
                    f"Amount at risk "
                    f"({recovery_case.amount_at_risk}) exceeds "
                    "the merchant automatic-action threshold "
                    f"({policy.human_approval_threshold})."
                ),
            )

        # =====================================================
        # 12. Nothing blocked the action
        # =====================================================

        return self._allowed(
            action,
            (
                "Action satisfies all configured "
                "merchant policy rules."
            ),
        )

    def evaluate_candidates(
        self,
        recovery_case: RecoveryCase,
        actions: list[RecoveryAction],
        policy: MerchantPolicy,
        context: PolicyContext,
    ) -> list[PolicyEvaluation]:
        """
        Evaluate candidate actions while preserving the order
        produced by the Candidate Action Generator.
        """

        return [
            self.evaluate(
                recovery_case=recovery_case,
                action=action,
                policy=policy,
                context=context,
            )
            for action in actions
        ]

    # =========================================================
    # Validation
    # =========================================================

    @staticmethod
    def _validate_inputs(
        recovery_case: RecoveryCase,
        action: RecoveryAction,
        policy: MerchantPolicy,
        context: PolicyContext,
    ) -> None:
        """
        Invalid policy inputs fail closed instead of silently
        allowing a potentially unsafe action.
        """

        if recovery_case.merchant_id != policy.merchant_id:
            raise ValueError(
                (
                    "Merchant policy does not belong to "
                    "the RecoveryCase merchant."
                )
            )

        if action.case_id != recovery_case.id:
            raise ValueError(
                (
                    "RecoveryAction does not belong to "
                    "the supplied RecoveryCase."
                )
            )

        if recovery_case.created_at.tzinfo is None:
            raise ValueError(
                (
                    "RecoveryCase.created_at must be "
                    "timezone-aware."
                )
            )

        if context.now.tzinfo is None:
            raise ValueError(
                "PolicyContext.now must be timezone-aware."
            )

        if (
            context.active_promise_to_pay
            and context.promise_due_at is None
        ):
            raise ValueError(
                (
                    "promise_due_at is required when "
                    "active_promise_to_pay=True."
                )
            )

        if (
            context.promise_due_at is not None
            and context.promise_due_at.tzinfo is None
        ):
            raise ValueError(
                "promise_due_at must be timezone-aware."
            )

        for history_action in context.action_history:
            if (
                history_action.executed_at is not None
                and history_action.executed_at.tzinfo is None
            ):
                raise ValueError(
                    (
                        "Executed action-history timestamps "
                        "must be timezone-aware."
                    )
                )

            if history_action.created_at.tzinfo is None:
                raise ValueError(
                    (
                        "Action-history created_at timestamps "
                        "must be timezone-aware."
                    )
                )

    # =========================================================
    # Case-state helpers
    # =========================================================

    @staticmethod
    def _case_is_recovered(
        recovery_case: RecoveryCase,
    ) -> bool:
        if (
            recovery_case.status
            == RecoveryCaseStatus.RECOVERED
        ):
            return True

        return (
            recovery_case.amount_at_risk > 0
            and recovery_case.recovered_amount
            >= recovery_case.amount_at_risk
        )

    # =========================================================
    # Contact helpers
    # =========================================================

    def _is_customer_contact(
        self,
        action: RecoveryAction,
    ) -> bool:
        return (
            action.action_type
            in self.CUSTOMER_CONTACT_ACTIONS
            and action.channel
            != CommunicationChannel.NONE
        )

    def _count_recent_contacts(
        self,
        recovery_case: RecoveryCase,
        policy: MerchantPolicy,
        context: PolicyContext,
    ) -> int:
        """
        Count only executed customer contacts for THIS case
        inside the rolling contact window.
        """

        window_start = (
            context.now
            - timedelta(
                days=policy.contact_window_days
            )
        )

        count = 0

        for history_action in context.action_history:

            # Another recovery case should never consume
            # this case's contact budget.
            if history_action.case_id != recovery_case.id:
                continue

            if not self._is_customer_contact(
                history_action
            ):
                continue

            # Proposed or scheduled actions have not contacted
            # the customer yet.
            if (
                history_action.status
                != ActionStatus.EXECUTED
            ):
                continue

            occurred_at = (
                history_action.executed_at
                or history_action.created_at
            )

            if (
                window_start
                <= occurred_at
                <= context.now
            ):
                count += 1

        return count

    # =========================================================
    # Time helpers
    # =========================================================

    @staticmethod
    def _merchant_local_time(
        now: datetime,
        timezone_name: str,
    ) -> datetime:
        try:
            timezone = ZoneInfo(
                timezone_name
            )
        except ZoneInfoNotFoundError as exc:
            raise ValueError(
                (
                    "Unknown merchant timezone: "
                    f"{timezone_name}"
                )
            ) from exc

        return now.astimezone(
            timezone
        )

    @staticmethod
    def _inside_quiet_hours(
        local_now: datetime,
        policy: MerchantPolicy,
    ) -> bool:
        current = local_now.time()

        start = policy.quiet_hours_start
        end = policy.quiet_hours_end

        # Same start/end means quiet-hours restriction
        # is disabled.
        if start == end:
            return False

        # Example: 13:00 -> 15:00
        if start < end:
            return start <= current < end

        # Example: 21:00 -> 08:00
        return (
            current >= start
            or current < end
        )

    @staticmethod
    def _next_allowed_contact_time(
        local_now: datetime,
        policy: MerchantPolicy,
    ) -> datetime:
        end = policy.quiet_hours_end
        start = policy.quiet_hours_start

        candidate = local_now.replace(
            hour=end.hour,
            minute=end.minute,
            second=end.second,
            microsecond=0,
        )

        if start > end:
            # Overnight window:
            # 23:00 -> tomorrow 08:00
            # 02:00 -> today 08:00
            if local_now.time() >= start:
                candidate += timedelta(days=1)

        elif candidate <= local_now:
            candidate += timedelta(days=1)

        return candidate

    # =========================================================
    # Result helpers
    # =========================================================

    @staticmethod
    def _allowed(
        action: RecoveryAction,
        explanation: str,
    ) -> PolicyEvaluation:
        return PolicyEvaluation(
            action_id=action.id,
            action_type=action.action_type,
            channel=action.channel,
            decision=PolicyDecision.ALLOWED,
            reason=PolicyReason.ACTION_ALLOWED,
            explanation=explanation,
        )

    @staticmethod
    def _blocked(
        action: RecoveryAction,
        reason: PolicyReason,
        explanation: str,
    ) -> PolicyEvaluation:
        return PolicyEvaluation(
            action_id=action.id,
            action_type=action.action_type,
            channel=action.channel,
            decision=PolicyDecision.BLOCKED,
            reason=reason,
            explanation=explanation,
        )

    @staticmethod
    def _deferred(
        action: RecoveryAction,
        reason: PolicyReason,
        explanation: str,
        eligible_at: datetime,
    ) -> PolicyEvaluation:
        return PolicyEvaluation(
            action_id=action.id,
            action_type=action.action_type,
            channel=action.channel,
            decision=PolicyDecision.DEFERRED,
            reason=reason,
            explanation=explanation,
            eligible_at=eligible_at,
        )

    @staticmethod
    def _approval_required(
        action: RecoveryAction,
        explanation: str,
    ) -> PolicyEvaluation:
        return PolicyEvaluation(
            action_id=action.id,
            action_type=action.action_type,
            channel=action.channel,
            decision=PolicyDecision.REQUIRES_APPROVAL,
            reason=PolicyReason.HUMAN_APPROVAL_REQUIRED,
            explanation=explanation,
        )