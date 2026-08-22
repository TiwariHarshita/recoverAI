from datetime import datetime, timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from app.domain.actions import RecoveryAction
from app.domain.enums import (
    ActionStatus,
    CaseType,
    CommunicationChannel,
    PolicyDecision,
    RecoveryActionType,
    RecoveryCaseStatus,
)
from app.domain.policies import MerchantPolicy
from app.domain.recovery_case import RecoveryCase

from app.policy import (
    MerchantPolicyEngine,
    PolicyContext,
    PolicyReason,
)

from app.services.candidate_actions import (
    generate_candidate_actions,
)
from app.services.diagnosis import diagnose_case


BASE_CREATED_AT = datetime(
    2026,
    8,
    20,
    12,
    0,
    tzinfo=timezone.utc,
)

BASE_NOW = datetime(
    2026,
    8,
    23,
    12,
    0,
    tzinfo=timezone.utc,
)


@pytest.fixture
def base_case() -> RecoveryCase:
    return RecoveryCase(
        merchant_id="merchant_001",
        customer_id="customer_001",
        case_type=CaseType.PAYMENT_FAILURE,
        amount_at_risk=Decimal("10000"),
        attempt_count=1,
        recovery_retry_count=0,
        previous_contacts=0,
        created_at=BASE_CREATED_AT,
        updated_at=BASE_CREATED_AT,
    )


def make_policy(
    merchant_id: str = "merchant_001",
) -> MerchantPolicy:
    return MerchantPolicy(
        merchant_id=merchant_id,
        max_contacts_per_case=3,
        contact_window_days=7,
        max_payment_retries=2,
        max_recovery_window_days=7,
        human_approval_threshold=Decimal("25000"),
        timezone="Asia/Kolkata",
        allowed_channels={
            CommunicationChannel.SMS,
            CommunicationChannel.EMAIL,
            CommunicationChannel.WHATSAPP,
        },
        allowed_actions=set(RecoveryActionType),
    )


def make_context(
    now: datetime = BASE_NOW,
    **kwargs,
) -> PolicyContext:
    return PolicyContext(
        now=now,
        **kwargs,
    )


def make_action(
    case: RecoveryCase,
    action_type: RecoveryActionType,
    channel: CommunicationChannel
    = CommunicationChannel.NONE,
) -> RecoveryAction:
    return RecoveryAction(
        case_id=case.id,
        action_type=action_type,
        channel=channel,
        reason="policy test action",
    )


def executed_contact(
    case: RecoveryCase,
    occurred_at: datetime,
    *,
    channel: CommunicationChannel
    = CommunicationChannel.SMS,
) -> RecoveryAction:
    return RecoveryAction(
        case_id=case.id,
        action_type=RecoveryActionType.SEND_REMINDER,
        channel=channel,
        status=ActionStatus.EXECUTED,
        reason="historical contact",
        created_at=occurred_at,
        executed_at=occurred_at,
    )


# ============================================================
# BASIC ALLOW / BLOCK
# ============================================================


def test_allowed_action_passes(
    base_case,
):
    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.CREATE_PAYMENT_LINK,
            CommunicationChannel.SMS,
        ),
        make_policy(),
        make_context(),
    )

    assert (
        result.decision
        == PolicyDecision.ALLOWED
    )

    assert (
        result.reason
        == PolicyReason.ACTION_ALLOWED
    )


def test_disabled_action_is_blocked(
    base_case,
):
    policy = make_policy()

    policy.allowed_actions.remove(
        RecoveryActionType.SEND_REMINDER
    )

    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.SEND_REMINDER,
            CommunicationChannel.SMS,
        ),
        policy,
        make_context(),
    )

    assert (
        result.decision
        == PolicyDecision.BLOCKED
    )

    assert (
        result.reason
        == PolicyReason.ACTION_DISABLED
    )


# ============================================================
# CHANNEL POLICY
# ============================================================


def test_disallowed_sms_channel_is_blocked(
    base_case,
):
    policy = make_policy()

    policy.allowed_channels.remove(
        CommunicationChannel.SMS
    )

    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.SEND_REMINDER,
            CommunicationChannel.SMS,
        ),
        policy,
        make_context(),
    )

    assert (
        result.decision
        == PolicyDecision.BLOCKED
    )

    assert (
        result.reason
        == PolicyReason.CHANNEL_DISABLED
    )


def test_email_uses_same_send_reminder_action(
    base_case,
):
    policy = make_policy()

    policy.allowed_channels.remove(
        CommunicationChannel.EMAIL
    )

    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.SEND_REMINDER,
            CommunicationChannel.EMAIL,
        ),
        policy,
        make_context(),
    )

    assert (
        result.decision
        == PolicyDecision.BLOCKED
    )

    assert (
        result.reason
        == PolicyReason.CHANNEL_DISABLED
    )


# ============================================================
# PARTIAL PAYMENT
# ============================================================


def test_partial_payment_flag_is_enforced(
    base_case,
):
    policy = make_policy()

    policy.allow_partial_payments = False

    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.OFFER_PARTIAL_PAYMENT,
            CommunicationChannel.EMAIL,
        ),
        policy,
        make_context(),
    )

    assert (
        result.decision
        == PolicyDecision.BLOCKED
    )

    assert (
        result.reason
        == PolicyReason.PARTIAL_PAYMENT_DISABLED
    )


# ============================================================
# DO NOT CONTACT
# ============================================================


def test_do_not_contact_blocks_customer_outreach(
    base_case,
):
    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.CREATE_PAYMENT_LINK,
            CommunicationChannel.SMS,
        ),
        make_policy(),
        make_context(
            customer_do_not_contact=True
        ),
    )

    assert (
        result.decision
        == PolicyDecision.BLOCKED
    )

    assert (
        result.reason
        == PolicyReason.CUSTOMER_DO_NOT_CONTACT
    )


def test_do_not_contact_does_not_block_non_contact_retry(
    base_case,
):
    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.DELAYED_RETRY,
        ),
        make_policy(),
        make_context(
            customer_do_not_contact=True
        ),
    )

    assert (
        result.decision
        == PolicyDecision.ALLOWED
    )


# ============================================================
# SAFE ACTIONS
# ============================================================


def test_stop_is_always_safe_after_validation(
    base_case,
):
    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.STOP,
        ),
        make_policy(),
        make_context(
            dispute_active=True
        ),
    )

    assert (
        result.decision
        == PolicyDecision.ALLOWED
    )


# ============================================================
# ALREADY RECOVERED
# ============================================================


def test_recovered_status_blocks_more_recovery(
    base_case,
):
    base_case.status = (
        RecoveryCaseStatus.RECOVERED
    )

    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.DELAYED_RETRY,
        ),
        make_policy(),
        make_context(),
    )

    assert (
        result.decision
        == PolicyDecision.BLOCKED
    )

    assert (
        result.reason
        == PolicyReason.CASE_ALREADY_RECOVERED
    )


def test_recovered_amount_blocks_more_recovery(
    base_case,
):
    base_case.recovered_amount = (
        base_case.amount_at_risk
    )

    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.DELAYED_RETRY,
        ),
        make_policy(),
        make_context(),
    )

    assert (
        result.decision
        == PolicyDecision.BLOCKED
    )

    assert (
        result.reason
        == PolicyReason.CASE_ALREADY_RECOVERED
    )


# ============================================================
# DISPUTES
# ============================================================


def test_dispute_blocks_collection_action(
    base_case,
):
    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.CREATE_PAYMENT_LINK,
            CommunicationChannel.SMS,
        ),
        make_policy(),
        make_context(
            dispute_active=True
        ),
    )

    assert (
        result.decision
        == PolicyDecision.BLOCKED
    )

    assert (
        result.reason
        == PolicyReason.DISPUTE_ACTIVE
    )


def test_dispute_allows_human_escalation(
    base_case,
):
    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.ESCALATE_TO_HUMAN,
        ),
        make_policy(),
        make_context(
            dispute_active=True
        ),
    )

    assert (
        result.decision
        == PolicyDecision.ALLOWED
    )


# ============================================================
# PROMISE TO PAY
# ============================================================


def test_active_promise_to_pay_defers_recovery(
    base_case,
):
    due_at = (
        BASE_NOW
        + timedelta(days=2)
    )

    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.DELAYED_RETRY,
        ),
        make_policy(),
        make_context(
            active_promise_to_pay=True,
            promise_due_at=due_at,
        ),
    )

    assert (
        result.decision
        == PolicyDecision.DEFERRED
    )

    assert (
        result.reason
        == PolicyReason.ACTIVE_PROMISE_TO_PAY
    )

    assert result.eligible_at == due_at


# ============================================================
# RETRY CEILING
# ============================================================


def test_retry_limit_uses_recovery_retry_count(
    base_case,
):
    base_case.attempt_count = 10

    base_case.recovery_retry_count = 2

    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.DELAYED_RETRY,
        ),
        make_policy(),
        make_context(),
    )

    assert (
        result.decision
        == PolicyDecision.BLOCKED
    )

    assert (
        result.reason
        == PolicyReason.RETRY_LIMIT_REACHED
    )


def test_normal_attempt_count_does_not_consume_retry_budget(
    base_case,
):
    base_case.attempt_count = 10

    base_case.recovery_retry_count = 1

    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.DELAYED_RETRY,
        ),
        make_policy(),
        make_context(),
    )

    assert (
        result.decision
        == PolicyDecision.ALLOWED
    )


# ============================================================
# ROLLING CONTACT WINDOW
# ============================================================


def test_contact_limit_counts_recent_executed_contacts(
    base_case,
):
    history = [
        executed_contact(
            base_case,
            BASE_NOW - timedelta(days=1),
        ),
        executed_contact(
            base_case,
            BASE_NOW - timedelta(days=2),
        ),
        executed_contact(
            base_case,
            BASE_NOW - timedelta(days=3),
        ),
        executed_contact(
            base_case,
            BASE_NOW - timedelta(days=10),
        ),
    ]

    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.SEND_REMINDER,
            CommunicationChannel.SMS,
        ),
        make_policy(),
        make_context(
            action_history=history
        ),
    )

    assert (
        result.decision
        == PolicyDecision.BLOCKED
    )

    assert (
        result.reason
        == PolicyReason.CONTACT_LIMIT_REACHED
    )


def test_old_contacts_do_not_consume_current_window(
    base_case,
):
    history = [
        executed_contact(
            base_case,
            BASE_NOW - timedelta(days=8),
        ),
        executed_contact(
            base_case,
            BASE_NOW - timedelta(days=9),
        ),
        executed_contact(
            base_case,
            BASE_NOW - timedelta(days=10),
        ),
    ]

    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.SEND_REMINDER,
            CommunicationChannel.SMS,
        ),
        make_policy(),
        make_context(
            action_history=history
        ),
    )

    assert (
        result.decision
        == PolicyDecision.ALLOWED
    )


def test_proposed_actions_do_not_count_as_contacts(
    base_case,
):
    proposed = make_action(
        base_case,
        RecoveryActionType.SEND_REMINDER,
        CommunicationChannel.SMS,
    )

    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.SEND_REMINDER,
            CommunicationChannel.SMS,
        ),
        make_policy(),
        make_context(
            action_history=[
                proposed,
                proposed,
                proposed,
            ]
        ),
    )

    assert (
        result.decision
        == PolicyDecision.ALLOWED
    )


def test_other_cases_do_not_consume_contact_limit(
    base_case,
):
    other_case_contact = RecoveryAction(
        case_id="rc_other",
        action_type=RecoveryActionType.SEND_REMINDER,
        channel=CommunicationChannel.SMS,
        status=ActionStatus.EXECUTED,
        reason="other case contact",
        created_at=(
            BASE_NOW
            - timedelta(days=1)
        ),
        executed_at=(
            BASE_NOW
            - timedelta(days=1)
        ),
    )

    history = [
        other_case_contact,
        other_case_contact,
        other_case_contact,
    ]

    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.SEND_REMINDER,
            CommunicationChannel.SMS,
        ),
        make_policy(),
        make_context(
            action_history=history
        ),
    )

    assert (
        result.decision
        == PolicyDecision.ALLOWED
    )


# ============================================================
# QUIET HOURS
# ============================================================


@pytest.mark.parametrize(
    (
        "hour",
        "minute",
        "expected",
    ),
    [
        (
            20,
            59,
            PolicyDecision.ALLOWED,
        ),
        (
            21,
            0,
            PolicyDecision.DEFERRED,
        ),
        (
            23,
            30,
            PolicyDecision.DEFERRED,
        ),
        (
            2,
            0,
            PolicyDecision.DEFERRED,
        ),
        (
            7,
            59,
            PolicyDecision.DEFERRED,
        ),
        (
            8,
            0,
            PolicyDecision.ALLOWED,
        ),
    ],
)
def test_quiet_hour_boundaries(
    base_case,
    hour,
    minute,
    expected,
):
    india = ZoneInfo(
        "Asia/Kolkata"
    )

    now = datetime(
        2026,
        8,
        23,
        hour,
        minute,
        tzinfo=india,
    )

    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.SEND_REMINDER,
            CommunicationChannel.SMS,
        ),
        make_policy(),
        make_context(
            now=now
        ),
    )

    assert result.decision == expected


def test_quiet_hours_return_next_eligible_time(
    base_case,
):
    india = ZoneInfo(
        "Asia/Kolkata"
    )

    now = datetime(
        2026,
        8,
        23,
        23,
        0,
        tzinfo=india,
    )

    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.SEND_REMINDER,
            CommunicationChannel.SMS,
        ),
        make_policy(),
        make_context(
            now=now
        ),
    )

    assert (
        result.decision
        == PolicyDecision.DEFERRED
    )

    assert (
        result.reason
        == PolicyReason.QUIET_HOURS
    )

    assert result.eligible_at == datetime(
        2026,
        8,
        24,
        8,
        0,
        tzinfo=india,
    )


# ============================================================
# HUMAN APPROVAL
# ============================================================


def test_high_value_action_requires_approval(
    base_case,
):
    base_case.amount_at_risk = Decimal(
        "25001"
    )

    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.CREATE_PAYMENT_LINK,
            CommunicationChannel.SMS,
        ),
        make_policy(),
        make_context(),
    )

    assert (
        result.decision
        == PolicyDecision.REQUIRES_APPROVAL
    )

    assert (
        result.reason
        == PolicyReason.HUMAN_APPROVAL_REQUIRED
    )


def test_exact_threshold_is_auto_allowed(
    base_case,
):
    base_case.amount_at_risk = Decimal(
        "25000"
    )

    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.DELAYED_RETRY,
        ),
        make_policy(),
        make_context(),
    )

    assert (
        result.decision
        == PolicyDecision.ALLOWED
    )


# ============================================================
# RECOVERY WINDOW
# ============================================================


def test_expired_recovery_window_blocks_automation(
    base_case,
):
    base_case.created_at = (
        BASE_NOW
        - timedelta(days=8)
    )

    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.DELAYED_RETRY,
        ),
        make_policy(),
        make_context(),
    )

    assert (
        result.decision
        == PolicyDecision.BLOCKED
    )

    assert (
        result.reason
        == PolicyReason.RECOVERY_WINDOW_EXPIRED
    )


def test_expired_window_allows_human_escalation(
    base_case,
):
    base_case.created_at = (
        BASE_NOW
        - timedelta(days=8)
    )

    result = MerchantPolicyEngine().evaluate(
        base_case,
        make_action(
            base_case,
            RecoveryActionType.ESCALATE_TO_HUMAN,
        ),
        make_policy(),
        make_context(),
    )

    assert (
        result.decision
        == PolicyDecision.ALLOWED
    )


# ============================================================
# FAIL-CLOSED VALIDATION
# ============================================================


def test_wrong_merchant_policy_fails_closed(
    base_case,
):
    with pytest.raises(
        ValueError,
        match="does not belong",
    ):
        MerchantPolicyEngine().evaluate(
            base_case,
            make_action(
                base_case,
                RecoveryActionType.WAIT,
            ),
            make_policy(
                "merchant_other"
            ),
            make_context(),
        )


def test_action_from_another_case_fails_closed(
    base_case,
):
    action = RecoveryAction(
        case_id="rc_other",
        action_type=RecoveryActionType.WAIT,
        reason="wrong case",
    )

    with pytest.raises(
        ValueError,
        match="does not belong",
    ):
        MerchantPolicyEngine().evaluate(
            base_case,
            action,
            make_policy(),
            make_context(),
        )


def test_naive_now_fails_closed(
    base_case,
):
    naive_now = datetime(
        2026,
        8,
        23,
        12,
        0,
    )

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        MerchantPolicyEngine().evaluate(
            base_case,
            make_action(
                base_case,
                RecoveryActionType.WAIT,
            ),
            make_policy(),
            make_context(
                now=naive_now
            ),
        )


def test_active_promise_without_due_date_fails_closed(
    base_case,
):
    with pytest.raises(
        ValueError,
        match="promise_due_at is required",
    ):
        MerchantPolicyEngine().evaluate(
            base_case,
            make_action(
                base_case,
                RecoveryActionType.WAIT,
            ),
            make_policy(),
            make_context(
                active_promise_to_pay=True
            ),
        )


def test_invalid_timezone_fails_closed_for_contact(
    base_case,
):
    policy = make_policy()

    policy.timezone = (
        "Not/A/Timezone"
    )

    with pytest.raises(
        ValueError,
        match="Unknown merchant timezone",
    ):
        MerchantPolicyEngine().evaluate(
            base_case,
            make_action(
                base_case,
                RecoveryActionType.SEND_REMINDER,
                CommunicationChannel.SMS,
            ),
            policy,
            make_context(),
        )


# ============================================================
# INTEGRATION WITH CANDIDATE GENERATOR
# ============================================================


def test_evaluate_candidates_preserves_order(
    base_case,
):
    actions = [
        make_action(
            base_case,
            RecoveryActionType.WAIT,
        ),
        make_action(
            base_case,
            RecoveryActionType.CREATE_PAYMENT_LINK,
            CommunicationChannel.SMS,
        ),
        make_action(
            base_case,
            RecoveryActionType.ESCALATE_TO_HUMAN,
        ),
    ]

    results = (
        MerchantPolicyEngine()
        .evaluate_candidates(
            base_case,
            actions,
            make_policy(),
            make_context(),
        )
    )

    assert [
        result.action_id
        for result in results
    ] == [
        action.id
        for action in actions
    ]


def test_candidate_generator_output_plugs_into_policy(
    base_case,
):
    diagnosis_case = (
        base_case.model_copy(
            update={
                "error_reason": "invalid_otp"
            }
        )
    )

    diagnosis = diagnose_case(
        diagnosis_case
    )

    candidates = (
        generate_candidate_actions(
            diagnosis_case,
            diagnosis,
        )
    )

    results = (
        MerchantPolicyEngine()
        .evaluate_candidates(
            diagnosis_case,
            candidates.actions,
            make_policy(),
            make_context(),
        )
    )

    assert [
        result.action_id
        for result in results
    ] == [
        action.id
        for action
        in candidates.actions
    ]

    assert all(
        result.decision
        == PolicyDecision.ALLOWED
        for result in results
    )