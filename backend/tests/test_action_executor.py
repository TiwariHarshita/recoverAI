from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest

from app.domain.actions import RecoveryAction
from app.domain.enums import (
    ActionStatus,
    CaseType,
    RecoveryActionType,
    RecoveryCaseStatus,
)
from app.domain.recovery_case import RecoveryCase
from app.integrations.razorpay import (
    PaymentLinkRequest,
    RazorpayIntegrationError,
    RazorpayTransportError,
)
from app.services.action_executor import RecoveryActionExecutor


NOW = datetime(2026, 8, 24, 17, 30, tzinfo=timezone.utc)


class RecordingRepository:
    def __init__(self) -> None:
        self.saved: list[Any] = []

    def save(self, value: Any) -> Any:
        self.saved.append(value.model_copy(deep=True))
        return value


class RecordingRazorpayClient:
    def __init__(self, response: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.response = response or {
            "id": "plink_test_123",
            "short_url": "https://rzp.io/i/test123",
            "status": "created",
        }
        self.error = error
        self.requests: list[PaymentLinkRequest] = []

    def create_payment_link(self, request: PaymentLinkRequest) -> dict[str, Any]:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return dict(self.response)


@pytest.fixture
def repositories() -> tuple[RecordingRepository, RecordingRepository]:
    return RecordingRepository(), RecordingRepository()


def make_case(
    action: RecoveryAction,
    *,
    status: RecoveryCaseStatus = RecoveryCaseStatus.PLANNED,
) -> RecoveryCase:
    return RecoveryCase(
        id=action.case_id,
        merchant_id="merchant_test",
        customer_id="cust_test",
        case_type=CaseType.PAYMENT_FAILURE,
        status=status,
        amount_at_risk=Decimal("4999.50"),
        currency="INR",
        selected_action_id=action.id,
        created_at=NOW - timedelta(hours=1),
        updated_at=NOW - timedelta(hours=1),
    )


def make_action(
    action_type: RecoveryActionType,
    *,
    status: ActionStatus = ActionStatus.PROPOSED,
    scheduled_for: datetime | None = None,
    amount: Decimal | None = None,
) -> RecoveryAction:
    return RecoveryAction(
        id=f"act_{action_type.value}",
        case_id="rc_test",
        action_type=action_type,
        status=status,
        scheduled_for=scheduled_for,
        amount=amount,
    )


def executor(
    repositories: tuple[RecordingRepository, RecordingRepository],
    *,
    razorpay_client: RecordingRazorpayClient | None = None,
) -> RecoveryActionExecutor:
    cases, actions = repositories
    return RecoveryActionExecutor(
        case_repository=cases,
        action_repository=actions,
        razorpay_client=razorpay_client,  # type: ignore[arg-type]
        clock=lambda: NOW,
    )


def test_create_payment_link_executes_selected_action_without_sending_message(repositories) -> None:
    action = make_action(RecoveryActionType.CREATE_PAYMENT_LINK)
    case = make_case(action)
    client = RecordingRazorpayClient()

    result = executor(repositories, razorpay_client=client).execute(case, action)

    assert result is action
    assert action.status == ActionStatus.EXECUTED
    assert action.executed_at == NOW
    assert case.status == RecoveryCaseStatus.ACTION_EXECUTED
    assert case.updated_at == NOW
    assert len(client.requests) == 1

    request = client.requests[0]
    assert request.amount_subunits == 499950
    assert request.currency == "INR"
    assert request.reference_id == action.id
    assert request.notify_sms is False
    assert request.notify_email is False
    assert request.reminder_enable is False
    assert request.notes == {
        "recoverai_case_id": case.id,
        "recoverai_action_id": action.id,
    }
    assert action.metadata["razorpay_payment_link_id"] == "plink_test_123"
    assert action.metadata["razorpay_payment_link_short_url"] == "https://rzp.io/i/test123"

    case_repo, action_repo = repositories
    assert len(case_repo.saved) == 1
    assert len(action_repo.saved) == 1


def test_create_payment_link_respects_amount_already_decided_upstream(repositories) -> None:
    action = make_action(
        RecoveryActionType.CREATE_PAYMENT_LINK,
        amount=Decimal("1200.25"),
    )
    case = make_case(action)
    client = RecordingRazorpayClient()

    executor(repositories, razorpay_client=client).execute(case, action)

    assert client.requests[0].amount_subunits == 120025


def test_create_payment_link_is_idempotent_after_execution(repositories) -> None:
    action = make_action(
        RecoveryActionType.CREATE_PAYMENT_LINK,
        status=ActionStatus.EXECUTED,
    )
    action.executed_at = NOW - timedelta(minutes=5)
    action.metadata["razorpay_payment_link_id"] = "plink_existing"
    case = make_case(action, status=RecoveryCaseStatus.ACTION_EXECUTED)
    client = RecordingRazorpayClient()

    result = executor(repositories, razorpay_client=client).execute(case, action)

    assert result is action
    assert client.requests == []
    assert repositories[0].saved == []
    assert repositories[1].saved == []


def test_provider_error_marks_action_failed_and_reraises_existing_exception(repositories) -> None:
    action = make_action(RecoveryActionType.CREATE_PAYMENT_LINK)
    case = make_case(action)
    error = RazorpayTransportError("Razorpay API request timed out")
    client = RecordingRazorpayClient(error=error)

    with pytest.raises(RazorpayTransportError) as caught:
        executor(repositories, razorpay_client=client).execute(case, action)

    assert caught.value is error
    assert action.status == ActionStatus.FAILED
    assert action.executed_at is None
    assert case.status == RecoveryCaseStatus.PLANNED
    assert action.metadata["execution_error_type"] == "RazorpayTransportError"
    assert len(repositories[0].saved) == 1
    assert len(repositories[1].saved) == 1


def test_invalid_payment_link_response_uses_existing_integration_error_hierarchy(repositories) -> None:
    action = make_action(RecoveryActionType.CREATE_PAYMENT_LINK)
    case = make_case(action)
    client = RecordingRazorpayClient(response={"status": "created"})

    with pytest.raises(RazorpayIntegrationError):
        executor(repositories, razorpay_client=client).execute(case, action)

    assert action.status == ActionStatus.FAILED


def test_delayed_retry_records_scheduling_state_only(repositories) -> None:
    scheduled_for = NOW + timedelta(hours=4)
    action = make_action(
        RecoveryActionType.DELAYED_RETRY,
        scheduled_for=scheduled_for,
    )
    case = make_case(action)

    executor(repositories).execute(case, action)

    assert action.status == ActionStatus.SCHEDULED
    assert action.scheduled_for == scheduled_for
    assert action.executed_at is None
    assert action.metadata["execution_mode"] == "scheduled_state_only"
    assert case.status == RecoveryCaseStatus.ACTION_SCHEDULED


def test_delayed_retry_requires_upstream_schedule(repositories) -> None:
    action = make_action(RecoveryActionType.DELAYED_RETRY)
    case = make_case(action)

    with pytest.raises(ValueError, match="requires scheduled_for"):
        executor(repositories).execute(case, action)

    assert action.status == ActionStatus.PROPOSED
    assert repositories[0].saved == []
    assert repositories[1].saved == []


def test_scheduled_delayed_retry_is_idempotent(repositories) -> None:
    action = make_action(
        RecoveryActionType.DELAYED_RETRY,
        status=ActionStatus.SCHEDULED,
        scheduled_for=NOW + timedelta(hours=2),
    )
    case = make_case(action, status=RecoveryCaseStatus.ACTION_SCHEDULED)

    result = executor(repositories).execute(case, action)

    assert result is action
    assert repositories[0].saved == []
    assert repositories[1].saved == []


def test_wait_has_no_external_side_effect_and_moves_case_to_waiting(repositories) -> None:
    action = make_action(RecoveryActionType.WAIT)
    case = make_case(action)

    executor(repositories).execute(case, action)

    assert action.status == ActionStatus.EXECUTED
    assert action.executed_at == NOW
    assert action.metadata["execution_mode"] == "no_external_side_effect"
    assert case.status == RecoveryCaseStatus.WAITING_CUSTOMER


def test_escalate_to_human_records_escalation_state(repositories) -> None:
    action = make_action(RecoveryActionType.ESCALATE_TO_HUMAN)
    case = make_case(action)

    executor(repositories).execute(case, action)

    assert action.status == ActionStatus.EXECUTED
    assert case.status == RecoveryCaseStatus.ESCALATED
    assert case.closed_at is None


def test_stop_closes_case(repositories) -> None:
    action = make_action(RecoveryActionType.STOP)
    case = make_case(action)

    executor(repositories).execute(case, action)

    assert action.status == ActionStatus.EXECUTED
    assert case.status == RecoveryCaseStatus.STOPPED
    assert case.closed_at == NOW


@pytest.mark.parametrize(
    "action_type",
    [
        RecoveryActionType.IMMEDIATE_RETRY,
        RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD,
        RecoveryActionType.SEND_REMINDER,
        RecoveryActionType.REQUEST_PROMISE_TO_PAY,
        RecoveryActionType.OFFER_PARTIAL_PAYMENT,
    ],
)
def test_future_responsibility_actions_are_explicitly_rejected(repositories, action_type) -> None:
    action = make_action(action_type)
    case = make_case(action)

    with pytest.raises(ValueError, match="intentionally deferred"):
        executor(repositories).execute(case, action)

    assert repositories[0].saved == []
    assert repositories[1].saved == []


def test_action_still_requiring_human_approval_cannot_execute(repositories) -> None:
    action = make_action(
        RecoveryActionType.CREATE_PAYMENT_LINK,
        status=ActionStatus.REQUIRES_APPROVAL,
    )
    case = make_case(action, status=RecoveryCaseStatus.WAITING_APPROVAL)

    with pytest.raises(ValueError, match="human approval"):
        executor(repositories, razorpay_client=RecordingRazorpayClient()).execute(case, action)


def test_approved_action_can_execute(repositories) -> None:
    action = make_action(
        RecoveryActionType.WAIT,
        status=ActionStatus.APPROVED,
    )
    case = make_case(action)

    executor(repositories).execute(case, action)

    assert action.status == ActionStatus.EXECUTED


def test_non_selected_action_cannot_execute(repositories) -> None:
    action = make_action(RecoveryActionType.WAIT)
    case = make_case(action)
    case.selected_action_id = "act_different"

    with pytest.raises(ValueError, match="not the action selected upstream"):
        executor(repositories).execute(case, action)


def test_action_from_different_case_cannot_execute(repositories) -> None:
    action = make_action(RecoveryActionType.WAIT)
    case = make_case(action)
    action.case_id = "rc_other"

    with pytest.raises(ValueError, match="does not belong"):
        executor(repositories).execute(case, action)


def test_recovered_case_cannot_start_new_execution(repositories) -> None:
    action = make_action(RecoveryActionType.WAIT)
    case = make_case(action, status=RecoveryCaseStatus.RECOVERED)

    with pytest.raises(ValueError, match="cannot execute a new action"):
        executor(repositories).execute(case, action)
