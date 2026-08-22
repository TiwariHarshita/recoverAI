from decimal import Decimal

from app.domain.actions import RecoveryAction
from app.domain.candidate_actions import CandidateActionSet
from app.domain.diagnosis import DiagnosisResult
from app.domain.enums import (
    CommunicationChannel,
    FailureClass,
    RecoveryActionType,
)
from app.domain.recovery_case import RecoveryCase


def make_action(
    case: RecoveryCase,
    action_type: RecoveryActionType,
    *,
    channel: CommunicationChannel = CommunicationChannel.NONE,
    amount: Decimal | None = None,
    reason: str,
) -> RecoveryAction:
    """
    Helper for creating normalized candidate actions.
    """

    return RecoveryAction(
        case_id=case.id,
        action_type=action_type,
        channel=channel,
        amount=amount,
        reason=reason,
    )


def generate_candidate_actions(
    case: RecoveryCase,
    diagnosis: DiagnosisResult,
) -> CandidateActionSet:
    """
    Return the recovery actions that are logically sensible
    for the diagnosed failure.

    This function does NOT:
    - rank actions
    - call ML
    - check merchant policies
    - execute anything

    Those are separate layers.
    """

    actions: list[RecoveryAction] = []
    notes: list[str] = []

    failure = diagnosis.failure_class

    # ========================================================
    # AUTHENTICATION FAILURE
    # ========================================================

    if failure == FailureClass.AUTHENTICATION_FAILURE:
        actions.extend(
            [
                make_action(
                    case,
                    RecoveryActionType.CREATE_PAYMENT_LINK,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "A fresh payment attempt can allow the customer "
                        "to complete authentication again."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.IMMEDIATE_RETRY,
                    reason=(
                        "Authentication failures are often temporary and "
                        "may succeed on a fresh attempt."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.SEND_REMINDER,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "A reminder can bring the customer back into the "
                        "payment flow."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.WAIT,
                    reason=(
                        "Waiting is retained as a valid low-friction "
                        "alternative."
                    ),
                ),
            ]
        )

        notes.append(
            "Same-method recovery remains logically possible."
        )

    # ========================================================
    # INSUFFICIENT FUNDS
    # ========================================================

    elif failure == FailureClass.INSUFFICIENT_FUNDS:
        actions.extend(
            [
                make_action(
                    case,
                    RecoveryActionType.DELAYED_RETRY,
                    reason=(
                        "A delayed retry may succeed after the customer's "
                        "available balance changes."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.SEND_REMINDER,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "A reminder can prompt the customer to retry when "
                        "funds are available."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.CREATE_PAYMENT_LINK,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "A payment link lets the customer complete payment "
                        "when convenient."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.WAIT,
                    reason=(
                        "Immediate repeated attempts may be ineffective "
                        "when funds are unavailable."
                    ),
                ),
            ]
        )

        notes.append(
            "Immediate retry is intentionally excluded because funds "
            "availability is unlikely to change instantly."
        )

    # ========================================================
    # EXPIRED INSTRUMENT
    # ========================================================

    elif failure == FailureClass.EXPIRED_INSTRUMENT:
        actions.extend(
            [
                make_action(
                    case,
                    RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "The existing payment instrument has expired and "
                        "must be replaced."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.CREATE_PAYMENT_LINK,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "A payment link can allow the customer to choose "
                        "another supported payment method."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.SEND_REMINDER,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "The customer can be prompted to update the "
                        "payment method."
                    ),
                ),
            ]
        )

        notes.append(
            "Same-instrument retry is excluded because the instrument "
            "is expired."
        )

    # ========================================================
    # BLOCKED INSTRUMENT
    # ========================================================

    elif failure == FailureClass.BLOCKED_INSTRUMENT:
        actions.extend(
            [
                make_action(
                    case,
                    RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "The blocked instrument should not be retried."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.CREATE_PAYMENT_LINK,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "A payment link can let the customer use an "
                        "alternative instrument."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.SEND_REMINDER,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "The customer may need to choose or activate a "
                        "different payment method."
                    ),
                ),
            ]
        )

    # ========================================================
    # INACTIVE INSTRUMENT
    # ========================================================

    elif failure == FailureClass.INACTIVE_INSTRUMENT:
        actions.extend(
            [
                make_action(
                    case,
                    RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "The current instrument is not active for the "
                        "attempted transaction."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.CREATE_PAYMENT_LINK,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "A payment link allows selection of another "
                        "payment option."
                    ),
                ),
            ]
        )

    # ========================================================
    # TRANSACTION LIMIT
    # ========================================================

    elif failure == FailureClass.TRANSACTION_LIMIT:
        actions.extend(
            [
                make_action(
                    case,
                    RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "Another payment method may avoid the current "
                        "transaction limit."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.CREATE_PAYMENT_LINK,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "A payment link can allow the customer to choose "
                        "another available payment method."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.DELAYED_RETRY,
                    reason=(
                        "Some limits may reset after a time window."
                    ),
                ),
            ]
        )

    # ========================================================
    # BANK DECLINE
    # ========================================================

    elif failure == FailureClass.BANK_DECLINE:
        actions.extend(
            [
                make_action(
                    case,
                    RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "A different payment method may avoid a repeated "
                        "issuer decline."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.CREATE_PAYMENT_LINK,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "The customer can retry using another payment "
                        "method."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.DELAYED_RETRY,
                    reason=(
                        "Some issuer declines may be temporary."
                    ),
                ),
            ]
        )

    # ========================================================
    # RISK DECLINE
    # ========================================================

    elif failure == FailureClass.RISK_DECLINE:
        actions.extend(
            [
                make_action(
                    case,
                    RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "A different supported instrument may avoid the "
                        "same risk-control outcome."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.ESCALATE_TO_HUMAN,
                    reason=(
                        "Risk declines may require manual review before "
                        "further recovery attempts."
                    ),
                ),
            ]
        )

        notes.append(
            "Aggressive automatic retries are excluded for risk-control "
            "failures."
        )

    # ========================================================
    # CUSTOMER CANCELLED
    # ========================================================

    elif failure == FailureClass.CUSTOMER_CANCELLED:
        actions.extend(
            [
                make_action(
                    case,
                    RecoveryActionType.CREATE_PAYMENT_LINK,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "The customer can resume payment later using a "
                        "fresh link."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.SEND_REMINDER,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "A lightweight reminder may recover an abandoned "
                        "payment attempt."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.WAIT,
                    reason=(
                        "Waiting avoids unnecessary pressure after a "
                        "customer-initiated cancellation."
                    ),
                ),
            ]
        )

    # ========================================================
    # PAYMENT TIMEOUT
    # ========================================================

    elif failure == FailureClass.PAYMENT_TIMEOUT:
        actions.extend(
            [
                make_action(
                    case,
                    RecoveryActionType.IMMEDIATE_RETRY,
                    reason=(
                        "A timeout may be transient and a fresh payment "
                        "attempt may succeed."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.CREATE_PAYMENT_LINK,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "A fresh payment link provides a new valid "
                        "payment session."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.DELAYED_RETRY,
                    reason=(
                        "If the timeout reflects temporary infrastructure "
                        "issues, a delayed retry may succeed."
                    ),
                ),
            ]
        )

    # ========================================================
    # NETWORK / GATEWAY
    # ========================================================

    elif failure == FailureClass.NETWORK_OR_GATEWAY:
        actions.extend(
            [
                make_action(
                    case,
                    RecoveryActionType.IMMEDIATE_RETRY,
                    reason=(
                        "Technical payment failures can be transient."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.DELAYED_RETRY,
                    reason=(
                        "A delayed retry may avoid a temporary outage."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.CREATE_PAYMENT_LINK,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "A fresh payment session may succeed after the "
                        "technical failure clears."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.WAIT,
                    reason=(
                        "Waiting is valid if an active degradation is "
                        "suspected."
                    ),
                ),
            ]
        )

    # ========================================================
    # MANDATE FAILURE
    # ========================================================

    elif failure == FailureClass.MANDATE_FAILURE:
        actions.extend(
            [
                make_action(
                    case,
                    RecoveryActionType.DELAYED_RETRY,
                    reason=(
                        "A temporary mandate-processing failure may "
                        "recover later."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "A new mandate or payment method may be required "
                        "if retries fail."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.CREATE_PAYMENT_LINK,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "The outstanding amount can still be collected "
                        "outside the recurring mandate."
                    ),
                ),
            ]
        )

    # ========================================================
    # MANDATE CANCELLED
    # ========================================================

    elif failure == FailureClass.MANDATE_CANCELLED:
        actions.extend(
            [
                make_action(
                    case,
                    RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "The cancelled mandate cannot be used for future "
                        "automatic collection."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.CREATE_PAYMENT_LINK,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "The outstanding amount can be collected through "
                        "another payment method."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.SEND_REMINDER,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "The customer should be informed that payment "
                        "authorization needs attention."
                    ),
                ),
            ]
        )

    # ========================================================
    # BUSINESS CONFIGURATION
    # ========================================================

    elif failure == FailureClass.BUSINESS_CONFIGURATION:
        actions.extend(
            [
                make_action(
                    case,
                    RecoveryActionType.ESCALATE_TO_HUMAN,
                    reason=(
                        "The issue requires merchant or integration "
                        "correction rather than customer recovery."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.STOP,
                    reason=(
                        "Customer-facing recovery should stop until the "
                        "merchant-side issue is corrected."
                    ),
                ),
            ]
        )

        notes.append(
            "Customer-facing recovery actions are intentionally excluded."
        )

    # ========================================================
    # CHECKOUT ABANDONMENT
    # ========================================================

    elif failure == FailureClass.CHECKOUT_ABANDONMENT:
        actions.extend(
            [
                make_action(
                    case,
                    RecoveryActionType.CREATE_PAYMENT_LINK,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "A fresh payment link can bring the customer back "
                        "to complete the purchase."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.SEND_REMINDER,
                    channel=CommunicationChannel.SMS,
                    reason=(
                        "A checkout reminder may recover purchase intent."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.WAIT,
                    reason=(
                        "Waiting remains a low-friction alternative for "
                        "recent abandonment."
                    ),
                ),
            ]
        )

    # ========================================================
    # OVERDUE RECEIVABLE
    # ========================================================

    elif failure == FailureClass.OVERDUE_RECEIVABLE:
        actions.extend(
            [
                make_action(
                    case,
                    RecoveryActionType.SEND_REMINDER,
                    channel=CommunicationChannel.EMAIL,
                    reason=(
                        "A structured payment reminder is appropriate "
                        "for an overdue receivable."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.REQUEST_PROMISE_TO_PAY,
                    channel=CommunicationChannel.EMAIL,
                    reason=(
                        "A promise-to-pay can establish a concrete future "
                        "payment commitment."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.OFFER_PARTIAL_PAYMENT,
                    channel=CommunicationChannel.EMAIL,
                    amount=case.amount_at_risk,
                    reason=(
                        "Partial collection may recover some revenue when "
                        "full immediate payment is difficult."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.ESCALATE_TO_HUMAN,
                    reason=(
                        "High-value or prolonged overdue receivables may "
                        "require manual intervention."
                    ),
                ),
            ]
        )

    # ========================================================
    # UNKNOWN
    # ========================================================

    else:
        actions.extend(
            [
                make_action(
                    case,
                    RecoveryActionType.ESCALATE_TO_HUMAN,
                    reason=(
                        "The failure is not sufficiently understood for "
                        "safe automated recovery."
                    ),
                ),
                make_action(
                    case,
                    RecoveryActionType.STOP,
                    reason=(
                        "Automatic recovery is paused until the failure "
                        "can be classified."
                    ),
                ),
            ]
        )

        notes.append(
            "Unknown diagnoses are intentionally handled conservatively."
        )

    return CandidateActionSet(
        case_id=case.id,
        failure_class=failure,
        actions=actions,
        generation_notes=notes,
    )