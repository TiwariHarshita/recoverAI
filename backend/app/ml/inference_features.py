from __future__ import annotations

from datetime import datetime
from enum import Enum
from decimal import Decimal

import pandas as pd

from app.domain.action_scoring import (
    MerchantScoringProfile,
    RecoverySourceContext,
)
from app.domain.actions import RecoveryAction
from app.domain.customer import Customer
from app.domain.diagnosis import DiagnosisResult
from app.domain.enums import PolicyDecision
from app.domain.recovery_case import RecoveryCase
from app.ml.dataset import MODEL_FEATURES


def _safe_nonnegative_hours(
    start: datetime,
    end: datetime,
) -> float:
    return max(
        0.0,
        (end - start).total_seconds() / 3600.0,
    )


def _previous_recovery_success_rate(
    customer: Customer,
) -> float:
    if customer.previous_recovery_attempts == 0:
        return 0.0

    return (
        customer.previous_recovery_successes
        / customer.previous_recovery_attempts
    )


def _categorical_value(
    value: object,
) -> object:
    """Return the canonical serialized value used by historical training."""

    if isinstance(value, Enum):
        return value.value

    return value


def build_action_feature_row(
    *,
    recovery_case: RecoveryCase,
    customer: Customer,
    diagnosis: DiagnosisResult,
    action: RecoveryAction,
    merchant: MerchantScoringProfile,
    source: RecoverySourceContext,
    policy_decision: PolicyDecision,
    eligible_action_count: int,
    selection_time: datetime,
    execute_at: datetime,
) -> dict[str, object]:
    """
    Build one live inference row using the exact frozen historical feature
    contract from app.ml.dataset.

    No post-outcome or simulator-hidden fields are accepted by this function.
    """

    if selection_time.tzinfo is None:
        raise ValueError("selection_time must be timezone-aware.")

    if execute_at.tzinfo is None:
        raise ValueError("execute_at must be timezone-aware.")

    if customer.merchant_id != recovery_case.merchant_id:
        raise ValueError(
            "Customer does not belong to the RecoveryCase merchant."
        )

    if merchant.merchant_id != recovery_case.merchant_id:
        raise ValueError(
            "Merchant scoring profile does not belong to the RecoveryCase merchant."
        )

    if action.case_id != recovery_case.id:
        raise ValueError(
            "RecoveryAction does not belong to the supplied RecoveryCase."
        )

    if eligible_action_count <= 0:
        raise ValueError(
            "eligible_action_count must be greater than zero."
        )

    average_order_value = merchant.average_order_value
    amount_at_risk = recovery_case.amount_at_risk

    amount_ratio = (
        float(amount_at_risk / average_order_value)
        if average_order_value > 0
        else 0.0
    )

    row: dict[str, object] = {
        "merchant_archetype": merchant.archetype,
        "case_type": _categorical_value(recovery_case.case_type),
        "currency": recovery_case.currency,
        "payment_method": _categorical_value(recovery_case.payment_method),
        "bank": source.bank,
        "error_code": recovery_case.error_code,
        "error_source": recovery_case.error_source,
        "error_step": recovery_case.error_step,
        "error_reason": recovery_case.error_reason,
        "failure_class": _categorical_value(diagnosis.failure_class),
        "diagnosis_certainty": _categorical_value(diagnosis.certainty),
        "preferred_payment_method": _categorical_value(customer.preferred_payment_method),
        "preferred_channel": _categorical_value(customer.preferred_channel),
        "language_preference": customer.language_preference,
        "action_type": _categorical_value(action.action_type),
        "channel": _categorical_value(action.channel),
        "policy_decision_at_selection": _categorical_value(policy_decision),
        "merchant_average_order_value": float(average_order_value),
        "amount_to_average_order_ratio": amount_ratio,
        "amount_at_risk": float(amount_at_risk),
        "payment_attempt_number": source.payment_attempt_number,
        "subscription_retry_count": source.subscription_retry_count,
        "invoice_days_overdue": source.invoice_days_overdue,
        "attempt_count": recovery_case.attempt_count,
        "recovery_retry_count": recovery_case.recovery_retry_count,
        "previous_contacts": recovery_case.previous_contacts,
        "case_age_hours": _safe_nonnegative_hours(
            recovery_case.created_at,
            execute_at,
        ),
        "customer_lifetime_value": float(customer.lifetime_value),
        "historical_payment_success_rate": (
            customer.historical_payment_success_rate
        ),
        "successful_payments": customer.successful_payments,
        "failed_payments": customer.failed_payments,
        "previous_recovery_attempts": customer.previous_recovery_attempts,
        "previous_recovery_successes": customer.previous_recovery_successes,
        "previous_recovery_success_rate": (
            _previous_recovery_success_rate(customer)
        ),
        "customer_tenure_days": max(
            0.0,
            (execute_at - customer.created_at).total_seconds() / 86400.0,
        ),
        "action_amount": (
            float(action.amount)
            if action.amount is not None
            else None
        ),
        "eligible_action_count": eligible_action_count,
        "action_delay_hours": _safe_nonnegative_hours(
            selection_time,
            execute_at,
        ),
        "mandate_active": source.mandate_active,
        "temporary_failure": diagnosis.temporary_failure,
        "retry_same_method_reasonable": (
            diagnosis.retry_same_method_reasonable
        ),
        "requires_new_payment_method": (
            diagnosis.requires_new_payment_method
        ),
        "customer_action_required": diagnosis.customer_action_required,
        "merchant_action_required": diagnosis.merchant_action_required,
        "customer_do_not_contact": customer.do_not_contact,
        "was_deferred": (
            policy_decision == PolicyDecision.DEFERRED
        ),
    }

    if set(row) != set(MODEL_FEATURES):
        missing = sorted(set(MODEL_FEATURES) - set(row))
        extra = sorted(set(row) - set(MODEL_FEATURES))
        raise RuntimeError(
            "Live model feature row does not match the frozen ML contract. "
            f"missing={missing}, extra={extra}"
        )

    # Preserve the canonical feature order used by the model layer.
    return {
        column: row[column]
        for column in MODEL_FEATURES
    }


def build_action_feature_frame(
    rows: list[dict[str, object]],
) -> pd.DataFrame:
    if not rows:
        raise ValueError(
            "At least one action feature row is required."
        )

    dataframe = pd.DataFrame(
        rows,
        columns=list(MODEL_FEATURES),
    )

    return dataframe
