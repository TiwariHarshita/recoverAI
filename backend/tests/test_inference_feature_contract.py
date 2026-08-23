from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd

from app.domain.action_scoring import MerchantScoringProfile, RecoverySourceContext
from app.domain.actions import RecoveryAction
from app.domain.customer import Customer
from app.domain.diagnosis import DiagnosisResult
from app.domain.enums import (
    CaseType,
    CommunicationChannel,
    DiagnosisCertainty,
    FailureClass,
    PolicyDecision,
    RecoveryActionType,
)
from app.domain.recovery_case import RecoveryCase
from app.ml.dataset import prepare_model_features
from app.ml.inference_features import build_action_feature_row


NOW = datetime(2026, 8, 23, 5, 0, tzinfo=timezone.utc)


def _objects():
    case = RecoveryCase(
        id="case_enum_contract",
        merchant_id="merchant_1",
        customer_id="customer_1",
        case_type=CaseType.PAYMENT_FAILURE,
        amount_at_risk=Decimal("1000.00"),
        currency="INR",
        payment_method=None,
        error_code="BAD_REQUEST_ERROR",
        error_source="customer",
        error_step="payment_authentication",
        error_reason="invalid_otp",
        attempt_count=1,
        created_at=NOW,
    )

    customer = Customer(
        id="customer_1",
        merchant_id="merchant_1",
        created_at=datetime(2025, 8, 23, tzinfo=timezone.utc),
        lifetime_value=Decimal("50000.00"),
        successful_payments=12,
        failed_payments=3,
        historical_payment_success_rate=0.80,
        previous_recovery_attempts=4,
        previous_recovery_successes=2,
        preferred_payment_method=None,
        preferred_channel=CommunicationChannel.SMS,
        language_preference="en",
        do_not_contact=False,
    )

    diagnosis = DiagnosisResult(
        failure_class=FailureClass.AUTHENTICATION_FAILURE,
        certainty=DiagnosisCertainty.EXACT,
        summary="Authentication failed.",
        temporary_failure=True,
        retry_same_method_reasonable=True,
        requires_new_payment_method=False,
        customer_action_required=True,
        merchant_action_required=False,
    )

    action = RecoveryAction(
        id="action_link",
        case_id=case.id,
        action_type=RecoveryActionType.CREATE_PAYMENT_LINK,
        channel=CommunicationChannel.SMS,
    )

    merchant = MerchantScoringProfile(
        merchant_id="merchant_1",
        archetype="ecommerce",
        average_order_value=Decimal("800.00"),
    )

    source = RecoverySourceContext(
        bank="HDFC",
        payment_attempt_number=1,
    )

    return case, customer, diagnosis, action, merchant, source


def test_live_feature_row_uses_historical_enum_values():
    case, customer, diagnosis, action, merchant, source = _objects()

    row = build_action_feature_row(
        recovery_case=case,
        customer=customer,
        diagnosis=diagnosis,
        action=action,
        merchant=merchant,
        source=source,
        policy_decision=PolicyDecision.ALLOWED,
        eligible_action_count=1,
        selection_time=NOW,
        execute_at=NOW,
    )

    assert row["case_type"] == "payment_failure"
    assert row["failure_class"] == "authentication_failure"
    assert row["diagnosis_certainty"] == "exact"
    assert row["preferred_channel"] == "sms"
    assert row["action_type"] == "create_payment_link"
    assert row["channel"] == "sms"
    assert row["policy_decision_at_selection"] == "allowed"


def test_shared_preprocessor_normalizes_domain_enums_to_values():
    case, customer, diagnosis, action, merchant, source = _objects()

    row = build_action_feature_row(
        recovery_case=case,
        customer=customer,
        diagnosis=diagnosis,
        action=action,
        merchant=merchant,
        source=source,
        policy_decision=PolicyDecision.ALLOWED,
        eligible_action_count=1,
        selection_time=NOW,
        execute_at=NOW,
    )

    # Simulate another serving caller passing Enum objects directly.
    row["case_type"] = CaseType.PAYMENT_FAILURE
    row["failure_class"] = FailureClass.AUTHENTICATION_FAILURE
    row["diagnosis_certainty"] = DiagnosisCertainty.EXACT
    row["preferred_channel"] = CommunicationChannel.SMS
    row["action_type"] = RecoveryActionType.CREATE_PAYMENT_LINK
    row["channel"] = CommunicationChannel.SMS
    row["policy_decision_at_selection"] = PolicyDecision.ALLOWED

    prepared = prepare_model_features(pd.DataFrame([row]))
    record = prepared.iloc[0]

    assert record["case_type"] == "payment_failure"
    assert record["failure_class"] == "authentication_failure"
    assert record["diagnosis_certainty"] == "exact"
    assert record["preferred_channel"] == "sms"
    assert record["action_type"] == "create_payment_link"
    assert record["channel"] == "sms"
    assert record["policy_decision_at_selection"] == "allowed"
