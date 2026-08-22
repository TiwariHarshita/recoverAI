from decimal import Decimal

from app.domain.enums import CaseType, PaymentMethod
from app.domain.recovery_case import RecoveryCase
from app.services.candidate_actions import generate_candidate_actions
from app.services.diagnosis import apply_diagnosis


case = RecoveryCase(
    merchant_id="merchant_1",
    customer_id="customer_123",
    case_type=CaseType.PAYMENT_FAILURE,
    amount_at_risk=Decimal("4999"),
    payment_method=PaymentMethod.CARD,
    error_source="customer",
    error_step="payment_authentication",
    error_reason="invalid_otp",
)


print("\n1. NEW RECOVERY CASE")
print(case.model_dump_json(indent=2))


diagnosis = apply_diagnosis(case)


print("\n2. DIAGNOSIS")
print(diagnosis.model_dump_json(indent=2))


candidates = generate_candidate_actions(
    case,
    diagnosis,
)


print("\n3. CANDIDATE ACTIONS")

for action in candidates.actions:
    print(
        f"- {action.action_type.value}: "
        f"{action.reason}"
    )


print("\n4. CASE STATUS")
print(case.status.value)