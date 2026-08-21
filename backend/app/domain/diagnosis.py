from datetime import datetime, timezone

from pydantic import BaseModel, Field

from .enums import DiagnosisCertainty, FailureClass


class DiagnosisEvidence(BaseModel):
    """
    A specific piece of evidence used by the diagnosis engine.

    Example:
        field="error_reason"
        value="invalid_otp"
        note="Matched a known authentication failure reason."
    """

    field: str

    value: str

    note: str


class DiagnosisResult(BaseModel):
    """
    Normalized diagnosis produced from a RecoveryCase.

    Important:
    - This is NOT an ML prediction.
    - There is deliberately no numeric "confidence".
    - Recovery probability will be predicted later by our ML model.
    """

    failure_class: FailureClass

    certainty: DiagnosisCertainty

    summary: str

    temporary_failure: bool

    retry_same_method_reasonable: bool

    requires_new_payment_method: bool

    customer_action_required: bool

    merchant_action_required: bool

    evidence: list[DiagnosisEvidence] = Field(
        default_factory=list
    )

    diagnosed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )