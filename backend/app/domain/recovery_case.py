from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .enums import (
    CaseType,
    FailureClass,
    PaymentMethod,
    RecoveryCaseStatus,
)


class RecoveryCase(BaseModel):
    id: str = Field(
        default_factory=lambda: f"rc_{uuid4().hex}"
    )

    merchant_id: str

    customer_id: str | None = None

    case_type: CaseType

    status: RecoveryCaseStatus = RecoveryCaseStatus.OPEN

    amount_at_risk: Decimal

    currency: str = "INR"

    payment_id: str | None = None

    subscription_id: str | None = None

    invoice_id: str | None = None

    checkout_id: str | None = None

    payment_method: PaymentMethod | None = None

    failure_class: FailureClass = FailureClass.UNKNOWN

    error_code: str | None = None

    error_source: str | None = None

    error_step: str | None = None

    error_reason: str | None = None

    error_description: str | None = None


    attempt_count: int = Field(
        default=0,
        ge=0
    )

    recovery_retry_count: int = Field(
        default=0,
        ge=0
    )

    previous_contacts: int = Field(
        default=0,
        ge=0
    )

    predicted_recovery_probability: float | None = Field(
        default=None,
        ge=0,
        le=1
    )

    expected_recovery_value: Decimal | None = None

    recovered_amount: Decimal = Decimal("0")

    selected_action_id: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    closed_at: datetime | None = None