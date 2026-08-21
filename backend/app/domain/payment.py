from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from .enums import PaymentMethod, PaymentStatus


class Payment(BaseModel):
    id: str

    merchant_id: str

    customer_id: str | None = None

    order_id: str | None = None

    amount: Decimal

    currency: str = "INR"

    status: PaymentStatus

    method: PaymentMethod = PaymentMethod.UNKNOWN

    bank: str | None = None

    attempt_number: int = Field(
        default=1,
        ge=1
    )

    error_code: str | None = None

    error_source: str | None = None

    error_step: str | None = None

    error_reason: str | None = None

    error_description: str | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    raw_payload: dict[str, Any] = Field(
        default_factory=dict
    )