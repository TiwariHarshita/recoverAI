from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from pydantic import BaseModel, Field, field_validator

from .enums import PaymentMethod, PaymentStatus


class PaymentAttempt(BaseModel):
    """Provider-independent facts for one attempt to collect a payment."""

    id: str = Field(min_length=1)
    merchant_id: str = Field(min_length=1)
    customer_id: str | None = None
    order_id: str | None = None
    amount: Decimal = Field(ge=0)
    currency: str = "INR"
    status: PaymentStatus
    method: PaymentMethod = PaymentMethod.UNKNOWN
    bank: str | None = None
    attempt_number: int = Field(default=1, ge=1)
    error_code: str | None = None
    error_source: str | None = None
    error_step: str | None = None
    error_reason: str | None = None
    error_description: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        normalized = value.strip().upper()
        if len(normalized) != 3:
            raise ValueError("currency must be a 3-letter code")
        return normalized
