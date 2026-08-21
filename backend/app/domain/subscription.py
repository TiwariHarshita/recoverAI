from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from .enums import PaymentMethod, SubscriptionStatus


class Subscription(BaseModel):
    id: str

    merchant_id: str

    customer_id: str

    amount: Decimal

    currency: str = "INR"

    status: SubscriptionStatus

    preferred_payment_method: PaymentMethod | None = None

    retry_count: int = Field(
        default=0,
        ge=0
    )

    mandate_active: bool = True

    current_period_start: datetime | None = None
    current_period_end: datetime | None = None

    next_charge_at: datetime | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    raw_payload: dict[str, Any] = Field(
        default_factory=dict
    )