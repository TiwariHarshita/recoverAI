from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from .enums import InvoiceStatus


class Invoice(BaseModel):
    id: str

    merchant_id: str

    customer_id: str

    amount_due: Decimal

    amount_paid: Decimal = Decimal("0")

    currency: str = "INR"

    status: InvoiceStatus

    issued_at: datetime

    due_at: datetime

    days_overdue: int = Field(
        default=0,
        ge=0
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    raw_payload: dict[str, Any] = Field(
        default_factory=dict
    )