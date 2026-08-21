from datetime import datetime, timezone
from decimal import Decimal

from pydantic import BaseModel, Field

from .enums import CommunicationChannel, PaymentMethod


class Customer(BaseModel):
    id: str

    merchant_id: str

    email: str | None = None
    phone: str | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    lifetime_value: Decimal = Decimal("0")

    successful_payments: int = 0
    failed_payments: int = 0

    historical_payment_success_rate: float = Field(
        default=0.0,
        ge=0,
        le=1
    )

    previous_recovery_attempts: int = 0
    previous_recovery_successes: int = 0

    preferred_payment_method: PaymentMethod | None = None

    preferred_channel: CommunicationChannel | None = None

    language_preference: str = "en"

    do_not_contact: bool = False

    timezone: str = "Asia/Kolkata"