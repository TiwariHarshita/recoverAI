from datetime import time
from decimal import Decimal

from pydantic import BaseModel, Field

from .enums import (
    CommunicationChannel,
    RecoveryActionType,
)


class MerchantPolicy(BaseModel):
    merchant_id: str

    max_contacts_per_case: int = Field(
        default=3,
        ge=0
    )

    max_payment_retries: int = Field(
        default=2,
        ge=0
    )

    max_recovery_window_days: int = Field(
        default=14,
        ge=1
    )

    human_approval_threshold: Decimal = Decimal("50000")

    allow_partial_payments: bool = True

    allow_voice_calls: bool = False

    quiet_hours_start: time = time(21, 0)

    quiet_hours_end: time = time(8, 0)

    allowed_channels: set[CommunicationChannel] = Field(
        default_factory=lambda: {
            CommunicationChannel.EMAIL,
            CommunicationChannel.SMS,
            CommunicationChannel.WHATSAPP,
        }
    )

    allowed_actions: set[RecoveryActionType] = Field(
        default_factory=lambda: set(RecoveryActionType)
    )