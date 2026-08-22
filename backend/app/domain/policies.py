from datetime import time
from decimal import Decimal

from pydantic import BaseModel, Field

from .enums import (
    CommunicationChannel,
    RecoveryActionType,
)


class MerchantPolicy(BaseModel):
    merchant_id: str

    # Maximum customer contacts for one recovery case inside
    # the rolling contact window below.
    max_contacts_per_case: int = Field(
        default=3,
        ge=0,
    )

    # Contact limit is evaluated over this rolling window.
    contact_window_days: int = Field(
        default=7,
        ge=1,
    )

    # Maximum retries initiated specifically by RecoverAI.
    # Compared against RecoveryCase.recovery_retry_count.
    max_payment_retries: int = Field(
        default=2,
        ge=0,
    )

    # Default automated recovery window.
    # Consumer merchants use 7 days by default.
    # B2B merchants can override this later, e.g. 45 days.
    max_recovery_window_days: int = Field(
        default=7,
        ge=1,
    )

    # Same currency unit as RecoveryCase.amount_at_risk.
    human_approval_threshold: Decimal = Field(
        default=Decimal("25000"),
        ge=0,
    )

    allow_partial_payments: bool = True

    allow_voice_calls: bool = False

    quiet_hours_start: time = time(21, 0)

    quiet_hours_end: time = time(8, 0)

    timezone: str = "Asia/Kolkata"

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