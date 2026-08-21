from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .enums import (
    ActionStatus,
    CommunicationChannel,
    RecoveryActionType,
)


class RecoveryAction(BaseModel):
    id: str = Field(
        default_factory=lambda: f"act_{uuid4().hex}"
    )

    case_id: str

    action_type: RecoveryActionType

    channel: CommunicationChannel = CommunicationChannel.NONE

    status: ActionStatus = ActionStatus.PROPOSED

    scheduled_for: datetime | None = None

    amount: Decimal | None = None

    predicted_recovery_probability: float | None = Field(
        default=None,
        ge=0,
        le=1
    )

    expected_recovery_value: Decimal | None = None

    reason: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    executed_at: datetime | None = None