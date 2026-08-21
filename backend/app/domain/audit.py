from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .enums import (
    AuditActor,
    AuditEventType,
)


class AuditEvent(BaseModel):
    id: str = Field(
        default_factory=lambda: f"audit_{uuid4().hex}"
    )

    case_id: str

    event_type: AuditEventType

    actor: AuditActor

    message: str

    data: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )