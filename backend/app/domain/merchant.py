from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from pydantic import AliasChoices, BaseModel, Field, field_validator


class Merchant(BaseModel):
    """Provider-independent merchant identity and business profile."""

    id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("id", "merchant_id"),
    )
    name: str | None = None
    archetype: str = "unknown"
    average_order_value: Decimal = Field(gt=0)
    currency: str = "INR"
    timezone: str = "Asia/Kolkata"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def merchant_id(self) -> str:
        """Compatibility name used by existing scoring call sites."""

        return self.id

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        normalized = value.strip().upper()
        if len(normalized) != 3:
            raise ValueError("currency must be a 3-letter code")
        return normalized

    @field_validator("timezone")
    @classmethod
    def require_timezone(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("timezone cannot be empty")
        return normalized
