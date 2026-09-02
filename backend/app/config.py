from __future__ import annotations

import os
from dataclasses import dataclass


APP_NAME = "RecoverAI"
APP_VERSION = "0.20.0"
DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://recoverai:recoverai@localhost:5433/recoverai"
)
DEFAULT_RAZORPAY_BASE_URL = "https://api.razorpay.com/v1"
DEFAULT_RAZORPAY_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class ApplicationSettings:
    """Environment-backed infrastructure settings used by the current app."""

    database_url: str
    razorpay_key_id: str
    razorpay_key_secret: str
    razorpay_base_url: str
    razorpay_timeout_seconds: float
    razorpay_webhook_secret: str

    @classmethod
    def from_env(cls) -> "ApplicationSettings":
        timeout_raw = os.getenv(
            "RAZORPAY_TIMEOUT_SECONDS",
            str(DEFAULT_RAZORPAY_TIMEOUT_SECONDS),
        )
        try:
            timeout = float(timeout_raw)
        except ValueError as exc:
            raise ValueError("RAZORPAY_TIMEOUT_SECONDS must be numeric") from exc

        return cls(
            database_url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
            razorpay_key_id=os.getenv("RAZORPAY_KEY_ID", ""),
            razorpay_key_secret=os.getenv("RAZORPAY_KEY_SECRET", ""),
            razorpay_base_url=os.getenv(
                "RAZORPAY_BASE_URL",
                DEFAULT_RAZORPAY_BASE_URL,
            ),
            razorpay_timeout_seconds=timeout,
            razorpay_webhook_secret=os.getenv("RAZORPAY_WEBHOOK_SECRET", ""),
        )
