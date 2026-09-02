from typing import Any

from pydantic import Field

from .payment_attempt import PaymentAttempt


class Payment(PaymentAttempt):
    """Compatibility form retaining integration/persistence payload data.

    New business logic should depend on :class:`PaymentAttempt`. Raw provider
    payloads remain at the existing integration and persistence boundaries.
    """

    raw_payload: dict[str, Any] = Field(
        default_factory=dict
    )
