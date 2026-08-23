from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

import httpx


DEFAULT_RAZORPAY_BASE_URL = "https://api.razorpay.com/v1"
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


class RazorpayIntegrationError(RuntimeError):
    """Base error for the RecoverAI Razorpay adapter."""


class RazorpayConfigurationError(RazorpayIntegrationError):
    """Raised when Razorpay credentials/configuration are unsafe or incomplete."""


class RazorpayTransportError(RazorpayIntegrationError):
    """Raised when the Razorpay API could not be reached."""


class RazorpayAPIError(RazorpayIntegrationError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_code: str | None = None,
        field: str | None = None,
        source: str | None = None,
        step: str | None = None,
        reason: str | None = None,
        raw_error: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.field = field
        self.source = source
        self.step = step
        self.reason = reason
        self.raw_error = dict(raw_error or {})


class RazorpayAuthenticationError(RazorpayAPIError):
    """Raised for invalid/unauthorized Razorpay credentials."""


class RazorpayNotFoundError(RazorpayAPIError):
    """Raised when a requested Razorpay entity does not exist."""


@dataclass(frozen=True)
class RazorpaySettings:
    key_id: str
    key_secret: str
    base_url: str = DEFAULT_RAZORPAY_BASE_URL
    timeout_seconds: float = 10.0
    enforce_test_mode: bool = True

    def __post_init__(self) -> None:
        key_id = self.key_id.strip()
        key_secret = self.key_secret.strip()
        base_url = self.base_url.rstrip("/")

        if not key_id:
            raise RazorpayConfigurationError("RAZORPAY_KEY_ID is required")
        if not key_secret:
            raise RazorpayConfigurationError("RAZORPAY_KEY_SECRET is required")
        if self.enforce_test_mode and not key_id.startswith("rzp_test_"):
            raise RazorpayConfigurationError(
                "RecoverAI Layer 19 only permits Razorpay Test Mode keys (rzp_test_...)"
            )
        if not base_url.startswith("https://") and base_url not in {
            "http://localhost",
            "http://127.0.0.1",
        }:
            raise RazorpayConfigurationError("RAZORPAY_BASE_URL must use HTTPS")
        if self.timeout_seconds <= 0:
            raise RazorpayConfigurationError("Razorpay timeout must be greater than zero")

        object.__setattr__(self, "key_id", key_id)
        object.__setattr__(self, "key_secret", key_secret)
        object.__setattr__(self, "base_url", base_url)

    @classmethod
    def from_env(cls) -> "RazorpaySettings":
        timeout_raw = os.getenv("RAZORPAY_TIMEOUT_SECONDS", "10")
        try:
            timeout = float(timeout_raw)
        except ValueError as exc:
            raise RazorpayConfigurationError(
                "RAZORPAY_TIMEOUT_SECONDS must be numeric"
            ) from exc

        return cls(
            key_id=os.getenv("RAZORPAY_KEY_ID", ""),
            key_secret=os.getenv("RAZORPAY_KEY_SECRET", ""),
            base_url=os.getenv("RAZORPAY_BASE_URL", DEFAULT_RAZORPAY_BASE_URL),
            timeout_seconds=timeout,
            enforce_test_mode=True,
        )

    def __repr__(self) -> str:
        return (
            "RazorpaySettings("
            f"key_id={self.key_id!r}, key_secret='***REDACTED***', "
            f"base_url={self.base_url!r}, timeout_seconds={self.timeout_seconds!r}, "
            f"enforce_test_mode={self.enforce_test_mode!r})"
        )


@dataclass(frozen=True)
class PaymentLinkCustomer:
    name: str | None = None
    email: str | None = None
    contact: str | None = None

    def to_payload(self) -> dict[str, str]:
        payload: dict[str, str] = {}
        if self.name:
            payload["name"] = self.name
        if self.email:
            payload["email"] = self.email
        if self.contact:
            payload["contact"] = self.contact
        return payload


@dataclass(frozen=True)
class PaymentLinkRequest:
    amount_subunits: int
    currency: str = "INR"
    description: str | None = None
    reference_id: str | None = None
    customer: PaymentLinkCustomer | None = None
    notify_sms: bool = False
    notify_email: bool = False
    reminder_enable: bool = False
    expire_by: int | None = None
    callback_url: str | None = None
    callback_method: str = "get"
    notes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.amount_subunits <= 0:
            raise ValueError("amount_subunits must be greater than zero")
        if not self.currency or len(self.currency) != 3:
            raise ValueError("currency must be a 3-letter code")
        if self.expire_by is not None and self.expire_by <= 0:
            raise ValueError("expire_by must be a positive Unix timestamp")
        if self.callback_method.lower() not in {"get", "post"}:
            raise ValueError("callback_method must be 'get' or 'post'")
        if len(self.notes) > 15:
            raise ValueError("Razorpay notes support at most 15 key-value pairs")

    @classmethod
    def from_inr(
        cls,
        amount: Decimal | str | int | float,
        **kwargs: Any,
    ) -> "PaymentLinkRequest":
        try:
            major = Decimal(str(amount))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("amount must be a valid INR amount") from exc
        if major <= 0:
            raise ValueError("amount must be greater than zero")
        paise = major * Decimal("100")
        if paise != paise.to_integral_value():
            raise ValueError("INR amount cannot contain fractions smaller than one paise")
        return cls(amount_subunits=int(paise), currency="INR", **kwargs)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "amount": self.amount_subunits,
            "currency": self.currency.upper(),
            "notify": {
                "sms": self.notify_sms,
                "email": self.notify_email,
            },
            "reminder_enable": self.reminder_enable,
        }
        if self.description:
            payload["description"] = self.description
        if self.reference_id:
            payload["reference_id"] = self.reference_id
        if self.customer:
            customer_payload = self.customer.to_payload()
            if customer_payload:
                payload["customer"] = customer_payload
        if self.expire_by is not None:
            payload["expire_by"] = self.expire_by
        if self.callback_url:
            payload["callback_url"] = self.callback_url
            payload["callback_method"] = self.callback_method.lower()
        if self.notes:
            payload["notes"] = dict(self.notes)
        return payload


class RazorpayClient:
    """
    Thin HTTP adapter around Razorpay's v1 REST API.

    This layer deliberately exposes provider payloads as dictionaries. Provider
    payload -> RecoverAI domain normalization belongs to the webhook/normalizer
    layer, not this transport adapter.
    """

    def __init__(
        self,
        settings: RazorpaySettings | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings or RazorpaySettings.from_env()
        self._client = httpx.Client(
            base_url=self.settings.base_url,
            auth=(self.settings.key_id, self.settings.key_secret),
            timeout=self.settings.timeout_seconds,
            headers={
                "Accept": "application/json",
                "User-Agent": "RecoverAI-Hackathon/1.0",
            },
            transport=transport,
        )

    def __enter__(self) -> "RazorpayClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _safe_id(entity_id: str, *, prefix: str | None = None) -> str:
        value = entity_id.strip()
        if not value or not _ID_RE.fullmatch(value):
            raise ValueError("Invalid Razorpay entity id")
        if prefix is not None and not value.startswith(prefix):
            raise ValueError(f"Expected Razorpay id beginning with {prefix!r}")
        return value

    @staticmethod
    def _parse_error(response: httpx.Response) -> tuple[str, dict[str, Any]]:
        try:
            payload = response.json()
        except ValueError:
            return f"Razorpay API returned HTTP {response.status_code}", {}

        raw = payload.get("error") if isinstance(payload, dict) else None
        if not isinstance(raw, dict):
            return f"Razorpay API returned HTTP {response.status_code}", {}

        description = raw.get("description")
        message = (
            str(description)
            if description
            else f"Razorpay API returned HTTP {response.status_code}"
        )
        return message, raw

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, params=params, json=json)
        except httpx.TimeoutException as exc:
            raise RazorpayTransportError("Razorpay API request timed out") from exc
        except httpx.RequestError as exc:
            raise RazorpayTransportError("Could not reach Razorpay API") from exc

        if response.is_error:
            message, raw = self._parse_error(response)
            error_type: type[RazorpayAPIError]
            if response.status_code in {401, 403}:
                error_type = RazorpayAuthenticationError
            elif response.status_code == 404:
                error_type = RazorpayNotFoundError
            else:
                error_type = RazorpayAPIError
            raise error_type(
                message,
                status_code=response.status_code,
                error_code=raw.get("code"),
                field=raw.get("field"),
                source=raw.get("source"),
                step=raw.get("step"),
                reason=raw.get("reason"),
                raw_error=raw,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RazorpayAPIError(
                "Razorpay API returned a non-JSON success response",
                status_code=response.status_code,
            ) from exc
        if not isinstance(payload, dict):
            raise RazorpayAPIError(
                "Razorpay API returned an unexpected JSON payload",
                status_code=response.status_code,
            )
        return payload

    def validate_credentials(self) -> dict[str, Any]:
        return self.list_payments(count=1, skip=0)

    def fetch_payment(self, payment_id: str) -> dict[str, Any]:
        payment_id = self._safe_id(payment_id, prefix="pay_")
        return self._request("GET", f"/payments/{payment_id}")

    def list_payments(
        self,
        *,
        count: int = 10,
        skip: int = 0,
        from_timestamp: int | None = None,
        to_timestamp: int | None = None,
    ) -> dict[str, Any]:
        if not 1 <= count <= 100:
            raise ValueError("count must be between 1 and 100")
        if skip < 0:
            raise ValueError("skip cannot be negative")
        params: dict[str, int] = {"count": count, "skip": skip}
        if from_timestamp is not None:
            params["from"] = from_timestamp
        if to_timestamp is not None:
            params["to"] = to_timestamp
        return self._request("GET", "/payments", params=params)

    def fetch_subscription(self, subscription_id: str) -> dict[str, Any]:
        subscription_id = self._safe_id(subscription_id, prefix="sub_")
        return self._request("GET", f"/subscriptions/{subscription_id}")

    def fetch_invoice(self, invoice_id: str) -> dict[str, Any]:
        invoice_id = self._safe_id(invoice_id, prefix="inv_")
        return self._request("GET", f"/invoices/{invoice_id}")

    def create_payment_link(self, request: PaymentLinkRequest) -> dict[str, Any]:
        return self._request("POST", "/payment_links", json=request.to_payload())

    def fetch_payment_link(self, payment_link_id: str) -> dict[str, Any]:
        payment_link_id = self._safe_id(payment_link_id, prefix="plink_")
        return self._request("GET", f"/payment_links/{payment_link_id}")

    def cancel_payment_link(self, payment_link_id: str) -> dict[str, Any]:
        payment_link_id = self._safe_id(payment_link_id, prefix="plink_")
        return self._request("POST", f"/payment_links/{payment_link_id}/cancel")
