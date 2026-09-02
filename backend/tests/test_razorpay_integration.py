from __future__ import annotations

import base64
from decimal import Decimal

import httpx
import pytest

from app.integrations.razorpay import (
    PaymentLinkCustomer,
    PaymentLinkRequest,
    RazorpayAPIError,
    RazorpayAuthenticationError,
    RazorpayClient,
    RazorpayConfigurationError,
    RazorpayNotFoundError,
    RazorpaySettings,
    RazorpayTransportError,
)


SETTINGS = RazorpaySettings(
    key_id="rzp_test_1234567890",
    key_secret="super-secret-test-key",
)


def test_settings_require_test_mode_key() -> None:
    with pytest.raises(RazorpayConfigurationError, match="Test Mode"):
        RazorpaySettings(key_id="rzp_live_123", key_secret="secret")


def test_settings_from_env_and_repr_redacts_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_env123")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "never-print-me")
    settings = RazorpaySettings.from_env()
    assert settings.key_id == "rzp_test_env123"
    assert "never-print-me" not in repr(settings)
    assert "REDACTED" in repr(settings)


def test_fetch_payment_uses_basic_auth_and_correct_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/payments/pay_123"
        expected = base64.b64encode(
            b"rzp_test_1234567890:super-secret-test-key"
        ).decode()
        assert request.headers["Authorization"] == f"Basic {expected}"
        return httpx.Response(200, json={"id": "pay_123", "status": "failed"})

    with RazorpayClient(SETTINGS, transport=httpx.MockTransport(handler)) as client:
        assert client.fetch_payment("pay_123")["status"] == "failed"


def test_validate_credentials_lists_one_payment() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/payments"
        assert request.url.params["count"] == "1"
        assert request.url.params["skip"] == "0"
        return httpx.Response(200, json={"entity": "collection", "count": 0, "items": []})

    with RazorpayClient(SETTINGS, transport=httpx.MockTransport(handler)) as client:
        result = client.validate_credentials()
    assert result["count"] == 0


def test_list_payments_supports_pagination_and_time_window() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert dict(request.url.params) == {
            "count": "25",
            "skip": "50",
            "from": "100",
            "to": "200",
        }
        return httpx.Response(200, json={"count": 0, "items": []})

    with RazorpayClient(SETTINGS, transport=httpx.MockTransport(handler)) as client:
        client.list_payments(count=25, skip=50, from_timestamp=100, to_timestamp=200)


def test_fetch_subscription_and_invoice() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        entity_id = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, json={"id": entity_id})

    with RazorpayClient(SETTINGS, transport=httpx.MockTransport(handler)) as client:
        assert client.fetch_subscription("sub_123")["id"] == "sub_123"
        assert client.fetch_invoice("inv_123")["id"] == "inv_123"
    assert seen == ["/v1/subscriptions/sub_123", "/v1/invoices/inv_123"]


def test_payment_link_request_converts_inr_to_paise_and_builds_payload() -> None:
    request = PaymentLinkRequest.from_inr(
        Decimal("2708.06"),
        description="Recover payment for case rc_123",
        reference_id="rc_123",
        customer=PaymentLinkCustomer(
            name="Test Customer",
            email="test@example.com",
            contact="+919999999999",
        ),
        notify_sms=True,
        reminder_enable=True,
        notes={"recovery_case_id": "rc_123"},
    )
    payload = request.to_payload()
    assert payload["amount"] == 270806
    assert payload["currency"] == "INR"
    assert payload["reference_id"] == "rc_123"
    assert payload["notify"] == {"sms": True, "email": False}
    assert payload["customer"]["email"] == "test@example.com"
    assert payload["notes"]["recovery_case_id"] == "rc_123"


def test_payment_link_rejects_sub_paise_amount() -> None:
    with pytest.raises(ValueError, match="paise"):
        PaymentLinkRequest.from_inr("10.001")


def test_create_fetch_and_cancel_payment_link() -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path == "/v1/payment_links":
            body = __import__("json").loads(request.content)
            assert body["amount"] == 10000
            return httpx.Response(200, json={"id": "plink_123", "status": "created"})
        if request.method == "GET":
            return httpx.Response(200, json={"id": "plink_123", "status": "created"})
        return httpx.Response(200, json={"id": "plink_123", "status": "cancelled"})

    with RazorpayClient(SETTINGS, transport=httpx.MockTransport(handler)) as client:
        created = client.create_payment_link(PaymentLinkRequest.from_inr("100.00"))
        fetched = client.fetch_payment_link("plink_123")
        cancelled = client.cancel_payment_link("plink_123")
    assert created["id"] == fetched["id"] == cancelled["id"] == "plink_123"
    assert calls == [
        ("POST", "/v1/payment_links"),
        ("GET", "/v1/payment_links/plink_123"),
        ("POST", "/v1/payment_links/plink_123/cancel"),
    ]


def test_authentication_error_preserves_provider_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={
            "error": {
                "code": "BAD_REQUEST_ERROR",
                "description": "Authentication failed",
                "source": "business",
                "step": "payment_authentication",
                "reason": "invalid_key",
            }
        })

    with RazorpayClient(SETTINGS, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RazorpayAuthenticationError) as exc_info:
            client.validate_credentials()
    assert exc_info.value.status_code == 401
    assert exc_info.value.error_code == "BAD_REQUEST_ERROR"
    assert exc_info.value.reason == "invalid_key"


def test_not_found_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"description": "Payment not found"}})

    with RazorpayClient(SETTINGS, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RazorpayNotFoundError, match="Payment not found"):
            client.fetch_payment("pay_missing")


def test_server_error_and_non_json_success_fail_closed() -> None:
    def server_error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="oops")

    with RazorpayClient(SETTINGS, transport=httpx.MockTransport(server_error)) as client:
        with pytest.raises(RazorpayAPIError) as exc_info:
            client.validate_credentials()
    assert exc_info.value.status_code == 500

    def bad_success(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json")

    with RazorpayClient(SETTINGS, transport=httpx.MockTransport(bad_success)) as client:
        with pytest.raises(RazorpayAPIError, match="non-JSON"):
            client.validate_credentials()


def test_timeout_is_wrapped_as_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    with RazorpayClient(SETTINGS, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RazorpayTransportError, match="timed out"):
            client.validate_credentials()


def test_entity_id_validation_prevents_invalid_paths() -> None:
    with RazorpayClient(
        SETTINGS,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    ) as client:
        with pytest.raises(ValueError):
            client.fetch_payment("../../etc/passwd")
        with pytest.raises(ValueError, match="pay_"):
            client.fetch_payment("sub_123")
