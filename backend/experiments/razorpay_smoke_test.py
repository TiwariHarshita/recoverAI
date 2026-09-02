from __future__ import annotations

import argparse
import json
from decimal import Decimal

from app.integrations.razorpay import (
    PaymentLinkRequest,
    RazorpayClient,
    RazorpaySettings,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="RecoverAI Razorpay Test Mode smoke test")
    parser.add_argument("--payment-id")
    parser.add_argument("--subscription-id")
    parser.add_argument("--invoice-id")
    parser.add_argument(
        "--create-link-amount",
        type=Decimal,
        help="Optional INR amount. Creates one Test Mode Payment Link.",
    )
    parser.add_argument("--reference-id", default="recoverai_layer19_smoke")
    args = parser.parse_args()

    settings = RazorpaySettings.from_env()
    print(f"Razorpay mode: test ({settings.key_id[:12]}...)")
    print(f"Gateway: {settings.base_url}")

    with RazorpayClient(settings) as client:
        collection = client.validate_credentials()
        print(f"Credentials OK. Payments returned: {collection.get('count', len(collection.get('items', [])))}")

        if args.payment_id:
            payment = client.fetch_payment(args.payment_id)
            print("Payment:", json.dumps({
                "id": payment.get("id"),
                "status": payment.get("status"),
                "amount": payment.get("amount"),
                "currency": payment.get("currency"),
                "method": payment.get("method"),
            }, indent=2))

        if args.subscription_id:
            subscription = client.fetch_subscription(args.subscription_id)
            print("Subscription:", json.dumps({
                "id": subscription.get("id"),
                "status": subscription.get("status"),
                "customer_id": subscription.get("customer_id"),
            }, indent=2))

        if args.invoice_id:
            invoice = client.fetch_invoice(args.invoice_id)
            print("Invoice:", json.dumps({
                "id": invoice.get("id"),
                "status": invoice.get("status"),
                "amount": invoice.get("amount"),
                "amount_paid": invoice.get("amount_paid"),
                "currency": invoice.get("currency"),
            }, indent=2))

        if args.create_link_amount is not None:
            link = client.create_payment_link(
                PaymentLinkRequest.from_inr(
                    args.create_link_amount,
                    description="RecoverAI Layer 19 Test Mode smoke link",
                    reference_id=args.reference_id,
                    notes={"source": "recoverai_layer19"},
                )
            )
            print("Payment Link:", json.dumps({
                "id": link.get("id"),
                "status": link.get("status"),
                "amount": link.get("amount"),
                "currency": link.get("currency"),
                "short_url": link.get("short_url"),
            }, indent=2))


if __name__ == "__main__":
    main()
