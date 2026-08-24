from __future__ import annotations

from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.db.database import get_session_factory
from app.integrations.razorpay_webhooks import (
    RazorpayWebhookConfigurationError,
    RazorpayWebhookPayloadError,
    RazorpayWebhookSignatureError,
)
from app.services.webhook_processor import RazorpayWebhookProcessor


router = APIRouter(tags=["webhooks"])


def get_db_session() -> Iterator[Session]:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.post("/webhooks/razorpay")
async def razorpay_webhook(
    request: Request,
    session: Session = Depends(get_db_session),
) -> dict[str, object]:
    raw_body = await request.body()
    signature = request.headers.get("x-razorpay-signature")
    event_id = request.headers.get("x-razorpay-event-id")

    try:
        result = RazorpayWebhookProcessor(session).process(
            raw_body,
            signature=signature,
            razorpay_event_id=event_id,
        )
    except RazorpayWebhookSignatureError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RazorpayWebhookPayloadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RazorpayWebhookConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "ok": True,
        "event_id": result.event_id,
        "event_type": result.event_type,
        "status": result.status,
        "duplicate": result.duplicate,
        "case_id": result.case_id,
    }
