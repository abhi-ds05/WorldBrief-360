"""
Webhook endpoints for external service integrations.
"""
import hashlib
import hmac
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Header, BackgroundTasks, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.database import get_db
from app.core.config import settings
from app.core.security import get_current_user, require_admin_role
from app.db.models import User, WebhookEvent, Webhook
from app.schemas import WebhookCreate, WebhookResponse
from app.services.integrations.news_api_client import process_news_webhook
from app.services.integrations.maps_client import process_geolocation_webhook
from app.services.community.wallet_service import process_payment_webhook

router = APIRouter()


# Webhook secrets (should be stored securely, e.g., in database)
WEBHOOK_SECRETS = {
    "stripe": settings.STRIPE_WEBHOOK_SECRET,
    "newsapi": settings.NEWSAPI_WEBHOOK_SECRET,
    "maps": settings.MAPS_WEBHOOK_SECRET,
    "payment": settings.PAYMENT_WEBHOOK_SECRET,
}


@router.post("/stripe")
async def stripe_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    stripe_signature: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """
    Handle Stripe webhooks for payment processing.
    """
    try:
        # Get raw body
        body = await request.body()
        
        # Verify webhook signature
        if stripe_signature:
            # In production, verify Stripe signature
            # For now, we'll trust the webhook
            pass
        
        # Parse event
        event_data = await request.json()
        
        # Log webhook event
        webhook_event = WebhookEvent(
            source="stripe",
            event_type=event_data.get("type"),
            payload=event_data,
            received_at=datetime.utcnow(),
        )
        db.add(webhook_event)
        db.commit()
        
        # Process event in background
        background_tasks.add_task(
            process_stripe_event,
            event_data,
            db
        )
        
        return {"status": "received"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Webhook error: {str(e)}")


async def process_stripe_event(event_data: Dict[str, Any], db: Session) -> None:
    """
    Process Stripe webhook event.
    """
    event_type = event_data.get("type")
    
    if event_type == "checkout.session.completed":
        # Handle successful payment
        session = event_data.get("data", {}).get("object", {})
        user_id = session.get("metadata", {}).get("user_id")
        amount = session.get("amount_total", 0) / 100  # Convert from cents
        
        if user_id:
            # Add coins to user's wallet
            from app.services.community.wallet_service import add_coins_to_wallet
            await add_coins_to_wallet(
                user_id=int(user_id),
                amount=amount,
                transaction_type="purchase",
                description="Coin purchase via Stripe",
                db=db
            )
    
    elif event_type == "charge.refunded":
        # Handle refund
        charge = event_data.get("data", {}).get("object", {})
        # Process refund logic
        pass
    
    # Update webhook event status
    event = db.query(WebhookEvent).order_by(WebhookEvent.id.desc()).first()
    if event:
        event.processed_at = datetime.utcnow()
        event.status = "processed"
        db.commit()


@router.post("/news")
async def news_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_newsapi_signature: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """
    Handle news API webhooks for new articles.
    """
    try:
        # Get request data
        body = await request.body()
        
        # Verify signature if provided
        if x_newsapi_signature and settings.NEWSAPI_WEBHOOK_SECRET:
            expected_signature = hmac.new(
                settings.NEWSAPI_WEBHOOK_SECRET.encode(),
                body,
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(x_newsapi_signature, expected_signature):
                raise HTTPException(status_code=401, detail="Invalid signature")
        
        # Parse data
        data = await request.json()
        
        # Log webhook event
        webhook_event = WebhookEvent(
            source="newsapi",
            event_type="new_articles",
            payload=data,
            received_at=datetime.utcnow(),
        )
        db.add(webhook_event)
        db.commit()
        
        # Process in background
        background_tasks.add_task(
            process_news_webhook,
            data,
            db
        )
        
        return {"status": "received"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"News webhook error: {str(e)}")


@router.post("/maps")
async def maps_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """
    Handle maps/geolocation service webhooks.
    """
    try:
        data = await request.json()
        
        # Log webhook event
        webhook_event = WebhookEvent(
            source="maps",
            event_type=data.get("event_type", "unknown"),
            payload=data,
            received_at=datetime.utcnow(),
        )
        db.add(webhook_event)
        db.commit()
        
        # Process in background
        background_tasks.add_task(
            process_geolocation_webhook,
            data,
            db
        )
        
        return {"status": "received"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Maps webhook error: {str(e)}")


@router.post("/payment/{provider}")
async def payment_webhook(
    provider: str,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """
    Handle payment provider webhooks.
    """
    valid_providers = ["paypal", "razorpay", "coinbase"]
    
    if provider not in valid_providers:
        raise HTTPException(status_code=400, detail=f"Invalid provider. Must be one of: {', '.join(valid_providers)}")
    
    try:
        data = await request.json()
        
        # Log webhook event
        webhook_event = WebhookEvent(
            source=provider,
            event_type=data.get("event_type", "payment"),
            payload=data,
            received_at=datetime.utcnow(),
        )
        db.add(webhook_event)
        db.commit()
        
        # Process in background
        background_tasks.add_task(
            process_payment_webhook,
            provider,
            data,
            db
        )
        
        return {"status": "received"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Payment webhook error: {str(e)}")


@router.post("/custom")
async def custom_webhook(
    webhook_data: WebhookCreate,
    x_webhook_secret: Optional[str] = Header(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> WebhookResponse:
    """
    Register a custom webhook endpoint.
    """
    # Verify webhook secret if provided
    if webhook_data.secret:
        if not x_webhook_secret or x_webhook_secret != webhook_data.secret:
            raise HTTPException(status_code=401, detail="Invalid webhook secret")
    
    # Create webhook record
    webhook = Webhook(
        user_id=current_user.id,
        name=webhook_data.name,
        url=webhook_data.url,
        secret=webhook_data.secret,
        events=webhook_data.events,
        is_active=True,
    )
    
    db.add(webhook)
    db.commit()
    db.refresh(webhook)
    
    return WebhookResponse.from_orm(webhook)


@router.get("/events")
async def list_webhook_events(
    source: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin_role),
) -> Dict[str, Any]:
    """
    List webhook events (admin only).
    """
    query = db.query(WebhookEvent)
    
    if source:
        query = query.filter(WebhookEvent.source == source)
    
    if event_type:
        query = query.filter(WebhookEvent.event_type == event_type)
    
    total = query.count()
    events = query.order_by(WebhookEvent.received_at.desc())\
                  .offset(offset)\
                  .limit(limit)\
                  .all()
    
    return {
        "events": [
            {
                "id": event.id,
                "source": event.source,
                "event_type": event.event_type,
                "received_at": event.received_at.isoformat(),
                "processed_at": event.processed_at.isoformat() if event.processed_at else None,
                "status": event.status,
                "payload_summary": str(event.payload)[:100] + "..." if len(str(event.payload)) > 100 else str(event.payload),
            }
            for event in events
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/events/{event_id}")
async def get_webhook_event(
    event_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin_role),
) -> Dict[str, Any]:
    """
    Get details of a specific webhook event (admin only).
    """
    event = db.query(WebhookEvent).filter(WebhookEvent.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Webhook event not found")
    
    return {
        "id": event.id,
        "source": event.source,
        "event_type": event.event_type,
        "received_at": event.received_at.isoformat(),
        "processed_at": event.processed_at.isoformat() if event.processed_at else None,
        "status": event.status,
        "payload": event.payload,
        "error_message": event.error_message,
    }


@router.post("/events/{event_id}/retry")
async def retry_webhook_event(
    event_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin_role),
) -> Dict[str, str]:
    """
    Retry processing a failed webhook event (admin only).
    """
    event = db.query(WebhookEvent).filter(WebhookEvent.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Webhook event not found")
    
    if event.status == "processed":
        raise HTTPException(status_code=400, detail="Event already processed")
    
    # Reset event status
    event.status = "pending"
    event.retry_count = (event.retry_count or 0) + 1
    db.commit()
    
    # Retry in background based on event source
    background_tasks.add_task(
        retry_webhook_processing,
        event,
        db
    )
    
    return {"message": "Webhook event retry queued"}


async def retry_webhook_processing(event: WebhookEvent, db: Session) -> None:
    """
    Retry processing a webhook event.
    """
    try:
        if event.source == "stripe":
            await process_stripe_event(event.payload, db)
        elif event.source == "newsapi":
            await process_news_webhook(event.payload, db)
        elif event.source == "maps":
            await process_geolocation_webhook(event.payload, db)
        elif event.source in ["paypal", "razorpay", "coinbase"]:
            await process_payment_webhook(event.source, event.payload, db)
        
        event.status = "processed"
        event.processed_at = datetime.utcnow()
        event.error_message = None
        
    except Exception as e:
        event.status = "failed"
        event.error_message = str(e)
    
    db.commit()


@router.get("/stats")
async def get_webhook_stats(
    days: int = Query(7, ge=1, le=30),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin_role),
) -> Dict[str, Any]:
    """
    Get webhook statistics (admin only).
    """
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Count by source
    source_stats = db.query(
        WebhookEvent.source,
        func.count(WebhookEvent.id).label("total"),
        func.count(WebhookEvent.id).filter(WebhookEvent.status == "processed").label("processed"),
        func.count(WebhookEvent.id).filter(WebhookEvent.status == "failed").label("failed"),
    ).filter(
        WebhookEvent.received_at >= start_date
    ).group_by(WebhookEvent.source).all()
    
    # Count by event type
    event_stats = db.query(
        WebhookEvent.event_type,
        func.count(WebhookEvent.id).label("count"),
    ).filter(
        WebhookEvent.received_at >= start_date
    ).group_by(WebhookEvent.event_type).order_by(func.count(WebhookEvent.id).desc()).limit(10).all()
    
    # Daily statistics
    daily_stats = []
    for i in range(days):
        date = datetime.utcnow() - timedelta(days=i)
        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        day_count = db.query(func.count(WebhookEvent.id)).filter(
            WebhookEvent.received_at >= day_start,
            WebhookEvent.received_at < day_end,
        ).scalar() or 0
        
        day_processed = db.query(func.count(WebhookEvent.id)).filter(
            WebhookEvent.received_at >= day_start,
            WebhookEvent.received_at < day_end,
            WebhookEvent.status == "processed",
        ).scalar() or 0
        
        daily_stats.append({
            "date": day_start.date().isoformat(),
            "total": day_count,
            "processed": day_processed,
            "success_rate": round(day_processed / day_count * 100, 2) if day_count > 0 else 0,
        })
    
    return {
        "period_days": days,
        "source_statistics": [
            {
                "source": source,
                "total": total,
                "processed": processed,
                "failed": failed,
                "success_rate": round(processed / total * 100, 2) if total > 0 else 0,
            }
            for source, total, processed, failed in source_stats
        ],
        "top_event_types": [
            {"event_type": event_type, "count": count}
            for event_type, count in event_stats
        ],
        "daily_statistics": daily_stats,
        "total_events": sum(stat["total"] for stat in source_stats),
        "overall_success_rate": round(
            sum(stat["processed"] for stat in source_stats) / sum(stat["total"] for stat in source_stats) * 100, 2
        ) if sum(stat["total"] for stat in source_stats) > 0 else 0,
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.post("/test")
async def test_webhook(
    webhook_url: str,
    payload: Dict[str, Any],
    secret: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin_role),
) -> Dict[str, Any]:
    """
    Test a webhook endpoint (admin only).
    """
    import httpx
    
    headers = {}
    if secret:
        # Create signature
        import hashlib
        import hmac
        payload_str = json.dumps(payload)
        signature = hmac.new(
            secret.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        headers["X-Webhook-Signature"] = signature
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                webhook_url,
                json=payload,
                headers=headers,
                timeout=10.0
            )
            
            return {
                "status_code": response.status_code,
                "response": response.text,
                "success": response.status_code < 400,
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@router.post("/verify")
async def verify_webhook_signature(
    payload: Dict[str, Any],
    signature: str,
    secret: str,
) -> Dict[str, bool]:
    """
    Verify a webhook signature.
    """
    try:
        import hashlib
        import hmac
        
        payload_str = json.dumps(payload, sort_keys=True)
        expected_signature = hmac.new(
            secret.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        is_valid = hmac.compare_digest(signature, expected_signature)
        
        return {
            "valid": is_valid,
            "expected_signature": expected_signature,
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Verification error: {str(e)}")