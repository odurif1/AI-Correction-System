"""Subscription management API endpoints."""

from fastapi import APIRouter, HTTPException, Request, Header, Depends
from sqlalchemy.orm import Session
import stripe
from loguru import logger
import os

from db.database import get_db
from db.models import User
from api.auth import get_current_user
from billing.stripe_client import StripeClient
from billing.webhook_handler import (
    handle_checkout_completed,
    handle_subscription_updated,
    handle_subscription_deleted,
    handle_payment_succeeded,
    handle_payment_failed,
)

router = APIRouter(prefix="/subscription", tags=["subscription"])

STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(..., alias="Stripe-Signature"),
    db: Session = Depends(get_db)
):
    """Handle Stripe webhook events."""
    payload = await request.body()

    # Verify signature
    try:
        event = stripe.Webhook.construct_event(
            payload, stripe_signature, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Route event to handler
    event_handlers = {
        "checkout.session.completed": handle_checkout_completed,
        "customer.subscription.updated": handle_subscription_updated,
        "customer.subscription.deleted": handle_subscription_deleted,
        "invoice.payment_succeeded": handle_payment_succeeded,
        "invoice.payment_failed": handle_payment_failed,
    }

    handler = event_handlers.get(event["type"])
    if handler:
        await handler(db, event["data"]["object"])

    return {"status": "ok"}


@router.get("/checkout/{tier}")
async def create_checkout_session(
    tier: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create Stripe Checkout session for tier upgrade."""
    if tier not in ["essentiel", "pro", "max"]:
        raise HTTPException(status_code=400, detail="Invalid tier")

    client = StripeClient()
    session = client.create_checkout_session(
        user_id=current_user.id,
        user_email=current_user.email,
        tier=tier
    )
    return {"checkout_url": session.url}


@router.get("/status")
async def get_subscription_status(
    current_user: User = Depends(get_current_user)
):
    """Get current user's subscription status and usage."""
    return {
        "tier": current_user.subscription_tier.value,
        "tokens_used": current_user.tokens_used_this_month,
        "monthly_limit": current_user.get_monthly_token_limit(),
        "remaining_tokens": current_user.remaining_tokens,
        "has_monthly_reset": current_user.has_monthly_reset(),
        "subscription_start": current_user.subscription_start.isoformat() if current_user.subscription_start else None,
        "subscription_end": current_user.subscription_end.isoformat() if current_user.subscription_end else None,
    }


@router.post("/portal")
async def create_portal_session(
    current_user: User = Depends(get_current_user)
):
    """Create Stripe Customer Portal session for billing management."""
    if not current_user.stripe_customer_id:
        raise HTTPException(status_code=400, detail="No Stripe customer found")

    client = StripeClient()
    frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
    session = client.create_portal_session(
        customer_id=current_user.stripe_customer_id,
        return_url=f"{frontend_url}/subscription"
    )
    return {"portal_url": session.url}


@router.get("/invoices")
async def list_invoices(
    current_user: User = Depends(get_current_user)
):
    """List billing history invoices."""
    if not current_user.stripe_customer_id:
        return {"invoices": []}

    client = StripeClient()
    invoices = client.list_invoices(current_user.stripe_customer_id)

    # Format invoices for frontend
    formatted_invoices = []
    for inv in invoices:
        formatted_invoices.append({
            "id": inv.id,
            "number": inv.number,
            "date": inv.created,
            "amount_paid": inv.amount_paid,
            "currency": inv.currency,
            "status": inv.status,
            "invoice_pdf": inv.invoice_pdf,
            "hosted_invoice_url": inv.hosted_invoice_url
        })

    return {"invoices": formatted_invoices}


@router.post("/update")
async def update_subscription(
    tier: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update subscription tier."""
    if tier not in ["essentiel", "pro", "max"]:
        raise HTTPException(status_code=400, detail="Invalid tier")

    if not current_user.stripe_subscription_id:
        raise HTTPException(status_code=400, detail="No subscription found")

    # Determine if this is a downgrade
    tier_order = {"essentiel": 1, "pro": 2, "max": 3}
    current_level = tier_order.get(current_user.subscription_tier.value, 0)
    new_level = tier_order.get(tier, 0)
    is_downgrade = new_level < current_level

    client = StripeClient()
    price_id = client.price_ids.get(tier)

    if not price_id:
        raise HTTPException(status_code=400, detail="Invalid tier")

    # Update subscription
    updated_sub = client.update_subscription(
        subscription_id=current_user.stripe_subscription_id,
        new_price_id=price_id,
        is_downgrade=is_downgrade
    )

    # Update user tier in database (will be synced by webhook too)
    from db.models import SubscriptionTier
    current_user.subscription_tier = SubscriptionTier(tier)
    db.commit()

    return {"success": True, "tier": tier}
