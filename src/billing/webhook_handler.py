"""Stripe webhook event handlers."""

from sqlalchemy.orm import Session
from loguru import logger
import stripe
from db.database import get_db
from db.models import User, SubscriptionTier
from billing.stripe_client import StripeClient
from datetime import datetime


# Price ID to tier mapping (load from env or Stripe)
PRICE_ID_TO_TIER = {}


def init_price_mappings():
    """Initialize price ID to tier mappings from environment."""
    import os
    PRICE_ID_TO_TIER.update({
        os.getenv("STRIPE_PRICE_ID_ESSENTIEL", ""): SubscriptionTier.ESSENTIEL,
        os.getenv("STRIPE_PRICE_ID_PRO", ""): SubscriptionTier.PRO,
        os.getenv("STRIPE_PRICE_ID_MAX", ""): SubscriptionTier.MAX,
    })


init_price_mappings()


async def handle_checkout_completed(db: Session, event_data: dict):
    """Handle successful checkout - upgrade user tier."""
    client_reference_id = event_data.get("client_reference_id")
    subscription_id = event_data.get("subscription")
    customer_id = event_data.get("customer")

    # Fetch subscription details
    subscription = stripe.Subscription.retrieve(subscription_id)
    price_id = subscription["items"]["data"][0]["price"]["id"]

    # Map price_id to tier
    tier = PRICE_ID_TO_TIER.get(price_id)
    if not tier:
        logger.error(f"Unknown price_id: {price_id}")
        return

    # Update user
    user = db.query(User).filter_by(id=client_reference_id).first()
    if user:
        user.subscription_tier = tier
        user.stripe_customer_id = customer_id
        user.stripe_subscription_id = subscription_id
        user.subscription_start = datetime.utcnow()
        user.subscription_end = datetime.fromtimestamp(subscription["current_period_end"])
        db.commit()
        logger.info(f"User {user.id} upgraded to {tier}")


async def handle_subscription_updated(db: Session, event_data: dict):
    """Handle subscription change (upgrade/downgrade)."""
    subscription_id = event_data["id"]
    price_id = event_data["items"]["data"][0]["price"]["id"]
    customer_id = event_data["customer"]

    tier = PRICE_ID_TO_TIER.get(price_id)
    if not tier:
        logger.error(f"Unknown price_id: {price_id}")
        return

    user = db.query(User).filter_by(stripe_customer_id=customer_id).first()
    if user:
        user.subscription_tier = tier
        user.subscription_end = datetime.fromtimestamp(event_data["current_period_end"])
        db.commit()
        logger.info(f"User {user.id} subscription updated to {tier}")


async def handle_subscription_deleted(db: Session, event_data: dict):
    """Handle subscription cancellation - downgrade to Free."""
    customer_id = event_data["customer"]

    user = db.query(User).filter_by(stripe_customer_id=customer_id).first()
    if user:
        user.subscription_tier = SubscriptionTier.FREE
        user.subscription_end = datetime.utcnow()
        user.stripe_subscription_id = None
        db.commit()
        logger.info(f"User {user.id} subscription canceled, downgraded to Free")


async def handle_payment_succeeded(db: Session, event_data: dict):
    """Handle successful recurring payment - reset monthly usage."""
    customer_id = event_data["customer"]

    user = db.query(User).filter_by(stripe_customer_id=customer_id).first()
    if user:
        user._reset_usage_if_new_month()
        db.commit()
        logger.info(f"User {user.id} monthly usage reset")


async def handle_payment_failed(db: Session, event_data: dict):
    """Handle payment failure - log but don't block immediately."""
    customer_id = event_data["customer"]
    logger.warning(f"Payment failed for customer {customer_id}")
    # Stripe retries 3 times over 1 week before canceling
    # Send email notification via SendGrid (future task)
