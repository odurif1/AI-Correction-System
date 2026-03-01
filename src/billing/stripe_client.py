"""Stripe client wrapper for subscription management."""

import os
import stripe
from typing import Optional, Dict, Any
from loguru import logger


class StripeClient:
    """Wrapper for Stripe API with La Corrigeuse configuration."""

    def __init__(self):
        self.api_key = os.getenv("STRIPE_SECRET_KEY")
        if not self.api_key:
            raise ValueError("STRIPE_SECRET_KEY environment variable is required")

        stripe.api_key = self.api_key

        # Price ID mappings from environment
        self.price_ids = {
            "essentiel": os.getenv("STRIPE_PRICE_ID_ESSENTIEL"),
            "pro": os.getenv("STRIPE_PRICE_ID_PRO"),
            "max": os.getenv("STRIPE_PRICE_ID_MAX"),
        }

    def create_checkout_session(
        self,
        user_id: str,
        user_email: str,
        tier: str
    ) -> stripe.checkout.Session:
        """Create Stripe Checkout session for tier upgrade."""
        price_id = self.price_ids.get(tier)
        if not price_id:
            raise ValueError(f"Invalid tier: {tier}")

        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            client_reference_id=user_id,
            customer_email=user_email,
            success_url=f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/subscription?success=true",
            cancel_url=f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/subscription?canceled=true",
        )
        return session

    def get_subscription(self, subscription_id: str) -> stripe.Subscription:
        """Retrieve subscription details from Stripe."""
        return stripe.Subscription.retrieve(subscription_id)

    def create_portal_session(
        self,
        customer_id: str,
        return_url: str
    ) -> stripe.billing_portal.Session:
        """Create a Customer Portal session for billing management."""
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )
        return session

    def list_invoices(
        self,
        customer_id: str,
        limit: int = 12
    ) -> list[stripe.Invoice]:
        """List invoices for a customer."""
        invoices = stripe.Invoice.list(
            customer=customer_id,
            limit=limit
        )
        return invoices.data

    def update_subscription(
        self,
        subscription_id: str,
        new_price_id: str,
        is_downgrade: bool = False
    ) -> stripe.Subscription:
        """Update subscription tier with appropriate proration."""
        # For upgrades: prorated immediate charge
        # For downgrades: take effect next billing cycle
        proration_behavior = "none" if is_downgrade else "create_prorations"

        # Get current subscription to find subscription item
        subscription = stripe.Subscription.retrieve(subscription_id)
        subscription_item_id = subscription["items"]["data"][0]["id"]

        return stripe.Subscription.modify(
            subscription_id,
            items=[{
                "id": subscription_item_id,
                "price": new_price_id
            }],
            proration_behavior=proration_behavior
        )
