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
