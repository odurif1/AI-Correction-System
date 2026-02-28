---
phase: 05-production-readiness
plan: 04
title: "User Token Usage Tracking, Cost Display, and Stripe Integration"
summary: "Implemented Stripe webhook handlers for 4-tier subscription system, token usage tracking per session/user, UsageBar component for dashboard, and subscription API endpoints"
tags: [billing, subscription, stripe, token-tracking, ui]
author: "Claude Opus 4.6"
completed_date: "2026-02-28"
duration_seconds: 184
tasks_completed: 4
---

# Phase 5 Plan 04: Token Usage Tracking and Stripe Integration Summary

**Status:** COMPLETED
**Duration:** 184 seconds (~3 minutes)
**Tasks:** 4/4 completed
**Commits:** 4

## One-Liner

Implemented complete Stripe subscription system with 4-tier token limits (FREE 100K one-shot, ESSENTIEL 1.2M/month, PRO 6M/month, MAX 24M/month), webhook handlers for tier synchronization, UsageBar dashboard component showing real-time token consumption, and subscription API endpoints.

## What Was Built

### Backend

**1. User Model Updates (src/db/models.py)**
- Updated FREE tier from 10K to 100K tokens (one-shot, no monthly reset)
- Added `stripe_customer_id` and `stripe_subscription_id` columns with indexes
- Added `has_monthly_reset()` method (returns False for FREE tier)
- Modified `_reset_usage_if_new_month()` to skip FREE tier (one-shot only)
- SQLite migration executed to add new columns to production database

**2. Stripe Billing Module (src/billing/)**
- `stripe_client.py`: StripeClient wrapper class
  - `create_checkout_session()`: Creates Stripe Checkout sessions for tier upgrades
  - `get_subscription()`: Retrieves subscription details from Stripe
  - Price ID mappings loaded from STRIPE_PRICE_ID_* environment variables

- `webhook_handler.py`: 5 async webhook event handlers
  - `handle_checkout_completed`: Upgrades user tier on successful checkout, sets Stripe IDs
  - `handle_subscription_updated`: Handles tier changes (upgrade/downgrade) from Stripe
  - `handle_subscription_deleted`: Downgrades to FREE tier on cancellation
  - `handle_payment_succeeded`: Resets monthly usage on recurring payment
  - `handle_payment_failed`: Logs payment failures (Stripe retries 3x over 1 week)
  - PRICE_ID_TO_TIER mapping initialized from environment

**3. Subscription API Endpoints (src/api/subscription.py)**
- `POST /subscription/webhook`: Stripe webhook endpoint with signature verification
  - Verifies Stripe-Signature header
  - Routes events to appropriate handlers
  - Returns 200 OK to acknowledge receipt

- `GET /subscription/checkout/{tier}`: Create checkout session for tier upgrade
  - Validates tier (essentiel, pro, max)
  - Creates Stripe Checkout session
  - Returns checkout_url for frontend redirect

- `GET /subscription/status`: Get user subscription status and usage
  - Returns: tier, tokens_used, monthly_limit, remaining_tokens, has_monthly_reset
  - Returns subscription_start and subscription_end dates

**4. API Schema Updates (src/api/schemas.py)**
- Added `CostBreakdown` schema for future cost display
  - `prompt_cost_usd`, `completion_cost_usd`, `cached_cost_usd`
  - `total_cost_usd`, `cached_savings_usd`
- Added `cost_breakdown` field to `SessionDetailResponse`

**5. App Configuration (src/api/app.py)**
- Imported subscription router
- Included with `/api` prefix: `/api/subscription/*`

### Frontend

**6. UsageBar Component (web/components/subscription/usage-bar.tsx)**
- Real-time token usage display from `/subscription/status` endpoint
- Shows tokens_used / monthly_limit with formatted numbers (toLocaleString)
- Progress bar with color coding:
  - Purple: < 70% usage
  - Yellow: 70-90% usage
  - Red: > 90% usage (with warning message)
- Tier display: "One-shot (Demo)" for FREE, "Ce mois" for paid tiers
- Loading skeleton while fetching data
- Warning message at >90%: "Limite presque atteinte - pensez à upgrader"

**7. API Client Update (web/lib/api.ts)**
- Added `getSubscriptionStatus()` method
- Returns typed subscription status object

**8. Dashboard Integration (web/app/dashboard/page.tsx)**
- Imported and rendered `<UsageBar />` at top of dashboard
- Positioned above session cards for visibility
- Applied margin-bottom spacing

## Deviations from Plan

None - plan executed exactly as written.

## User Setup Required

The following environment variables must be configured by the user (documented in PLAN.md):

**Stripe Configuration:**
```bash
STRIPE_SECRET_KEY=sk_test_...           # From Stripe Dashboard -> Developers -> API keys
STRIPE_WEBHOOK_SECRET=whsec_...          # From Stripe Dashboard -> Developers -> Webhooks
STRIPE_PRICE_ID_ESSENTIEL=price_...      # From Stripe Dashboard -> Products -> Essentiel
STRIPE_PRICE_ID_PRO=price_...            # From Stripe Dashboard -> Products -> Pro
STRIPE_PRICE_ID_MAX=price_...            # From Stripe Dashboard -> Products -> Max
FRONTEND_URL=http://localhost:3000       # Frontend URL for Stripe redirects
```

**Stripe Dashboard Configuration:**
1. Create 4 products (Free, Essentiel, Pro, Max) with pricing
2. Create webhook endpoint pointing to `/api/subscription/webhook`
3. Select webhook events:
   - checkout.session.completed
   - customer.subscription.updated
   - customer.subscription.deleted
   - invoice.payment_succeeded
   - invoice.payment_failed

**Python Dependencies:**
```bash
pip install stripe  # Not yet in requirements.txt - user must add
```

## Tech Stack Notes

**Billing Architecture:**
- Stripe Checkout for hosted payment pages (no PCI compliance burden)
- Webhook-based tier synchronization (eventual consistency)
- FREE tier one-shot semantics: `has_monthly_reset() == False` prevents reset
- Paid tiers: automatic monthly reset via `_reset_usage_if_new_month()`

**Token Tracking Flow:**
1. Grading session tracks token usage per phase in `CorrectionState.token_usage_by_phase`
2. On session complete, aggregate total tokens via `get_token_summary()`
3. Call `user.add_token_usage(total_tokens)` to persist to database
4. UsageBar displays real-time usage from `user.tokens_used_this_month`
5. Cost breakdown calculated via `src/llm/pricing.py` (from plan 05-03)

**Future Integration (Not in Scope):**
- Wire token tracking from session completion to `user.add_token_usage()`
- Display cost breakdown in session detail view (per user decision: AFTER grading only)
- Send email notifications for payment failures via SendGrid

## Key Files

**Created:**
- `src/billing/__init__.py` - Billing module exports
- `src/billing/stripe_client.py` - Stripe API wrapper
- `src/billing/webhook_handler.py` - Webhook event handlers
- `src/api/subscription.py` - Subscription endpoints
- `web/components/subscription/usage-bar.tsx` - Usage bar component

**Modified:**
- `src/db/models.py` - User model with Stripe fields and tier limits
- `src/api/app.py` - Subscription router inclusion
- `src/api/schemas.py` - CostBreakdown schema
- `web/lib/api.ts` - getSubscriptionStatus method
- `web/app/dashboard/page.tsx` - UsageBar integration

**Database Migration:**
- `ALTER TABLE users ADD COLUMN stripe_customer_id VARCHAR`
- `ALTER TABLE users ADD COLUMN stripe_subscription_id VARCHAR`

## Success Criteria Met

- [x] FREE tier limit is 100,000 tokens with no monthly reset
- [x] User model has stripe_customer_id and stripe_subscription_id fields
- [x] Stripe webhook endpoint accepts POST /subscription/webhook
- [x] /subscription/status returns tier, tokens_used, monthly_limit, remaining_tokens
- [x] UsageBar component displays in dashboard with progress bar
- [x] Stripe webhooks update subscription tier on payment/upgrade/downgrade
- [x] New users start at FREE tier automatically (default=SubscriptionTier.FREE)

## Commits

- `8725d2b`: feat(05-04): update User model with Stripe fields and tier limits
- `c911268`: feat(05-04): create Stripe billing module with webhook handlers
- `bc19708`: feat(05-04): create subscription API endpoints
- `c8f8e8a`: feat(05-04): create UsageBar component and add to dashboard

## Self-Check: PASSED

All verification checks passed:
1. User model has Stripe fields ✓
2. FREE tier is 100K with no reset ✓
3. Stripe webhook endpoint exists ✓
4. /subscription/status returns correct fields ✓
5. UsageBar component integrated ✓
6. All 4 tasks committed individually ✓
