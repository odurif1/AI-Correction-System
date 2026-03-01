---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Polish & Fix
status: executing
last_updated: "2026-03-01T22:56:36.313Z"
progress:
  total_phases: 2
  completed_phases: 2
  total_plans: 6
  completed_plans: 6
  percent: 100
---

# State: La Corrigeuse

**Last updated:** 2026-03-01
**Milestone:** v1.1 — Polish & Fix
**Status:** Phase 7 in progress

## Project Reference

**Core Value:** Qualité des corrections — les notes doivent être justes, fiables et consistantes. La confiance du prof dans les résultats prime sur la vitesse.

**What this is:**
La Corrigeuse est un assistant de correction automatique pour enseignants. Elle lit les copies manuscrites (PDF) et les corrige via LLM avec double-vérification IA. Le prof upload, l'IA corrige, le prof valide et exporte vers Pronote ou Excel.

**Current focus:**
v1.1 — Polish & Fix: Token deduction fix, subscription UX, bug hunting.

## Current Position

**Phase:** 7 (Subscription UX Polish) - IN PROGRESS
**Plan:** 07-01-stripe-portal-integration - COMPLETE
**Status:** Plan 7-1 complete, ready for plan 7-2
**Progress:** [██████████] 100%

**Target for v1.1:**
- Fix token deduction (backend bug - copy count vs token count) ✓ DONE
- Subscription UX polish (portal, billing history, upgrade/downgrade)
- Bug hunting pass through app

## Accumulated Context

(Preserved from v1.0 milestone — see .planning/milestones/v1.0-phases/ for history)

### Decisions Made

**From v1.0:**
- Dual-LLM comparison for reliability
- Vision directe (pas OCR)
- SQLite for auth
- Next.js standalone
- Cost display AFTER grading
- Single model (no tiering)
- Prompt caching enabled

**From v1.1 Research:**
- Token deduction: deduct AFTER grading, not before
- Use session_id as idempotency key
- No new libraries needed (code fix only)
- Add usage_records table for audit trail

**From v1.1 Phase 6 (Token Deduction Fix):**
- TokenDeductionService integrates into grading completion flow
- WebSocket completion events include tokens_used and remaining_tokens
- InsufficientTokensError handled with user-friendly error messages
- Idempotency prevents double-charging on retry/reconnection
- Pass orchestrator.ai as provider (works with both single and dual LLM modes)
- Robust exception handling with safe defaults and graceful degradation (plan 04)

**From v1.1 Phase 7 Plan 1 (Stripe Portal Integration):**
- Stripe Customer Portal integration for self-service billing management
- Portal session creation with customer validation (free tier excluded)
- "Gérer la facturation" button for paid tiers only
- French error messages for user-friendly UX
- Return URL to /subscription page for context continuity
- External redirect pattern using window.location.href
- [Phase 07]: Upgrade uses create_prorations for immediate prorated charge
- [Phase 07]: Downgrade uses none proration to take effect next billing cycle
- [Phase 07]: BillingHistory returns null for free tier (early return pattern)

### Quick Tasks Completed

| # | Description | Date | Commit |
|---|-------------|------|--------|
| 1 | Améliorer UX modification notes: auto-save on blur + bouton reset | 2026-02-28 | 807da2b |

### Research Completed

| File | Purpose | Date |
|------|---------|------|
| `.planning/research/STACK_TOKENS.md` | Token management fix + Stripe best practices | 2026-02-28 |
| `.planning/research/FEATURES.md` | Token deduction patterns, subscription UX | 2026-02-28 |
| `.planning/research/ARCHITECTURE.md` | Token deduction service architecture | 2026-02-28 |
| `.planning/research/PITFALLS.md` | Token management pitfalls and prevention | 2026-02-28 |

---

## Next Steps

Phase 7 (Subscription UX Polish) is in progress. Plan 7-1 (Stripe Portal Integration) is complete:

**Completed:**
- TokenDeductionService created with idempotency and token aggregation (Phase 6)
- UsageRecord model created for audit trail (Phase 6)
- API integration complete - grading flow now deducts actual tokens (Phase 6)
- Exception handling hardened - grading completes even on billing failures (Phase 6)
- Stripe Customer Portal integration for self-service billing (Plan 7-1)
- Portal session creation with customer validation (Plan 7-1)
- "Gérer la facturation" button for paid tiers (Plan 7-1)

**Remaining v1.1 work:**
1. Plan 7-2: Billing history display and upgrade/downgrade flows
2. Bug hunting pass through app

**Recommended next actions:**
1. Execute Plan 7-2: billing history display + upgrade/downgrade
2. Test portal integration with Stripe Customer Portal
3. Bug hunting pass through app

Run `/gsd:execute-plan` to continue with plan 7-2.

---
*State updated: 2026-03-01 for Phase 7 Plan 01 completion*
