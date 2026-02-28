---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Polish & Fix
status: completed
last_updated: "2026-02-28T20:31:39.719Z"
progress:
  total_phases: 1
  completed_phases: 1
  total_plans: 3
  completed_plans: 3
  percent: 100
---

# State: La Corrigeuse

**Last updated:** 2026-02-28
**Milestone:** v1.1 — Polish & Fix
**Status:** Milestone complete

## Project Reference

**Core Value:** Qualité des corrections — les notes doivent être justes, fiables et consistantes. La confiance du prof dans les résultats prime sur la vitesse.

**What this is:**
La Corrigeuse est un assistant de correction automatique pour enseignants. Elle lit les copies manuscrites (PDF) et les corrige via LLM avec double-vérification IA. Le prof upload, l'IA corrige, le prof valide et exporte vers Pronote ou Excel.

**Current focus:**
v1.1 — Polish & Fix: Token deduction fix, subscription UX, bug hunting.

## Current Position

**Phase:** 6 (Token Deduction Fix) - COMPLETE
**Plan:** 06-03-api-integration - COMPLETE
**Status:** Phase 6 complete, ready for next phase
**Progress:** [██████████] 100%

**Target for v1.1:**
- Fix token deduction (backend bug - copy count vs token count)
- Move token bar to Subscription page
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

Phase 6 (Token Deduction Fix) is complete. Token deduction bug is fixed:

**Completed:**
- TokenDeductionService created with idempotency and token aggregation
- UsageRecord model created for audit trail
- API integration complete - grading flow now deducts actual tokens

**Recommended next actions:**
1. Test grading flow with various token balances
2. Verify WebSocket events display token usage correctly
3. Test insufficient tokens error handling
4. Move token bar to Subscription page (remaining v1.1 work)
5. Bug hunting pass through app

Run `/gsd:plan-phase` to start the next phase.

---
*State updated: 2026-02-28 for v1.1 milestone*
