---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Polish & Fix
current_phase: 6
current_plan: —
status: ready_for_planning
last_updated: "2026-02-28T18:30:00.000Z"
progress:
  total_phases: 3
  completed_phases: 0
  total_plans: 6
  completed_plans: 0
  percent: 0
---

# State: La Corrigeuse

**Last updated:** 2026-02-28
**Milestone:** v1.1 — Polish & Fix
**Status:** Ready for planning

## Project Reference

**Core Value:** Qualité des corrections — les notes doivent être justes, fiables et consistantes. La confiance du prof dans les résultats prime sur la vitesse.

**What this is:**
La Corrigeuse est un assistant de correction automatique pour enseignants. Elle lit les copies manuscrites (PDF) et les corrige via LLM avec double-vérification IA. Le prof upload, l'IA corrige, le prof valide et exporte vers Pronote ou Excel.

**Current focus:**
v1.1 — Polish & Fix: Token deduction fix, subscription UX, bug hunting.

## Current Position

**Phase:** 6 (Token Deduction Fix)
**Plan:** —
**Status:** Ready for planning
**Progress:** [░░░░░░░░░░] 0%

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

Run `/gsd:plan-phase 6` to create detailed execution plans for token deduction fix.

---
*State updated: 2026-02-28 for v1.1 milestone*
