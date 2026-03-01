---
gsd_state_version: 1.0
milestone: v1.2
milestone_name: TBD
status: planning
last_updated: "2026-03-02T00:00:00.000Z"
progress:
  total_phases: 1
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# State: La Corrigeuse

**Last updated:** 2026-03-02
**Milestone:** v1.2 — Planning
**Status:** Ready for new milestone planning

## Project Reference

**Core Value:** Qualité des corrections — les notes doivent être justes, fiables et consistantes. La confiance du prof dans les résultats prime sur la vitesse.

**What this is:**
La Corrigeuse est un assistant de correction automatique pour enseignants. Elle lit les copies manuscrites (PDF) et les corrige via LLM avec double-vérification IA. Le prof upload, l'IA corrige, le prof valide et exporte vers Pronote ou Excel.

**Current focus:**
v1.2 planning — Bug hunting pass, improvements, or new features based on user feedback.

## Current Position

**Phase:** 8 (Bug Hunting & Quality Pass) - NOT STARTED
**Status:** v1.1 shipped, ready for v1.2 planning
**Progress:** [░░░░░░░░░░] 0%

**Shipped in v1.1:**
- ✓ Token deduction fix (actual tokens, not copy count)
- ✓ UsageRecord audit trail
- ✓ Stripe Customer Portal
- ✓ Billing history + upgrade/downgrade

## Accumulated Context

### Decisions Made

**From v1.0:**
- Dual-LLM comparison for reliability
- Vision directe (pas OCR)
- SQLite for auth
- Next.js standalone
- Cost display AFTER grading
- Single model (no tiering)
- Prompt caching enabled

**From v1.1:**
- Service layer for token deduction (testability, separation of concerns)
- Row locking for concurrent operations
- Database-level idempotency via UniqueConstraint
- Stripe Portal for self-service billing
- Proration: immediate upgrades, next-cycle downgrades

### Completed Milestones

| Milestone | Phases | Plans | Shipped |
|-----------|--------|-------|---------|
| v1.0 MVP | 5 | 23 | 2026-02-28 |
| v1.1 Polish & Fix | 2 | 6 | 2026-03-01 |

---

## Next Steps

v1.1 is complete and archived. Phase 8 (Bug Hunting) is available in the roadmap but not yet planned.

**Recommended actions:**
1. `/gsd:new-milestone` — Start v1.2 planning (questioning → research → requirements → roadmap)
2. Or `/gsd:plan-phase 8` — Plan the Bug Hunting phase directly

---
*State updated: 2026-03-02 after v1.1 milestone completion*
