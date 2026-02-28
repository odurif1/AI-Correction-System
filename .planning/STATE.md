---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Polish & Fix
current_phase: Not started
current_plan: —
status: defining_requirements
last_updated: "2026-02-28T17:45:00.000Z"
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# State: La Corrigeuse

**Last updated:** 2026-02-28
**Milestone:** v1.1 — Polish & Fix
**Status:** Defining requirements

## Project Reference

**Core Value:** Qualité des corrections — les notes doivent être justes, fiables et consistantes. La confiance du prof dans les résultats prime sur la vitesse.

**What this is:**
La Corrigeuse est un assistant de correction automatique pour enseignants. Elle lit les copies manuscrites (PDF) et les corrige via LLM avec double-vérification IA. Le prof upload, l'IA corrige, le prof valide et exporte vers Pronote ou Excel.

**Current focus:**
v1.1 — Polish & Fix: Token deduction, UX improvements, bug hunting.

## Current Position

**Phase:** Not started (defining requirements)
**Plan:** —
**Status:** Defining requirements
**Progress:** [░░░░░░░░░░] 0%

**Target for v1.1:**
- Fix token deduction (backend)
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

### Quick Tasks Completed

| # | Description | Date | Commit |
|---|-------------|------|--------|
| 1 | Améliorer UX modification notes: auto-save on blur + bouton reset | 2026-02-28 | 807da2b |

---
*State reset: 2026-02-28 for v1.1 milestone*
