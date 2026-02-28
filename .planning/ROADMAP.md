# Roadmap: La Corrigeuse

**Created:** 2026-02-26
**Current Milestone:** v1.1 — Polish & Fix

## Milestones

- ✅ **v1.0 MVP** — Phases 1-5 (shipped 2026-02-28)
- 🚧 **v1.1 Polish & Fix** — Token deduction, UX improvements, bug fixes (in progress)

## Phases

<details>
<summary>✅ v1.0 MVP (Phases 1-5) — SHIPPED 2026-02-28</summary>

- [x] Phase 1: Security Foundation (5/5 plans) — completed 2026-02-27
- [x] Phase 2: Observability & Monitoring (4/4 plans) — completed 2026-02-27
- [x] Phase 3: Core Grading Experience (4/4 plans) — completed 2026-02-27
- [x] Phase 4: User Interface & Polish (6/6 plans) — completed 2026-02-27
- [x] Phase 5: Production Readiness (4/4 plans) — completed 2026-02-28

**Archive:** `.planning/milestones/v1.0-ROADMAP.md`

</details>

### 🚧 v1.1 Polish & Fix (Current Milestone)

| Phase | Goal | Status | Plans |
|-------|------|--------|-------|
| **Phase 6** | Token Deduction Fix | 🔴 Not started | 0/3 |
| **Phase 7** | Subscription UX Polish | 🔴 Not started | 0/2 |
| **Phase 8** | Bug Hunting & Quality Pass | 🔴 Not started | 0/1 |

---

#### Phase 6: Token Deduction Fix

**Goal:** Fix token deduction so users are charged for actual tokens used

**Context:**
- Token tracking exists in AI providers (`BaseProvider._log_call()`)
- Bug: `app.py:1273` passes copy count instead of token count to `User.add_token_usage()`
- Result: Users charged ~30 tokens instead of ~150,000 tokens

**Plans:**
1. **PLAN-6.1: Token Aggregation Service** — Create service to aggregate tokens from providers
2. **PLAN-6.2: Database Integration** — Add usage_records table for audit trail
3. **PLAN-6.3: API Integration** — Wire service into grading completion flow

**Delivers:**
- ✅ Correct token deduction from user balances
- ✅ Idempotent deduction (no double-charging)
- ✅ Audit trail for every deduction

**Dependencies:** None

---

#### Phase 7: Subscription UX Polish

**Goal:** Move token bar to subscription page and enhance subscription management

**Context:**
- Token usage bar currently on dashboard
- Subscription page exists but lacks full billing management
- Users expect usage/billing in the same location

**Plans:**
1. **PLAN-7.1: Move Usage Bar** — Relocate from dashboard to subscription page
2. **PLAN-7.2: Subscription Enhancements** — Add Stripe portal, billing history, upgrade CTAs

**Delivers:**
- ✅ Token usage on subscription page with color-coded thresholds
- ✅ Stripe Customer Portal for payment method management
- ✅ Billing history section

**Dependencies:** Phase 6 (token deduction must work first)

---

#### Phase 8: Bug Hunting & Quality Pass

**Goal:** Identify and fix bugs through full application walkthrough

**Context:**
- v1.0 MVP shipped quickly, may have edge case bugs
- Need systematic pass to find and fix issues
- Focus on user-facing bugs, not refactoring

**Plans:**
1. **PLAN-8.1: Bug Hunting Pass** — Systematic walkthrough of all user flows

**Areas to check:**
- Authentication flow (login, logout, password reset)
- Session workflow (upload, pre-analysis, grading, review)
- Grade editing (inline editing, reset, disagreement resolution)
- Export (CSV, JSON, Excel download)
- Subscription (tier display, upgrade flow)
- Error handling (API errors, network failures)
- Responsive design (tablet and laptop layouts)

**Delivers:**
- ✅ No critical bugs remaining
- ✅ Application stable and reliable

**Dependencies:** Phase 6, Phase 7

---

## Known Gaps (Deferred)

| Gap | Status | Resolution |
|-----|--------|------------|
| Docker deployment | Deferred | Wait for PyMuPDF Python 3.13+ support |
| Calibration (GRAD-07) | Deferred | v1.2+ |
| CI/CD pipeline (OPS-03) | Deferred | Manual deployment for pilot |
| Token estimation accuracy | Deferred | v1.2+ |
| Usage notification system | Deferred | v1.2+ |

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Security Foundation | v1.0 | 5/5 | Complete | 2026-02-27 |
| 2. Observability | v1.0 | 4/4 | Complete | 2026-02-27 |
| 3. Core Grading | v1.0 | 4/4 | Complete | 2026-02-27 |
| 4. UI & Polish | v1.0 | 6/6 | Complete | 2026-02-27 |
| 5. Production | v1.0 | 4/4 | Complete | 2026-02-28 |
| 6. Token Deduction | v1.1 | 0/3 | Not started | — |
| 7. Subscription UX | v1.1 | 0/2 | Not started | — |
| 8. Bug Hunting | v1.1 | 0/1 | Not started | — |

---
*Roadmap created: 2026-02-26*
*v1.0 MVP shipped: 2026-02-28*
*v1.1 planning: 2026-02-28*
