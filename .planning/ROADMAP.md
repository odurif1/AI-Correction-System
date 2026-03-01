# Roadmap: La Corrigeuse

**Created:** 2026-02-26
**Current Milestone:** Planning v1.2

## Milestones

- ✅ **v1.0 MVP** — Phases 1-5 (shipped 2026-02-28)
- ✅ **v1.1 Polish & Fix** — Phases 6-7 (shipped 2026-03-01)
- 📋 **v1.2** — Bug hunting, improvements (planned)

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

<details>
<summary>✅ v1.1 Polish & Fix (Phases 6-7) — SHIPPED 2026-03-01</summary>

- [x] Phase 6: Token Deduction Fix (4/4 plans) — completed 2026-02-28
- [x] Phase 7: Subscription UX Polish (2/2 plans) — completed 2026-03-01

**Archive:** `.planning/milestones/v1.1-ROADMAP.md`

</details>

### 📋 v1.2 (Planned)

| Phase | Goal | Status | Plans |
|-------|------|--------|-------|
| **Phase 8** | Bug Hunting & Quality Pass | 🔴 Not started | 0/1 |

---

#### Phase 8: Bug Hunting & Quality Pass

**Goal:** Identify and fix bugs through full application walkthrough

**Context:**
- v1.1 shipped, may have edge case bugs
- Need systematic pass to find and fix issues
- Focus on user-facing bugs, not refactoring

**Plans:**
1. **PLAN-8.1: Bug Hunting Pass** — Systematic walkthrough of all user flows

**Areas to check:**
- Authentication flow (login, logout, password reset)
- Session workflow (upload, pre-analysis, grading, review)
- Grade editing (inline editing, reset, disagreement resolution)
- Export (CSV, JSON, Excel download)
- Subscription (tier display, upgrade flow, billing history)
- Error handling (API errors, network failures)
- Responsive design (tablet and laptop layouts)

**Delivers:**
- ✅ No critical bugs remaining
- ✅ Application stable and reliable

**Dependencies:** None (can start anytime)

---

## Known Gaps (Deferred)

| Gap | Status | Resolution |
|-----|--------|------------|
| Docker deployment | Deferred | Wait for PyMuPDF Python 3.13+ support |
| Calibration (GRAD-07) | Deferred | v1.3+ |
| CI/CD pipeline (OPS-03) | Deferred | Manual deployment for pilot |
| Token estimation accuracy | Deferred | v1.3+ |
| Usage notification system | Deferred | v1.3+ |

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Security Foundation | v1.0 | 5/5 | Complete | 2026-02-27 |
| 2. Observability | v1.0 | 4/4 | Complete | 2026-02-27 |
| 3. Core Grading | v1.0 | 4/4 | Complete | 2026-02-27 |
| 4. UI & Polish | v1.0 | 6/6 | Complete | 2026-02-27 |
| 5. Production | v1.0 | 4/4 | Complete | 2026-02-28 |
| 6. Token Deduction | v1.1 | 4/4 | Complete | 2026-02-28 |
| 7. Subscription UX | v1.1 | 2/2 | Complete | 2026-03-01 |
| 8. Bug Hunting | v1.2 | 0/1 | Not started | — |

---
*Roadmap created: 2026-02-26*
*v1.0 MVP shipped: 2026-02-28*
*v1.1 Polish & Fix shipped: 2026-03-01*
