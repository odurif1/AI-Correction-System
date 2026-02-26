# Roadmap: La Corrigeuse

**Created:** 2026-02-26
**Depth:** Quick (3-5 phases, critical path focus)
**Coverage:** 53/53 v1 requirements mapped

## Phases

- [ ] **Phase 1: Security Foundation** - Secure authentication, multi-tenant data isolation, input validation
- [ ] **Phase 2: Observability & Monitoring** - Structured logging, error tracking, metrics, health checks
- [ ] **Phase 3: Core Grading Experience** - Complete grading workflow with dual-LLM, detection, review, and export
- [ ] **Phase 4: User Interface & Polish** - Complete web UI with professional design and responsive layout
- [ ] **Phase 5: Production Readiness** - Containerization, CI/CD, security scanning, cost optimization

## Phase Details

### Phase 1: Security Foundation

**Goal:** Users can securely authenticate and their data is isolated from other users.

**Depends on:** Nothing (first phase)

**Requirements:**
- Security: SEC-01, SEC-02, SEC-03, SEC-04, SEC-05, SEC-06, SEC-07, SEC-08, SEC-09, SEC-10, SEC-11
- Authentication: AUTH-01, AUTH-02, AUTH-03, AUTH-04, AUTH-05, AUTH-06

**Success Criteria** (what must be TRUE):
1. User can create account with email/password and log in securely
2. User session persists across browser refreshes via JWT token
3. User cannot access another user's sessions or grading data (multi-tenant isolation)
4. Application fails fast on startup if JWT secret or API keys are missing/default
5. File uploads are validated for PDF structure and size limits

**Plans:** TBD

---

### Phase 2: Observability & Monitoring

**Goal:** System behavior is visible through logs, metrics, and error tracking for production debugging.

**Depends on:** Phase 1 (requires authenticated users to track)

**Requirements:**
- Authentication: AUTH-07, AUTH-08
- Observability: OBS-01, OBS-02, OBS-03, OBS-04, OBS-05, OBS-06, OBS-07

**Success Criteria** (what must be TRUE):
1. All API requests are logged with method, path, status, latency, and correlation ID
2. Errors are captured with full stack traces and sent to Sentry (or equivalent)
3. User can request password reset via email link with token expiration
4. Health check endpoint (/health) returns API and database status
5. Business metrics (grading operations, token usage, active sessions) are tracked and queryable

**Plans:** TBD

---

### Phase 3: Core Grading Experience

**Goal:** Teachers can upload PDF copies, receive AI-graded results with dual-LLM verification, review grades, and export results.

**Depends on:** Phase 1 (requires user-scoped session storage), Phase 2 (requires progress tracking)

**Requirements:**
- Core Grading: GRAD-01, GRAD-02, GRAD-03, GRAD-04, GRAD-05, GRAD-06, GRAD-07, GRAD-08, GRAD-09
- Export: EXPT-01, EXPT-02, EXPT-03, EXPT-04

**Success Criteria** (what must be TRUE):
1. User can upload multiple PDF copies and see automatic grading scale detection
2. LLM reads handwritten content directly via vision (no OCR step required)
3. Dual-LLM grading mode runs two models independently and displays comparison results
4. Grading progress updates in real-time via WebSocket
5. User can review grades, adjust individual scores, and export to CSV/JSON/Excel

**Plans:** TBD

---

### Phase 4: User Interface & Polish

**Goal:** Web application provides a clean, professional interface that works seamlessly on tablets and laptops.

**Depends on:** Phase 3 (requires complete grading workflow to build UI around)

**Requirements:**
- User Interface: UI-01, UI-02, UI-03, UI-04, UI-05, UI-06, UI-07

**Success Criteria** (what must be TRUE):
1. Dashboard displays recent sessions with quick actions to upload or resume grading
2. Upload workflow guides user through PDF selection and grading options step-by-step
3. Grading progress screen shows real-time updates via WebSocket with clear status indicators
4. Review screen displays all grades with inline editing capability and save confirmation
5. Application works responsively on tablets (768px+) and laptops with professional, clean aesthetic

**Plans:** TBD

---

### Phase 5: Production Readiness

**Goal:** Application is containerized, tested, and cost-optimized for commercial deployment.

**Depends on:** Phase 4 (requires complete application to deploy), Phase 2 (requires metrics to monitor costs)

**Requirements:**
- Operations: OPS-01, OPS-02, OPS-03, OPS-04, OPS-05
- Cost Management: COST-01, COST-02, COST-03, COST-04

**Success Criteria** (what must be TRUE):
1. Application containerizes with Docker multi-stage builds and runs via docker-compose
2. CI/CD pipeline runs tests and security scans (pip-audit, bandit) on every push
3. Token usage is tracked per session with cost estimates displayed before grading
4. Lightweight model is used for detection phase, premium model for grading (model tiering)
5. Prompt caching is enabled for repeated grading criteria to reduce costs

**Plans:** TBD

---

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Security Foundation | 0/2 | Not started | - |
| 2. Observability & Monitoring | 0/2 | Not started | - |
| 3. Core Grading Experience | 0/2 | Not started | - |
| 4. User Interface & Polish | 0/2 | Not started | - |
| 5. Production Readiness | 0/2 | Not started | - |

## Phase Dependencies

```
Phase 1 (Security)
    ↓
Phase 2 (Observability)
    ↓
Phase 3 (Core Grading)
    ↓
Phase 4 (UI) ────────→ Phase 5 (Production)
```

**Critical path:** 1 → 2 → 3 → 4 → 5 (sequential)

**Parallelization opportunity:** Phase 5 (Production Readiness) can begin in parallel with Phase 4 (UI) once Phase 3 (Core Grading) is complete, as containerization and CI/CD don't depend on frontend polish.

## Risk-Based Ordering

This roadmap follows **risk mitigation priority**:

1. **Security first** — vulnerabilities are blocking; data breaches irreversible
2. **Observability second** — production without monitoring is "flying blind"
3. **Core functionality third** — delivers primary value proposition (grading)
4. **UI fourth** — usability matters but internal users can tolerate rougher UX
5. **Operations fifth** — deployment automation needed but manual deploy viable for pilot

Each phase addresses critical pitfalls identified in research:
- Phase 1: Hardcoded secrets, cross-tenant data leakage, file upload DoS
- Phase 2: Missing observability, inability to debug production issues
- Phase 3: Incomplete grading workflow, missing export functionality
- Phase 4: Poor UX hampers adoption
- Phase 5: Cost explosion, deployment failures, security vulnerabilities in CI/CD

---
*Roadmap created: 2026-02-26*
*Ready for planning: `/gsd:plan-phase 1`*
