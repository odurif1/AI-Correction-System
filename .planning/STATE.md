# State: La Corrigeuse

**Last updated:** 2026-02-26
**Current phase:** Phase 1 (Security Foundation)
**Status:** Not started

## Project Reference

**Core Value:** Qualité des corrections — les notes doivent être justes, fiables et consistantes. La confiance du prof dans les résultats prime sur la vitesse.

**What this is:**
La Corrigeuse est un assistant de correction automatique pour enseignants. Elle lit les copies manuscrites (PDF) et les corrige via LLM avec double-vérification IA. Le prof upload, l'IA corrige, le prof valide et exporte vers Pronote ou Excel.

**Current focus:**
Establishing security foundation before commercial release. Critical vulnerabilities (hardcoded JWT secrets, missing multi-tenant isolation) are blocking production deployment.

## Current Position

**Phase:** 1 - Security Foundation
**Plan:** TBD (not yet planned)
**Status:** Not started
**Progress:** █░░░░░░░░░ 10%

**What's being built:**
Secure authentication with environment-based secrets, multi-tenant data isolation (user-scoped sessions), input validation, and file upload security.

**Why this phase:**
Security vulnerabilities are BLOCKING for production. Hardcoded JWT secrets allow token forgery. Missing data isolation means teachers can access each other's grading data. These must be addressed before any production deployment.

## Performance Metrics

**Requirements coverage:** 53/53 v1 requirements mapped to phases
**Planned phases:** 5 (quick depth - critical path focus)
**Estimated timeline:** 3-4 weeks to commercial release (from PROJECT.md)

**Cumulative progress:**
- Phase 1: 0/17 requirements complete
- Phase 2: 0/9 requirements complete
- Phase 3: 0/13 requirements complete
- Phase 4: 0/7 requirements complete
- Phase 5: 0/7 requirements complete

## Accumulated Context

### Decisions Made

**Roadmap structure (2026-02-26):**
- Chose 5-phase structure following risk-mitigation ordering from research
- Grouped Security + Auth into Phase 1 (blocking vulnerabilities)
- Separated Observability into Phase 2 (production readiness foundation)
- Combined Core Grading + Export into Phase 3 (complete user workflow)
- UI as Phase 4 (professional polish)
- Operations + Cost as Phase 5 (deployment optimization)
- Quick depth: Combined aggressive, focus on critical path only

**Rationale:** This ordering ensures we don't build features on an insecure foundation. Security vulnerabilities are irreversible if deployed to production. Observability is required before we can safely debug production issues. Core grading delivers the primary value proposition. UI polish and production automation complete the commercial offering.

### Active Todos

**Immediate (Phase 1):**
1. Plan Phase 1 details: `/gsd:plan-phase 1`
2. Execute Phase 1, Plan 1: Security foundation implementation

**Upcoming:**
- Phase 2: Observability & Monitoring
- Phase 3: Core Grading Experience
- Phase 4: User Interface & Polish
- Phase 5: Production Readiness

### Known Blockers

**Security vulnerabilities (BLOCKING):**
- JWT secret hardcoded in source code — allows attackers to forge tokens
- Multi-tenant data isolation missing — teachers can access each other's sessions
- Input validation incomplete — risk of prompt injection attacks
- File upload security missing — DoS vulnerability from malicious PDFs

**Resolution:** Addressed in Phase 1. Cannot proceed to production until these are fixed.

**AGPL License Risk (LEGAL):**
- PyMuPDF is AGPL-licensed — using in SaaS may require opening source code
- Need to assess commercial license vs replacement with MIT alternative

**Resolution required:** Legal review before production launch. Not blocking development but must resolve before commercial deployment.

### Accumulated Wisdom

**From research/SUMMARY.md:**
- Brownfield enhancement strategy: Use FastAPI middleware/dependency injection to add cross-cutting concerns without breaking existing business logic
- Critical pitfalls: Hardcoded secrets (#1), AGPL violation (#2), prompt injection (#3), cost explosion (#4), cross-tenant leakage (#5)
- AI SaaS economics: Token costs scale linearly; must optimize from day one with model tiering and prompt caching

**From codebase analysis:**
- Existing codebase is well-structured (~31k LOC Python, Next.js frontend)
- Core functionality exists: grading, dual-LLM, PDF vision, export, JWT auth
- Main gaps: Security hardening, observability, UI completion, production deployment

**Key architectural pattern:**
"Middleware shell" approach — add security, monitoring, logging as wrappers around existing routes without modifying business logic. Non-breaking, testable, reversible.

## Session Continuity

**Last action:** Roadmap created with 5 phases covering 53 v1 requirements
**Next action:** Execute `/gsd:plan-phase 1` to create detailed plans for Phase 1 (Security Foundation)

**Quick context for next session:**
We're building an AI-powered grading SaaS for French teachers. The codebase exists but has critical security vulnerabilities. We created a 5-phase roadmap starting with security. We need to plan Phase 1 in detail and begin implementation.

**State preservation:**
- PROJECT.md: Core value, requirements, constraints, key decisions
- REQUIREMENTS.md: 53 v1 requirements with phase mappings and traceability
- ROADMAP.md: 5 phases with goals, requirements, success criteria
- research/SUMMARY.md: Architecture patterns, pitfalls, stack recommendations
- STATE.md (this file): Project memory across sessions

---
*State initialized: 2026-02-26*
*Ready for Phase 1 planning*
