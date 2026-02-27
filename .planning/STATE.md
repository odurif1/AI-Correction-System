---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: Phase 4 (User Interface & Polish)
current_plan: 04.2 (Fix Dual-LLM Grading Completion)
status: completed
last_updated: "2026-02-27T21:18:09Z"
progress:
  total_phases: 4
  completed_phases: 3
  total_plans: 20
  completed_plans: 20
  percent: 100
---

# State: La Corrigeuse

**Last updated:** 2026-02-27
**Current phase:** Phase 4 (User Interface & Polish)
**Current plan:** 04-04 (Review Screen with Inline Editing)
**Status:** Milestone complete

## Project Reference

**Core Value:** Qualité des corrections — les notes doivent être justes, fiables et consistantes. La confiance du prof dans les résultats prime sur la vitesse.

**What this is:**
La Corrigeuse est un assistant de correction automatique pour enseignants. Elle lit les copies manuscrites (PDF) et les corrige via LLM avec double-vérification IA. Le prof upload, l'IA corrige, le prof valide et exporte vers Pronote ou Excel.

**Current focus:**
Core grading experience - PDF upload workflow with multi-scale detection and teacher confirmation.

## Current Position

**Phase:** 4 - User Interface & Polish
**Plan:** 04-04 - Review Screen with Inline Editing
**Status:** Complete
**Progress:** [██████████] 95%

**What was built:**
Review screen with sortable grades table, inline editing with auto-save, dual-LLM disagreement highlighting, PDF links, and CSV/JSON/Excel export dropdown.

**Why this phase:**
Phase 3 (Core Grading Experience) is complete. UI polish and professional presentation are required for production launch.

## Performance Metrics

**Requirements coverage:** 53/53 v1 requirements mapped to phases
**Planned phases:** 5 (quick depth - critical path focus)
**Estimated timeline:** 3-4 weeks to commercial release (from PROJECT.md)

**Cumulative progress:**
- Phase 1: 5/5 requirements complete (SEC-01, SEC-02, SEC-03, SEC-04, SEC-05)
- Phase 2: 2/9 requirements complete (AUTH-07, AUTH-08)
- Phase 3: 6/13 requirements complete (GRAD-01, GRAD-02, GRAD-03, GRAD-04, GRAD-05, GRAD-06)
- Phase 4: 3/3 requirements complete (UI-04, UI-05, UI-06)

## Performance Metrics

**Phase 3 Plan 01 (2026-02-27):**
- Duration: 2 minutes (130 seconds)
- Tasks completed: 3
- Files modified: 4
- Commits: 3

**Phase 3 Plan 02 (2026-02-27):**
- Duration: 1 minute (102 seconds)
- Tasks completed: 3
- Files modified: 2
- Commits: 3
- Grading mode selection (single/dual) via force_single_llm parameter
- WebSocket progress event constants for type-safe communication
- Progress sync on WebSocket connect for reconnection support

**Phase 4 Plan 02 (2026-02-27):**
- Duration: 2 minutes (178 seconds)
- Tasks completed: 4
- Files modified: 4
- Commits: 5
- Multi-file upload support (up to 50 PDFs) with per-file progress tracking
- FileWithProgress interface for external progress state management
- Purple accent color theme (purple-600) for primary CTAs
- Removed web/ from .gitignore to track frontend changes

**Phase 4 Plan 04 (2026-02-27):**
- Duration: 4 minutes (250 seconds)
- Tasks completed: 5
- Files modified: 4
- Commits: 5
- Review screen with sortable grades table (copy/student, questions, total, percentage)
- Inline editing via EditableGradeCell component (click to edit, Enter/Esc to save/cancel)
- Auto-save each edit with TanStack Query mutation
- Dual-LLM disagreement highlighting (AlertTriangle icon)
- Export dropdown with CSV, JSON, Excel options
- Responsive design (horizontal scroll, sticky first column, purple accent on sort indicators)

**Phase 4 Plan 02 (2026-02-27):**
- Duration: 2 minutes (101 seconds)
- Tasks completed: 1
- Files modified: 1
- Commits: 1
- API_WORKFLOW_STATE constant with CorrectionState(auto_mode=True)
- All 6 GradingSessionOrchestrator API instantiations now pass workflow_state parameter
- Student name disagreements auto-resolve using confidence-based logic instead of blocking CLI prompts
- Dual-LLM batch grading now completes successfully in API context

## Accumulated Context

### Decisions Made

**API auto_mode for orchestrator creation (2026-02-27):**
- API_WORKFLOW_STATE = CorrectionState(auto_mode=True) constant for all API-created orchestrators
- Prevents blocking rich.prompt.Prompt calls in web API context
- Student name disagreements auto-resolve using higher confidence LLM or LLM1 fallback
- Fixes UAT issue where grading never completed due to CLI prompt blocking

**Grading mode selection via force_single_llm (2026-02-27):**
- Single mode uses force_single_llm=True for faster grading without comparison
- Dual mode (default) uses comparison mode from settings for verification
- Grading mode stored in session_progress for UI reference and reconnection sync

**WebSocket progress event structure (2026-02-27):**
- Defined 6 event types as constants: COPY_START, QUESTION_DONE, COPY_DONE, COPY_ERROR, SESSION_COMPLETE, SESSION_ERROR
- Each event has documented data structure for type-safe communication
- Progress sync event sends current state on WebSocket connect for reconnection support

**Multi-scale grading scale detection (2026-02-27):**
- Return all candidate scales with confidence scores instead of single best guess
- Teacher selects final scale before grading begins (prevents costly re-grades)
- Backward compatible: single scale becomes first candidate automatically
- Stored in PreAnalysisResult.candidate_scores: [{"scale": {...}, "confidence": 0.85}, ...]

**Password reset via SendGrid (2026-02-27):**
- Used SendGrid for email delivery (100 emails/day free tier)
- Token storage with SHA-256 hashing before database storage
- Generic forgot-password response prevents email enumeration
- 30-minute token expiration with automatic cleanup
- Auto-login after successful reset (returns JWT token)
- Fixed package name: 'sendgrid' not 'python-sendgrid'

**Structured logging with Loguru (2026-02-27):**
- Migrated to JSON logs with serialize=True for log aggregation
- Added asgi-correlation-id middleware for request tracing
- Correlation ID automatically propagated via contextvars

**Sentry integration (2026-02-27):**
- Added sentry-sdk with FastAPI integration
- Configured for development/production environments
- Health check endpoint with database connectivity

**Health check and metrics (2026-02-27):**
- Log-based metrics collection (latency percentiles, error rate, business metrics)
- Thread-safe in-memory metrics storage with singleton pattern
- Health check endpoint (/health) with database connectivity
- Request middleware records metrics for all requests
- Business metrics track grading operations and token usage
- Sentry user context set in get_current_user for error association

**Multi-tenant session storage (2026-02-26):**
- Changed base path from `data/{user_id}/` to `data/sessions/{user_id}/` for clearer organization
- Made user_id mandatory in SessionStore - no backward compatibility fallback
- Fresh start approach: delete legacy shared sessions on startup rather than migrate
- Enforced file-level isolation with ValueError on missing user_id

**Roadmap structure (2026-02-26):**
- Chose 5-phase structure following risk-mitigation ordering from research
- Grouped Security + Auth into Phase 1 (blocking vulnerabilities)
- Separated Observability into Phase 2 (production readiness foundation)
- Combined Core Grading + Export into Phase 3 (complete user workflow)
- UI as Phase 4 (professional polish)
- Operations + Cost as Phase 5 (deployment optimization)
- Quick depth: Combined aggressive, focus on critical path only

**Rationale:** This ordering ensures we don't build features on an insecure foundation. Security vulnerabilities are irreversible if deployed to production. Observability is required before we can safely debug production issues. Core grading delivers the primary value proposition. UI polish and production automation complete the commercial offering.
- [Phase 01-security-foundation]: Use Pydantic field_validator for default value rejection - Declarative validation integrates with Settings initialization, fail-fast on startup
- [Phase 01-security-foundation]: Dynamic secret fetching in JWT operations - Allows settings reload without restart, avoids stale SECRET_KEY references
- [Phase 01-security-foundation]: Single startup event for validation - Database init already runs on startup, security validation should happen first
- [Phase 01-security-foundation]: User-scoped rate limiting key function - extracts user_id from JWT for authenticated requests, falls back to IP for unauthenticated
- [Phase 01-security-foundation]: Rate limit exception handler returns 429 with French message and retry-after header
- [Phase 01]: Access control middleware returns user_id for downstream SessionStore construction - Avoids duplicate current_user.id references in endpoint logic
- [Phase 01]: Path traversal protection uses Path.resolve().relative_to() for validation - More robust against symlinks and edge cases than string prefix checking
- [Phase 02]: Loguru JSON logging with serialize=True and correlation ID binding for production observability
- [Phase 02]: Log-based metrics instead of Prometheus for v1
- [Phase 02]: Thread-safe in-memory metrics storage
- [Phase 02]: Database check via SELECT 1 query
- [Phase 03]: Single mode uses force_single_llm parameter for faster grading
- [Phase 03]: Dual mode uses comparison mode from settings for verification
- [Phase 03]: Progress sync on WebSocket connect for reconnection support
- [Phase 03]: Grade update validation split between schema (basic type) and endpoint (business rules requiring session context)
- [Phase 03]: Immediate persistence on grade update - changes saved before response returned via SessionStore.save_session()
- [Phase 03]: Enhanced session endpoint returns complete grading_audit with dual-LLM comparison data for review UI
- [Phase 03]: Excel export with openpyxl: Styled headers (bold white on blue), auto-adjusted columns, French localization
- [Phase 03]: Multi-format export endpoint: CSV/JSON/Excel via single URL with format-specific media types
- [Phase 04]: Removed web/ from .gitignore to track frontend UI polish changes in Phase 4
- [Phase 04]: Single FormData request for multi-file upload - simpler than individual XHRs per file
- [Phase 04]: Progress callback with fileIndex parameter enables per-file progress tracking
- [Phase 04]: Purple accent color theme (purple-600) for primary CTAs across Phase 4
- [Phase 04]: Review screen with inline editing - click grade cell to edit, auto-save on Enter/blur, TanStack Query for cache invalidation
- [Phase 04]: Sortable table with sticky first column for horizontal scrolling on smaller screens
- [Phase 04]: Dual-LLM disagreement icons at per-question level using grading_audit.questions[].resolution.agreement field
- [Phase 04.1]: 20-second interval for useRotatingMessage based on user feedback

### Active Todos

**Immediate (Phase 3):**
1. Complete Phase 3 plans (01-07)
2. Finish core grading workflow and UI polish
3. Prepare for production beta testing

**Upcoming:**
- Phase 3: Core Grading Experience
- Phase 4: User Interface & Polish
- Phase 5: Production Readiness

### Known Blockers

**Security vulnerabilities (BLOCKING):**
- ~~JWT secret hardcoded in source code — allows attackers to forge tokens~~ FIXED in plan 01-02
- ~~Multi-tenant data isolation missing — teachers can access each other's sessions~~ FIXED in plan 01-03
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

**Last action:** Completed Phase 4 Plan 04.2 - Fix Dual-LLM Grading Completion
**Next action:** UAT gap closure plan complete. Ready for re-testing of previously blocked tests (13, 15, 19, 20, 21) or Phase 5.

**Session info:**
- Stopped at: Completed 04.2-Fix Dual-LLM Grading Completion (auto_mode for API orchestrators, name disagreement auto-resolution)
- Timestamp: 2026-02-27T21:18:09Z

**Quick context for next session:**
We're building an AI-powered grading SaaS for French teachers. Phase 1 (Security Foundation) complete. Phase 2 (Observability & Monitoring) complete. Phase 3 (Core Grading Experience) complete. Phase 4 (User Interface & Polish) complete including UAT gap closure (04.2). All 20 plans across 4 phases are done. Dual-LLM grading now completes successfully with auto_mode enabled. System ready for production beta testing or Phase 5 (Production Readiness & Cost Optimization).

**State preservation:**
- PROJECT.md: Core value, requirements, constraints, key decisions
- REQUIREMENTS.md: 53 v1 requirements with phase mappings and traceability
- ROADMAP.md: 5 phases with goals, requirements, success criteria
- research/SUMMARY.md: Architecture patterns, pitfalls, stack recommendations
- STATE.md (this file): Project memory across sessions

---
*State initialized: 2026-02-26*
*Ready for Phase 1 planning*
