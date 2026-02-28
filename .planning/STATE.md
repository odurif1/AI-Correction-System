---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: Phase 5 (Production Readiness)
current_plan: 05-01 (Docker Containerization)
status: completed
last_updated: "2026-02-28T16:53:10.163Z"
progress:
  total_phases: 5
  completed_phases: 5
  total_plans: 24
  completed_plans: 24
  percent: 100
---

# State: La Corrigeuse

**Last updated:** 2026-02-28
**Current phase:** Phase 5 (Production Readiness)
**Current plan:** 05-01 (Docker Containerization)
**Status:** v1.0 milestone complete

## Project Reference

**Core Value:** Qualité des corrections — les notes doivent être justes, fiables et consistantes. La confiance du prof dans les résultats prime sur la vitesse.

**What this is:**
La Corrigeuse est un assistant de correction automatique pour enseignants. Elle lit les copies manuscrites (PDF) et les corrige via LLM avec double-vérification IA. Le prof upload, l'IA corrige, le prof valide et exporte vers Pronote ou Excel.

**Current focus:**
Production readiness - containerization, security scanning, cost optimization, subscription management.

## Current Position

**Phase:** 5 - Production Readiness
**Plan:** 05-01 - Docker Containerization
**Status:** Complete
**Progress:** [██████████] 100%

**What was built:**
Multi-stage Docker build (Python 3.11 + Node 20) for FastAPI + Next.js, Docker Compose orchestration with nginx reverse proxy, rate limiting (10r/s), WebSocket support, volume-based SQLite persistence.

**Why this phase:**
Phases 1-4 complete. Phase 5 enables production deployment with containerization, security scanning, cost tracking, and subscription management.

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

**Phase 5 Plan 01 (2026-02-28):**
- Duration: 25 minutes
- Tasks completed: 3
- Files created: 5 (Dockerfile, .dockerignore, docker-compose.yml, nginx/Dockerfile, nginx/nginx.conf)
- Files modified: 1 (requirements.txt)
- Commits: 3
- Multi-stage Docker build (Python 3.11-slim + Node 20-alpine)
- Final image: 833MB (acceptable for scientific stack)
- Non-root user execution for security
- Health check endpoint integration
- Docker Compose orchestration (app + nginx services)
- nginx reverse proxy with rate limiting (10r/s, burst 20)
- WebSocket support with 3600s timeout
- Volume-based persistence (SQLite + sessions)

**Phase 5 Plan 03 (2026-02-28):**
- Duration: 2 minutes (131 seconds)
- Tasks completed: 3
- Files modified: 5
- Commits: 2
- Provider-specific pricing tables (9 models: Claude, OpenAI, Gemini) in src/llm/pricing.py
- Cost calculation function with cached token discounts (prompt_cost, completion_cost, cached_cost, total_cost, cached_savings)
- OpenAI provider extracts cached tokens from response.usage.prompt_tokens_details.cached
- Gemini provider extracts cached tokens from response.usage_metadata.cached_content_token_count
- Prompt caching settings added to Settings (enable_prompt_caching=True, cache_ttl_seconds=300)
- Costs displayed AFTER grading only (user decision) - not before grading

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
- [Phase 05-03]: Provider-specific pricing tables centralized in src/llm/pricing.py - 9 models (Claude, OpenAI, Gemini) with prompt/completion/cached pricing
- [Phase 05-03]: Cost displayed AFTER grading only (user decision) - not before grading to avoid decision paralysis
- [Phase 05-03]: Prompt caching enabled by default with 5-minute TTL - OpenAI (50% discount), Gemini (75% discount), Claude (90% discount)
- [Phase 05-03]: Cached tokens extracted from API response metadata (prompt_tokens_details.cached, cached_content_token_count)
- [Phase 05]: Security scanning integration with pip-audit and bandit - JSON output for CI/CD, medium severity threshold, continue on findings
- [Phase 05]: FREE tier: 100K tokens one-shot (no monthly reset)
- [Phase 05]: Stripe Checkout for hosted payments (no PCI burden)
- [Phase 05]: Webhook-based tier sync (eventual consistency)

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

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 1 | Améliorer UX modification notes: auto-save on blur + bouton reset | 2026-02-28 | b3d7f1d | [1-am-liorer-ux-modification-notes-auto-sav](./quick/1-am-liorer-ux-modification-notes-auto-sav/) |

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

**Last activity:** 2026-02-28 - Completed quick task 1: Améliorer UX modification notes: auto-save on blur + bouton reset

**Last action:** Completed Phase 5 Plan 03 - Token Cost Estimation and Prompt Caching
**Next action:** Continue Phase 5 plans (05-04: Docker containerization, 05-01: Security scanning)

**Session info:**
- Stopped at: Completed 05-03 Token Cost Estimation and Prompt Caching (pricing tables, cached token extraction, cost calculation)
- Timestamp: 2026-02-28T15:44:00Z

**Quick context for next session:**
We're building an AI-powered grading SaaS for French teachers. Phases 1-4 complete (Security, Observability, Core Grading, UI). Phase 5 (Production Readiness) in progress. Plan 05-03 complete: provider-specific pricing tables, cached token extraction from API responses, cost calculation infrastructure. Cost display will be integrated in dashboard/session detail AFTER grading per user decision. Ready for 05-04 (Docker) or 05-01 (security scanning).

**State preservation:**
- PROJECT.md: Core value, requirements, constraints, key decisions
- REQUIREMENTS.md: 53 v1 requirements with phase mappings and traceability
- ROADMAP.md: 5 phases with goals, requirements, success criteria
- research/SUMMARY.md: Architecture patterns, pitfalls, stack recommendations
- STATE.md (this file): Project memory across sessions

---
*State initialized: 2026-02-26*
*Ready for Phase 1 planning*
