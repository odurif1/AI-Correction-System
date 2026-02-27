# Requirements: La Corrigeuse

**Defined:** 2026-02-26
**Core Value:** Qualité des corrections — les notes doivent être justes, fiables et consistantes. La confiance du prof dans les résultats prime sur la vitesse.

## v1 Requirements

Requirements for initial commercial release. Each maps to roadmap phases.

### Security (Critical)

- [x] **SEC-01**: JWT secret loaded from environment variable, never hardcoded
- [x] **SEC-02**: Application fails fast on startup if critical secrets are missing or default values detected
- [x] **SEC-03**: All API keys (Gemini, OpenAI, GLM) loaded from environment variables
- [x] **SEC-04**: Multi-tenant data isolation — users cannot access other users' sessions or grading data
- [x] **SEC-05**: User-scoped session storage (data/sessions/{user_id}/)
- [x] **SEC-06**: Access control middleware verifies ownership on every data access request
- [x] **SEC-07**: Input validation on all API endpoints (Pydantic models)
- [x] **SEC-08**: File upload security — validate PDF structure, enforce size limits, reject malicious files
- [x] **SEC-09**: Rate limiting per user on uploads and API calls
- [x] **SEC-10**: CORS properly configured for production domains
- [x] **SEC-11**: Security headers applied (HSTS, X-Frame-Options, X-Content-Type-Options)

### Authentication

- [x] **AUTH-01**: User can sign up with email and password
- [x] **AUTH-02**: Password hashing with bcrypt (no plain text storage)
- [x] **AUTH-03**: User can log in with email and password
- [x] **AUTH-04**: JWT token issued on successful login with configurable expiration
- [x] **AUTH-05**: User session persists across browser refresh (token stored securely)
- [x] **AUTH-06**: User can log out (token invalidation)
- [x] **AUTH-07**: User can request password reset via email link
- [x] **AUTH-08**: Password reset tokens expire after reasonable time (15-60 minutes)

### Observability

- [x] **OBS-01**: Structured JSON logging with correlation IDs for request tracing
- [x] **OBS-02**: All API requests logged with method, path, status, latency
- [x] **OBS-03**: Errors captured with full stack traces and context
- [x] **OBS-04**: Production error tracking integrated (Sentry or equivalent)
- [x] **OBS-05**: Health check endpoint (/health) returns API and database status
- [x] **OBS-06**: Request metrics collected (latency, error rates, throughput)
- [x] **OBS-07**: Business metrics tracked (grading operations, token usage per phase, active sessions)

### Core Grading

- [x] **GRAD-01**: User can upload multiple PDF copies (batch upload)
- [x] **GRAD-02**: System automatically detects grading scale (barème) from PDF
- [x] **GRAD-03**: LLM reads handwritten content directly via vision (no OCR step)
- [x] **GRAD-04**: Single-LLM grading mode produces grade and feedback
- [x] **GRAD-05**: Dual-LLM grading mode runs two models independently and compares results
- [x] **GRAD-06**: Grading progress displayed via WebSocket updates
- [ ] **GRAD-07**: Calibration across copies detects grading inconsistencies
- [ ] **GRAD-08**: User can review and adjust grades before finalizing
- [ ] **GRAD-09**: Grading results persist and can be resumed later

### Export

- [ ] **EXPT-01**: User can export grading results to CSV
- [ ] **EXPT-02**: User can export grading results to JSON
- [ ] **EXPT-03**: User can export grading results to Excel format
- [ ] **EXPT-04**: Export includes student identifiers, grades per question, total, and feedback

### User Interface

- [ ] **UI-01**: Dashboard displays recent sessions and quick actions
- [ ] **UI-02**: Upload workflow guides user through PDF selection and grading options
- [ ] **UI-03**: Grading progress shown with real-time updates
- [ ] **UI-04**: Review screen displays all grades with ability to edit
- [ ] **UI-05**: Responsive design works on tablets and laptops
- [ ] **UI-06**: Clean, professional aesthetic (not playful or consumer-oriented)
- [ ] **UI-07**: Error states display user-friendly messages (not stack traces)

### Operations

- [ ] **OPS-01**: Application containerized with Docker multi-stage builds
- [ ] **OPS-02**: Docker Compose configuration for local development and production
- [ ] **OPS-03**: CI/CD pipeline runs tests and security scans on every push
- [ ] **OPS-04**: Automated dependency vulnerability scanning (pip-audit)
- [ ] **OPS-05**: Static security analysis integrated (bandit)

### Cost Management

- [ ] **COST-01**: Token usage tracked per session and per user
- [ ] **COST-02**: Token costs estimated and displayed to user before grading
- [ ] **COST-03**: Model tiering — lightweight model for detection, premium for grading
- [ ] **COST-04**: Prompt caching enabled for repeated grading criteria

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Competitive Differentiators

- **DIFF-01**: Pronote direct integration (export grades to French school system)
- **DIFF-02**: Long-term memory of student performance across sessions
- **DIFF-03**: Fine-grained AI feedback generation for students
- **DIFF-04**: Plagiarism detection across submitted copies
- **DIFF-05**: Multi-language support for international expansion
- **DIFF-06**: Explainable AI decisions with full reasoning trace

### Advanced Features

- **ADV-01**: Outcome-based pricing (pay per graded paper)
- **ADV-02**: Team/organization accounts with shared sessions
- **ADV-03**: Custom grading rubrics saved per teacher/subject
- **ADV-04**: Offline mode with sync (PWA)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Mobile native app | Web-first approach; responsive design sufficient for v1 |
| OAuth/Social login | Email/password sufficient; adds complexity and privacy concerns |
| Real-time chat support | Not core to grading; email support sufficient |
| Video content storage | High bandwidth/storage costs; external hosting available |
| Free tier | AI costs non-trivial; free trial instead |
| On-premise deployment | SaaS model simpler; enterprise later |
| Gamification | Teachers are professionals; patronizing UX risk |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| SEC-01 | Phase 1 | Complete |
| SEC-02 | Phase 1 | Complete |
| SEC-03 | Phase 1 | Complete |
| SEC-04 | Phase 1 | Complete |
| SEC-05 | Phase 1 | Complete |
| SEC-06 | Phase 1 | Complete |
| SEC-07 | Phase 1 | Complete |
| SEC-08 | Phase 1 | Complete |
| SEC-09 | Phase 1 | Complete |
| SEC-10 | Phase 1 | Complete |
| SEC-11 | Phase 1 | Complete |
| AUTH-01 | Phase 1 | Complete |
| AUTH-02 | Phase 1 | Complete |
| AUTH-03 | Phase 1 | Complete |
| AUTH-04 | Phase 1 | Complete |
| AUTH-05 | Phase 1 | Complete |
| AUTH-06 | Phase 1 | Complete |
| AUTH-07 | Phase 2 | Complete |
| AUTH-08 | Phase 2 | Complete |
| OBS-01 | Phase 2 | Complete |
| OBS-02 | Phase 2 | Complete |
| OBS-03 | Phase 2 | Complete |
| OBS-04 | Phase 2 | Complete |
| OBS-05 | Phase 2 | Complete |
| OBS-06 | Phase 2 | Complete |
| OBS-07 | Phase 2 | Complete |
| GRAD-01 | Phase 3 | Complete |
| GRAD-02 | Phase 3 | Complete |
| GRAD-03 | Phase 3 | Complete |
| GRAD-04 | Phase 3 | Complete |
| GRAD-05 | Phase 3 | Complete |
| GRAD-06 | Phase 3 | Complete |
| GRAD-07 | Phase 3 | Pending |
| GRAD-08 | Phase 3 | Pending |
| GRAD-09 | Phase 3 | Pending |
| EXPT-01 | Phase 3 | Pending |
| EXPT-02 | Phase 3 | Pending |
| EXPT-03 | Phase 3 | Pending |
| EXPT-04 | Phase 3 | Pending |
| UI-01 | Phase 4 | Pending |
| UI-02 | Phase 4 | Pending |
| UI-03 | Phase 4 | Pending |
| UI-04 | Phase 4 | Pending |
| UI-05 | Phase 4 | Pending |
| UI-06 | Phase 4 | Pending |
| UI-07 | Phase 4 | Pending |
| OPS-01 | Phase 5 | Pending |
| OPS-02 | Phase 5 | Pending |
| OPS-03 | Phase 5 | Pending |
| OPS-04 | Phase 5 | Pending |
| OPS-05 | Phase 5 | Pending |
| COST-01 | Phase 5 | Pending |
| COST-02 | Phase 5 | Pending |
| COST-03 | Phase 5 | Pending |
| COST-04 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 53 total
- Mapped to phases: 53
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-26*
*Last updated: 2026-02-26 after research synthesis*
