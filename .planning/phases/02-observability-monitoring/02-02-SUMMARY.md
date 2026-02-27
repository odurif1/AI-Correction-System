---
phase: 02-observability-monitoring
plan: 02
title: Sentry Error Tracking Integration
one_liner: Sentry error tracking with FastAPI integration, health check filtering, and sensitive data stripping
status: complete
date_completed: "2026-02-27"
duration_minutes: 3
tasks_completed: 3
deviation_count: 0
subsystem: Observability
tags: [sentry, error-tracking, observability, fastapi]
requirements_met: [OBS-03, OBS-04]

dependency_graph:
  requires:
    - phase: "01-security-foundation"
      reason: "Requires existing FastAPI app structure with middleware and exception handling patterns"
  provides:
    - component: "Sentry error tracking"
      interfaces: ["init_sentry()", "sentry_exception_handler()", "set_user_context()"]
  affects:
    - component: "Global exception handling"
      impact: "All uncaught exceptions now captured in Sentry"

tech_stack:
  added:
    - library: "sentry-sdk[fastapi]"
      version: "1.40.0"
      purpose: "Error tracking and aggregation"
  patterns:
    - "Global exception handler with @app.exception_handler(Exception)"
    - "Sentry before_send callback for sensitive data filtering"
    - "Graceful degradation when DSN not configured"

key_files:
  created:
    - path: "src/middleware/__init__.py"
      purpose: "Middleware package initialization"
    - path: "src/middleware/error_handler.py"
      purpose: "Sentry integration, exception handling, sensitive data filtering"
  modified:
    - path: "requirements.txt"
      changes: "Added sentry-sdk[fastapi]==1.40.0"
    - path: "src/config/settings.py"
      changes: "Added sentry_dsn, sentry_environment, sentry_traces_sample_rate fields"
    - path: "src/api/app.py"
      changes: "Imported error_handler, initialized Sentry in startup_event, added global exception handler"

decisions_made:
  - decision: "Empty SENTRY_DSN allows graceful degradation"
    rationale: "Development environments may not have Sentry configured. Logging warning and disabling Sentry is better than crashing."
    alternatives_considered:
      - "Make SENTRY_DSN required (rejected: would break development workflow)"
      - "Use mock Sentry for development (rejected: unnecessary complexity)"
  - decision: "Filter health check requests via before_send_transaction"
    rationale: "Health check polling creates noise in Sentry. /health requests are synthetic and not user errors."
    impact: "Reduces Sentry quota usage, cleaner error dashboards"
  - decision: "Strip Authorization, Cookie, X-API-Key headers"
    rationale: "Security best practice - never send credentials or session tokens to external services"
    impact: "Prevents credential leakage in Sentry events"

metrics:
  duration_seconds: 203
  commits: 3
  files_created: 2
  files_modified: 3
  lines_added: 164
  tests_passing: null
---

# Phase 02 - Plan 02: Sentry Error Tracking Integration Summary

## Overview

Integrated Sentry error tracking into La Corrigeuse FastAPI application for production error visibility and debugging. All uncaught exceptions are now automatically captured with full stack traces, request context, and user information. Sensitive data (Authorization headers, cookies, API keys) is automatically stripped before sending to Sentry. Health check requests are filtered to avoid noise.

**Duration:** 3 minutes 23 seconds

## What Was Built

### 1. Dependency and Configuration (Task 1)
- Added `sentry-sdk[fastapi]==1.40.0` to requirements.txt
- Extended Settings class with Sentry configuration:
  - `sentry_dsn: str = ""` - Sentry project DSN (empty = disabled)
  - `sentry_environment: str = "development"` - Environment tag for events
  - `sentry_traces_sample_rate: float = 0.1` - Performance tracing sample rate

### 2. Error Handler Module (Task 2)
Created `src/middleware/error_handler.py` with three key functions:

**init_sentry()** - Initialize Sentry SDK
- Configures FastAPI integration for automatic request data capture
- Configures LoggingIntegration to send ERROR-level logs as events
- Sets traces_sample_rate to 1.0 (development) or 0.1 (production)
- Filters health check transactions via `before_send_transaction`
- Filters sensitive headers via `before_send` callback
- Logs warning and returns early if SENTRY_DSN is empty (graceful degradation)

**_filter_sensitive_data()** - Private callback for Sentry event filtering
- Removes Authorization, Cookie, X-API-Key headers from request
- Removes password, token, api_key, secret, jwt_secret from extra data
- Runs on every event before sending to Sentry

**sentry_exception_handler()** - Global exception handler
- Captures exception in Sentry via `sentry_sdk.capture_exception()`
- Logs error with request path and method
- Returns generic French error message to user (security best practice)
- Registered as `@app.exception_handler(Exception)` for all uncaught exceptions

**set_user_context()** - User context association helper
- Associates user_id, email, username with Sentry events
- Call after authentication to track which user experienced errors

### 3. App Integration (Task 3)
Modified `src/api/app.py`:
- Imported `init_sentry`, `sentry_exception_handler`, `set_user_context`
- Called `init_sentry()` in startup_event after security validation
- Registered global exception handler with `@app.exception_handler(Exception)`

## Verification Results

All success criteria passed:
- ✓ sentry-sdk dependency added to requirements.txt
- ✓ SENTRY_DSN and SENTRY_ENVIRONMENT added to Settings
- ✓ Sentry initialized on app startup with FastAPI integration
- ✓ Global exception handler captures all uncaught exceptions
- ✓ Health check requests filtered from Sentry (before_send_transaction)
- ✓ Sensitive headers (Authorization, Cookie, X-API-Key) stripped before sending
- ✓ Generic error message returned to user, full details in Sentry
- ✓ Application runs without errors when SENTRY_DSN is empty (graceful degradation)

## Deviations from Plan

**None - plan executed exactly as written.**

No bugs, missing functionality, or blocking issues encountered. All tasks completed on first attempt without deviations.

## Authentication Gates

**None encountered.**

Plan did not require external API keys or service configuration. Sentry DSN is optional (empty string = graceful degradation).

## Technical Details

### Sentry Integration Pattern
```python
# Startup initialization
init_sentry(
    dsn=settings.sentry_dsn,
    environment=settings.sentry_environment,
    sample_rate=settings.sentry_traces_sample_rate,
    debug=(settings.sentry_environment == "development")
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return await sentry_exception_handler(request, exc)
```

### Health Check Filtering
```python
before_send_transaction=lambda event: None if event.get("transaction", "").startswith("/health") else event
```
- Returns None for health check transactions (drop them)
- All other transactions pass through normally

### Sensitive Data Filtering
```python
event["request"]["headers"] = {
    k: v for k, v in event["request"]["headers"].items()
    if k.lower() not in ["authorization", "cookie", "x-api-key"]
}
```
- Filters headers at Sentry event level (before sending)
- Case-insensitive matching for header names

### Graceful Degradation
```python
if not dsn or dsn == "":
    logger.warning("SENTRY_DSN not configured - error tracking disabled")
    return
```
- Early return if DSN is empty or None
- Application continues normally without Sentry
- No errors or crashes in development environments

## Integration Notes

**For future authentication enhancements:**
- Call `set_user_context(user_id, email, username)` after JWT validation
- This associates errors with specific users in Sentry dashboard
- Enables "Affected Users" filtering in Sentry issue views

**For production deployment:**
- Set `SENTRY_DSN` environment variable to Sentry project DSN
- Set `SENTRY_ENVIRONMENT=production` for 10% trace sampling
- Set `SENTRY_TRACES_SAMPLE_RATE=0.1` to control performance monitoring overhead

**For development:**
- Leave `SENTRY_DSN` empty to disable error tracking
- Application logs "SENTRY_DSN not configured - error tracking disabled"
- No Sentry events sent, no errors thrown

## Commits

1. **850e698** - `feat(02-02): add Sentry SDK dependency and configuration`
   - Added sentry-sdk[fastapi]==1.40.0
   - Added SENTRY_DSN, SENTRY_ENVIRONMENT, SENTRY_TRACES_SAMPLE_RATE to Settings

2. **46d444f** - `feat(02-02): create Sentry error handler module`
   - Created src/middleware/__init__.py
   - Implemented init_sentry(), sentry_exception_handler(), set_user_context()
   - Added health check filtering via before_send_transaction
   - Added sensitive header stripping

3. **19e1f2a** - `feat(02-02): wire Sentry into app startup and exception handling`
   - Imported error_handler functions in app.py
   - Initialized Sentry in startup_event
   - Added global exception handler for all uncaught exceptions

## Requirements Satisfied

- **OBS-03:** Errors captured with full stack traces and context - ✓ Implemented via Sentry SDK
- **OBS-04:** Production error tracking integrated (Sentry or equivalent) - ✓ Implemented with Sentry

## Self-Check: PASSED

**Files created:**
- ✓ src/middleware/__init__.py exists
- ✓ src/middleware/error_handler.py exists

**Files modified:**
- ✓ requirements.txt contains sentry-sdk[fastapi]==1.40.0
- ✓ src/config/settings.py contains sentry_dsn, sentry_environment, sentry_traces_sample_rate
- ✓ src/api/app.py contains init_sentry() call and global_exception_handler

**Commits exist:**
- ✓ 850e698: feat(02-02): add Sentry SDK dependency and configuration
- ✓ 46d444f: feat(02-02): create Sentry error handler module
- ✓ 19e1f2a: feat(02-02): wire Sentry into app startup and exception handling

**Success criteria:**
- ✓ All 8 success criteria verified and passed

---

*Plan completed: 2026-02-27*
*Executor: GSD Plan Executor (Phase 02-02)*
