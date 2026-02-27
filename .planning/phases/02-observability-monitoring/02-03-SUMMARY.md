---
phase: 02-observability-monitoring
plan: 03
title: "Health Check & Metrics"
one_liner: "Health check endpoint with database connectivity and log-based metrics collection for latency, errors, and business operations"
subsystem: "Observability & Monitoring"
tags:
  - observability
  - health-check
  - metrics
  - monitoring
  - production-readiness

dependency_graph:
  requires:
    - "02-01 (Structured logging with Loguru)"
  provides:
    - "Health check endpoint for load balancers"
    - "Request metrics (latency percentiles, error rate)"
    - "Business metrics (grading operations, token usage)"
  affects:
    - "Phase 3: Core Grading Experience (metrics for performance analysis)"
    - "Phase 5: Production Readiness (monitoring foundation)"

tech_stack:
  added:
    - "Loguru: Structured JSON logging for metrics emission"
  patterns:
    - "Singleton pattern: Global metrics_collector instance"
    - "Thread-safe metrics: Lock-protected in-memory aggregation"
    - "Log-based metrics: Emit metrics as structured JSON logs"

key_files:
  created:
    - path: "src/api/health.py"
      description: "Health check endpoint with database connectivity"
      exports: ["router as health_router", "health_check()"]
    - path: "src/utils/metrics.py"
      description: "Request and business metrics collection utilities"
      exports: ["MetricsCollector", "get_metrics_collector()"]
  modified:
    - path: "src/api/app.py"
      description: "Health router integration, metrics initialization, request middleware"
      changes:
        - "Imported health_router and get_metrics_collector"
        - "Included health router with tags"
        - "Initialized metrics_collector in app.state during startup"
        - "Removed old basic /health endpoint"
        - "RequestLoggingMiddleware records request metrics"
        - "create_session records active session metrics"
        - "start_grading records grading operation metrics"
    - path: "src/api/auth.py"
      description: "Sentry user context for error association"
      changes:
        - "get_current_user sets Sentry user context"

decisions:
  - title: "Log-based metrics instead of Prometheus for v1"
    rationale: "Simpler architecture, sufficient for production monitoring, log aggregation tools (Datadog, Loki) can query metrics fields"
    impact: "Metrics stored in JSON logs, reduced operational complexity, no separate metrics server"
  - title: "Thread-safe in-memory metrics storage"
    rationale: "Fast access from async endpoints, lock-protected collections prevent race conditions, metrics reset on restart acceptable for v1"
    impact: "Metrics lost on restart, no persistence, but simple and performant"
  - title: "Database check via SELECT 1 query"
    rationale: "Fast connection validation, doesn't require specific tables, works with any database"
    impact: "Quick health checks, detects connection issues but not table-level corruption"

metrics:
  duration_seconds: 245
  completed_date: "2026-02-27T00:40:49Z"
  tasks_completed: 4
  files_created: 2
  files_modified: 2
  commits: 4
  requirements_completed:
    - OBS-05
    - OBS-06
    - OBS-07

deviations_from_plan: |
  None - plan executed exactly as written.

authentication_gates: |
  None

---

# Phase 02 Plan 03: Health Check & Metrics - Summary

## Objective

Implement health check endpoint and metrics collection for monitoring system health, request performance, and business operations visibility in production.

## What Was Built

### 1. Health Check Endpoint (`/health`)

**File:** `src/api/health.py`

- Returns JSON with `status`, `version`, and `database` fields
- HTTP 200 when healthy, HTTP 503 when database disconnected
- Database connectivity check via fast `SELECT 1` query
- Router tagged for OpenAPI documentation

**Response format:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "database": "connected"
}
```

### 2. Metrics Collection System

**File:** `src/utils/metrics.py`

**Request Metrics:**
- Latency percentiles: p50, p95, p99
- Error rate: (4xx/5xx responses) / total requests
- Status code distribution
- Total request count

**Business Metrics:**
- Grading operations count
- Total token usage (aggregated across sessions)
- Active sessions count

**Implementation:**
- Thread-safe singleton (`get_metrics_collector()`)
- Lock-protected in-memory storage
- Metrics logged as structured JSON via Loguru
- Resets on restart (acceptable for v1)

### 3. App Integration

**File:** `src/api/app.py`

- Health router included with `tags=["health"]`
- Metrics collector initialized in `startup_event()`
- Old basic `/health` endpoint removed
- `RequestLoggingMiddleware` records request metrics
- `create_session` endpoint records active sessions
- `start_grading` background task records grading operations

### 4. Sentry User Context

**File:** `src/api/auth.py`

- `get_current_user()` now sets Sentry user context
- All errors associated with authenticated user
- Improves error tracking and debugging

## Architecture Decisions

### Log-Based Metrics (vs Prometheus)

**Decision:** Emit metrics as structured JSON logs instead of running Prometheus server.

**Rationale:**
- Simpler architecture for v1
- Log aggregation tools (Datadog, Loki, CloudWatch) can query metric fields
- No separate metrics server to maintain
- Sufficient for production monitoring

**Trade-off:** Metrics lost on restart (acceptable - not a long-term metrics store).

### Thread-Safe In-Memory Storage

**Decision:** Lock-protected collections for metrics aggregation.

**Rationale:**
- Fast access from async endpoints
- Prevents race conditions from concurrent requests
- Simple implementation
- Metrics reset on restart acceptable for v1

### Database Health Check via SELECT 1

**Decision:** Use `SELECT 1` query to verify database connectivity.

**Rationale:**
- Fast (doesn't query tables)
- Works with any database
- Detects connection issues immediately
- Doesn't require specific schema

## Integration Points

**Dependencies:**
- `02-01` (Structured logging with Loguru) - Metrics logged as JSON
- Database module - `SessionLocal()` for health check
- Sentry module - User context set in `get_current_user()`

**Provides to:**
- Load balancers - `/health` endpoint for health checks
- Monitoring systems - Structured log fields for metrics aggregation
- Production ops - Visibility into system health and performance

## Requirements Mapped

- **OBS-05:** Health check endpoint with database connectivity
- **OBS-06:** Request metrics (latency, errors, throughput)
- **OBS-07:** Business metrics (grading operations, token usage)

## Testing

**Verification tests passed:**
```bash
# Health router imports
from api.health import router

# Metrics collector works
mc = get_metrics_collector()
mc.record_request('GET', '/health', 200, 45.2)
assert mc.get_error_rate() == 0.0

# Business metrics work
mc.record_grading_operation('session-123', 15000)
assert mc.get_business_metrics()['grading_operations'] == 1
```

**Manual testing (for production):**
1. Start application: `uvicorn src.api.app:app --reload`
2. Test health check: `curl http://localhost:8000/health`
3. Test health failure: Stop database, verify 503 response
4. Make requests, verify metrics logged in JSON logs
5. Check log output for "Metrics snapshot" entries

## Known Limitations

1. **Metrics reset on restart:** In-memory storage loses metrics on restart. For v1, this is acceptable. For v2, consider persistent metrics store or Prometheus.

2. **Token usage estimation:** Current implementation uses copy count * 10,000 as estimate. Should track actual token usage from LLM responses.

3. **No metrics endpoint:** Metrics are logged but not exposed via API endpoint. For v1, log aggregation is sufficient. For v2, consider `/metrics` endpoint.

## Next Steps

**Phase 3 (Core Grading Experience):**
- Metrics now available for performance analysis
- Grading operations tracked for capacity planning
- Token usage tracked for cost optimization

**Production readiness:**
- Health checks ready for load balancer integration
- Structured logs ready for log aggregation (Datadog, Loki, CloudWatch)
- Business metrics support capacity planning and cost analysis

## Commits

1. `ab4c686` - feat(02-03): add metrics collector utility
2. `d13d97a` - feat(02-03): add health check endpoint with database connectivity
3. `9c3dd48` - feat(02-03): integrate health check and metrics into app
4. `42731bc` - feat(02-03): add metrics recording to middleware and endpoints

## Summary

Successfully implemented health check endpoint and log-based metrics collection. The system now has production-ready health checks for load balancers and comprehensive metrics for request performance, error rates, and business operations. Metrics are emitted as structured JSON logs, enabling integration with log aggregation tools without the complexity of a separate metrics server.

All requirements from the plan were met with no deviations. The implementation follows the established patterns from Phase 2 (structured logging with Loguru, Sentry integration) and provides the foundation for production monitoring.
