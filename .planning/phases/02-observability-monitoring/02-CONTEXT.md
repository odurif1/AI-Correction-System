# Phase 2: Observability & Monitoring - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

System behavior is visible through logs, metrics, and error tracking for production debugging. Includes password reset functionality via email.

</domain>

<decisions>
## Implementation Decisions

### Password Reset
- **Email provider:** SendGrid (100 emails/day free tier)
- **Token expiration:** 30 minutes
- **Post-reset behavior:** Auto-login (user is logged in after successful reset)
- **Sender address:** noreply@lacorrigeuse.fr
- Token stored in database with hashed value (security)
- Email template: Simple text email with reset link

### Logging Format
- **Format:** Structured JSON logs
- **Correlation ID:** X-Request-ID header, included in all logs
- **Fields:** timestamp, level, request_id, method, path, status_code, latency_ms, user_id (if authenticated)
- **Levels:** DEBUG (dev only), INFO (requests), WARNING (validation errors), ERROR (exceptions)
- **Sensitive data:** Never log passwords, tokens, or API keys

### Error Tracking
- **Provider:** Sentry (free tier: 5K errors/month)
- **Capture:** Automatic for uncaught exceptions
- **Context:** Include user_id, request_id, request path
- **Sampling:** 100% in dev, 10% in production (avoid quota limits)
- **User feedback:** Basic error message to user, full details to Sentry

### Health & Metrics
- **Health endpoint:** GET /health returns JSON with status, version, database connection
- **Response format:** `{"status": "healthy", "version": "x.y.z", "database": "connected"}`
- **Metrics storage:** Log-based (no Prometheus/external DB for v1)
- **Request metrics:** Latency percentiles (p50, p95, p99), error rate, requests per minute
- **Business metrics:** Grading operations count, token usage per phase, active sessions count

### Claude's Discretion
- Exact log format structure
- Sentry integration details (DSN configuration)
- Health check timeout handling
- Metrics aggregation approach

</decisions>

<specifics>
## Specific Ideas

- Standard observability patterns - no special requirements
- Keep it simple for v1 - can enhance later

</specifics>

<deferred>
## Deferred Ideas

- None - discussion stayed within phase scope

</deferred>

---

*Phase: 02-observability-monitoring*
*Context gathered: 2026-02-27*
