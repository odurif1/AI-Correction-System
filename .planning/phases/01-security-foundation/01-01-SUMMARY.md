---
phase: 01-security-foundation
plan: 01
subsystem: Authentication Security
tags: [security, jwt, environment-variables, configuration-validation]
requirements: [SEC-01, SEC-02, SEC-03, AUTH-02, AUTH-04]

dependency_graph:
  requires:
    - phase_01_security_foundation_01_01: "None - first task in security foundation"
  provides:
    - phase_01_security_foundation_01_02: "Environment-based JWT secret and API keys for rate limiting setup"
    - phase_01_security_foundation_01_03: "Secure configuration foundation for data isolation implementation"
    - phase_01_security_foundation_01_04: "Validated settings for input validation framework"
    - phase_01_security_foundation_01_05: "Secure auth module for password policy enforcement"
    - phase_01_security_foundation_01_06: "Startup validation pattern for security headers middleware"
  affects:
    - src/api/auth.py: "JWT signing/verification now uses environment secret"
    - src/config/settings.py: "Security-critical configuration with validation"
    - src/api/app.py: "Startup validation prevents insecure launches"

tech_stack:
  added:
    - "Pydantic field_validator: Custom validation for default value rejection"
    - "Pydantic ValidationError: Proper exception type for startup validation"
  patterns:
    - "Environment-based configuration with fail-fast startup validation"
    - "Required field validation with min_length constraints"
    - "Dynamic secret fetching via get_settings() singleton"
    - "Startup event handlers for security validation"

key_files:
  created:
    - "None - no new files created, only modifications"
  modified:
    - "src/config/settings.py: Added jwt_secret field with validators, API key validation"
    - "src/api/auth.py: Removed hardcoded SECRET_KEY, JWT operations use get_settings()"
    - "src/api/app.py: Added ValidationError import, startup event validates settings"

decisions:
  - title: "Use Pydantic field_validator for default value rejection"
    rationale: "Declarative validation integrates with Settings initialization, fail-fast on startup"
    alternatives: ["Manual validation in startup event", "Custom Settings class", "secrets library"]
    impact: "Application exits with clear error if JWT_SECRET is default/weak"

  - title: "Dynamic secret fetching in JWT operations"
    rationale: "Allows settings reload without restart, avoids stale SECRET_KEY references"
    alternatives: ["Module-level SECRET_KEY constant", "Secret cached at import time"]
    impact: "JWT creation/verification always uses current environment value"

  - title: "Single startup event for validation"
    rationale: "Database init already runs on startup, security validation should happen first"
    alternatives: ["Separate validation event", "Decorator-based validation on routes"]
    impact: "Application exits before accepting requests if secrets are invalid"

metrics:
  duration_seconds: 125
  tasks_completed: 3
  files_modified: 3
  commits: 3
  completed_date: "2026-02-26T23:13:17Z"
---

# Phase 1 Plan 1: Environment-Based Security Configuration Summary

**JWT secret and API keys loaded from environment variables with startup validation and fail-fast behavior.**

## Overview

Migrated hardcoded JWT secret and API keys from source code to environment variables with Pydantic Settings validation. The application now fails to start if critical security settings are missing or use default values. This prevents token forgery attacks and ensures secrets are properly configured before accepting any requests.

## Tasks Completed

### Task 1: Add JWT Secret Field to Settings with Validation

**Files:** `src/config/settings.py`

**Changes:**
- Added `jwt_secret: str = Field(..., min_length=32)` - required field with 32-character minimum
- Added `@field_validator('jwt_secret')` to reject default/weak values:
  - Rejects: `your-secret-key-change-in-production`, `secret`, `test`, `password`, `change-me`, `default-secret`
  - Provides helpful error message: `"JWT_SECRET cannot be a default value. Generate with: openssl rand -base64 32"`
- Added `@field_validator('ai_provider')` to ensure required API key is set:
  - Validates that `AI_CORRECTION_{PROVIDER}_API_KEY` is set for the configured provider
  - Example: If `ai_provider="gemini"`, then `gemini_api_key` must be non-empty

**Commit:** `1bbb533` - feat(01-01): add JWT secret field with validation to Settings

### Task 2: Update Auth Module to Use Settings

**Files:** `src/api/auth.py`

**Changes:**
- Removed hardcoded `SECRET_KEY = "your-secret-key-change-in-production"`
- Added import: `from config.settings import get_settings`
- Updated `create_access_token()` to use `settings.jwt_secret` dynamically
- Updated `decode_token()` to use `settings.jwt_secret` dynamically
- Preserved existing bcrypt password hashing (passlib CryptContext) - no changes needed per AUTH-02
- Preserved 7-day token expiration (`ACCESS_TOKEN_EXPIRE_HOURS = 24 * 7`) - unchanged per CONTEXT.md

**Verification:** No hardcoded `SECRET_KEY` remains in auth.py. All JWT operations use `get_settings().jwt_secret`.

**Commit:** `163ae85` - feat(01-01): update auth module to use environment-based JWT secret

### Task 3: Add Startup Validation Event

**Files:** `src/api/app.py`

**Changes:**
- Added import: `from pydantic import ValidationError`
- Updated `@app.on_event("startup")` handler:
  - Calls `get_settings()` first (before database initialization)
  - Logs successful validation: `"Security configuration validated. Provider: {provider}"`
  - Catches `ValidationError` and logs error details
  - Exits with `SystemExit(1)` if validation fails
- Placement: Validation runs before database init, preventing insecure application launch

**Verification:** Startup event exists and validates settings on application launch.

**Commit:** `afc68e6` - feat(01-01): add startup validation for security settings

## Deviations from Plan

### Auto-fixed Issues

**None.** Plan executed exactly as written. All tasks completed without deviations or unexpected issues.

## Environment Variables Added

| Variable | Required | Min Length | Validation |
|----------|----------|------------|------------|
| `AI_CORRECTION_JWT_SECRET` | Yes | 32 characters | Rejects default/weak values |
| `AI_CORRECTION_GEMINI_API_KEY` | Conditional* | - | Required if `AI_CORRECTION_AI_PROVIDER=gemini` |
| `AI_CORRECTION_OPENAI_API_KEY` | Conditional* | - | Required if `AI_CORRECTION_AI_PROVIDER=openai` |
| `AI_CORRECTION_GLM_API_KEY` | Conditional* | - | Required if `AI_CORRECTION_AI_PROVIDER=glm` |
| `AI_CORRECTION_OPENROUTER_API_KEY` | Conditional* | - | Required if `AI_CORRECTION_AI_PROVIDER=openrouter` |

*API keys are required based on the configured `AI_CORRECTION_AI_PROVIDER` value.

## Validation Rules Implemented

1. **JWT Secret Validation:**
   - Required field - application exits if not set
   - Minimum 32 characters (prevents weak secrets)
   - Rejects common default values (preventable security mistakes)
   - Clear error message with generation command

2. **API Key Validation:**
   - Ensures API key exists for the configured AI provider
   - Prevents runtime errors when making LLM API calls
   - Validates on startup before any requests are processed

## Error Messages on Startup Failure

### Missing JWT_SECRET
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for Settings
jwt_secret
  Field required [type=missing, input_value={}, input_type=dict]
```

### Default/Weak JWT_SECRET
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for Settings
jwt_secret
  JWT_SECRET cannot be a default value. Generate with: openssl rand -base64 32
```

### Missing API Key for Configured Provider
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for Settings
ai_provider
  AI_CORRECTION_GEMINI_API_KEY required when provider=gemini
```

## Success Criteria Met

- [x] Application exits on startup with clear error message if `JWT_SECRET` is not set
- [x] Application exits on startup if `JWT_SECRET` is a default/weak value
- [x] Application exits on startup if API key is missing for configured AI provider
- [x] JWT tokens created after changes use environment variable secret
- [x] Existing password hashing with bcrypt continues to work (no changes)
- [x] Token expiration remains at 7 days

## Manual Verification Steps

To verify the implementation:

1. **Test missing JWT_SECRET:**
   ```bash
   unset AI_CORRECTION_JWT_SECRET
   python -m src.main
   # Expected: Application exits with ValidationError
   ```

2. **Test default JWT_SECRET value:**
   ```bash
   export AI_CORRECTION_JWT_SECRET="secret"
   python -m src.main
   # Expected: Application exits with "cannot be a default value" error
   ```

3. **Test successful startup with valid settings:**
   ```bash
   export AI_CORRECTION_JWT_SECRET="$(openssl rand -base64 32)"
   export AI_CORRECTION_AI_PROVIDER="gemini"
   export AI_CORRECTION_GEMINI_API_KEY="your-api-key"
   python -m src.main
   # Expected: Application starts, logs "Security configuration validated"
   ```

4. **Verify JWT tokens use environment secret:**
   ```python
   from src.api.auth import create_access_token, decode_token
   from src.config.settings import get_settings

   token = create_access_token("user123")
   payload = decode_token(token)
   # Token should decode successfully using get_settings().jwt_secret
   ```

## Implementation Notes

### Pydantic Settings Pattern

The implementation uses Pydantic Settings with field validators for declarative validation:

```python
class Settings(BaseSettings):
    jwt_secret: str = Field(..., min_length=32)  # Required, min 32 chars

    @field_validator('jwt_secret')
    @classmethod
    def reject_default_values(cls, v: str) -> str:
        forbidden = ['your-secret-key-change-in-production', 'secret', ...]
        if v.lower() in forbidden:
            raise ValueError("...")
        return v
```

This pattern ensures:
- Validation happens on Settings instantiation
- Error messages are clear and actionable
- No manual validation code needed in business logic

### Startup Event Ordering

The startup event validates settings **before** database initialization:

```python
@app.on_event("startup")
async def startup_event():
    # 1. Security validation (NEW)
    try:
        settings = get_settings()
        logger.info(f"Security configuration validated")
    except ValidationError as e:
        logger.error(f"Configuration error: {e}")
        raise SystemExit(1)

    # 2. Database initialization (EXISTING)
    from db import init_db
    init_db()
```

This ensures the application fails fast before accepting any requests.

### Dynamic Secret Fetching

JWT operations fetch the secret dynamically rather than caching at import time:

```python
def create_access_token(user_id: str) -> str:
    settings = get_settings()  # Fetches current environment value
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)
```

This allows for potential future settings reload without restart.

## Next Steps

Subsequent plans in Phase 1 build on this secure configuration foundation:

- **Plan 01-02:** Rate limiting using validated configuration
- **Plan 01-03:** Multi-tenant data isolation (JWT-based user ID injection)
- **Plan 01-04:** Input validation framework extending Pydantic patterns
- **Plan 01-05:** Password policy enforcement using validated auth module
- **Plan 01-06:** Security headers middleware (uses startup validation pattern)

## Security Impact

### Before (Vulnerabilities)
- JWT secret hardcoded in source: `SECRET_KEY = "your-secret-key-change-in-production"`
- Application would start successfully with default secret
- Attackers could forge tokens and authenticate as any user
- No validation that API keys were configured

### After (Secure)
- JWT secret required from environment variable: `AI_CORRECTION_JWT_SECRET`
- Application exits on startup if secret is missing, too short, or default value
- API key validation ensures provider is configured
- Dynamic secret fetching prevents stale credentials

**Risk Reduction:** Critical security vulnerability (hardcoded JWT secret) eliminated. Token forgery attacks prevented.

---

*Summary completed: 2026-02-26*
*Phase: 01-security-foundation, Plan: 01*
*Execution time: 125 seconds*
*Commits: 3*
