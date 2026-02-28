# Phase 6: Token Deduction Fix - Research

**Researched:** 2026-02-28
**Domain:** Token Accounting & Idempotency
**Confidence:** HIGH

## Summary

This phase fixes a critical bug where users are charged ~30 tokens (copy count) instead of ~150,000 tokens (actual usage). The root cause is in `src/api/app.py:1281` where `len(session.graded_copies)` is passed to `User.add_token_usage()` instead of the actual token count from providers.

The fix requires: (1) creating a `TokenDeductionService` to aggregate tokens from AI providers after grading completes, (2) adding a `usage_records` table for audit trail, and (3) wiring the service into the grading completion flow with idempotency to prevent double-charging.

**Primary recommendation:** This is a code-only fix—no new libraries needed. Token tracking already exists in `BaseProvider._log_call()` and `ComparisonProvider.get_token_usage()`. The missing piece is the aggregation layer that deducts actual tokens from user balances.

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Deduction Timing:** Deduct tokens **after grading completes** (not per-copy)
  - Simpler implementation: one DB write per session
  - No partial charges if grading fails midway

- **Failure Handling:** **No charge for partial work** — only deduct if grading completes successfully
  - User never pays for failed/incomplete grading sessions

- **Audit Trail Storage:** **New `usage_records` table** with dedicated schema
  - Clean separation from user table, queryable for reports and debugging

- **Idempotency:** **Check `usage_records` for existing `session_id`** before creating new record
  - Prevents double-charging on retries or re-requests

- **Service Location:** **Dedicated service class** in `src/services/token_service.py`
  - Clean separation, testable, reusable across API and CLI

- **Cached Tokens:** **Count all tokens** including cached tokens from Gemini context caching
  - Cached tokens still represent compute resources consumed

- **Data Model:** **Full breakdown**: store `prompt_tokens`, `completion_tokens`, `cached_tokens`, and `total`
  - Enables future cost analysis and optimization insights

### Claude's Discretion

None—all implementation decisions are locked.

### Deferred Ideas (OUT OF SCOPE)

- Token estimation before grading (v1.2+)
- Usage notification system (v1.2+)
- Subscription UX polish (Phase 7)

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| **SQLAlchemy** | bundled (from pyproject.toml) | ORM for usage_records model | Already installed, provides declarative base for new table |
| **sqlite3** | bundled (Python stdlib) | Database for usage_records | Current database, sufficient for audit trail |
| **FastAPI BackgroundTasks** | 0.115.0+ | Async deduction after grading | Already installed, used for existing grading flow |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **loguru** | 0.7.3+ | Structured logging for deduction events | Already installed, use for all service logging |
| **pydantic** | 2.9.0+ | Result models (TokenDeductionResult) | Already installed, type validation |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| SQLAlchemy | Raw SQL | More control, but breaks ORM pattern used elsewhere |
| Service class | Inline function in app.py | Service is testable, reusable for CLI |
| usage_records table | Store in User JSON column | Breaks normalization, can't query efficiently |

**Installation:**
```bash
# No new packages required
```

## Architecture Patterns

### Recommended Project Structure

```
src/
├── services/
│   ├── __init__.py
│   └── token_service.py        # NEW: TokenDeductionService
├── db/
│   ├── models.py               # MODIFY: Add UsageRecord model
│   └── database.py             # NO CHANGE
├── api/
│   └── app.py                  # MODIFY: Replace buggy call with service
└── ai/
    ├── base_provider.py        # NO CHANGE: get_token_usage() already exists
    └── comparison_provider.py  # NO CHANGE: Aggregates both LLMs
```

### Pattern 1: Transactional Deduction with Idempotency

**What:** Wrap deduction in database transaction with row locking, check for existing session_id before creating new record

**When to use:** All token deduction operations

**Example:**
```python
# Source: Based on existing User model patterns in src/db/models.py
from sqlalchemy import exc
from loguru import logger

class TokenDeductionService:
    def deduct_grading_usage(
        self,
        user_id: str,
        provider: BaseProvider,
        session_id: str,
        db: Session
    ) -> TokenDeductionResult:
        """Deduct tokens with transactional consistency and idempotency."""

        # 1. Check idempotency: already deducted for this session?
        existing = db.query(UsageRecord).filter(
            UsageRecord.session_id == session_id,
            UsageRecord.user_id == user_id
        ).first()

        if existing:
            logger.info(f"Session {session_id} already deducted, returning cached")
            return TokenDeductionResult(
                tokens_deducted=existing.total_tokens,
                remaining_tokens=db.query(User).get(user_id).remaining_tokens,
                usage_record_id=existing.id,
                is_duplicate=True
            )

        # 2. Get actual usage from provider
        usage = provider.get_token_usage()
        total_tokens = usage["prompt_tokens"] + usage["completion_tokens"]

        if total_tokens == 0:
            return TokenDeductionResult(tokens_deducted=0)

        # 3. Lock user row and verify sufficient tokens
        user = db.query(User).filter(User.id == user_id).with_for_update().first()
        if not user:
            raise UserNotFoundError(user_id)

        if not user.can_use_tokens(total_tokens):
            raise InsufficientTokensError(
                tokens_required=total_tokens,
                tokens_remaining=user.remaining_tokens
            )

        # 4. Create usage record (audit trail)
        record = UsageRecord(
            user_id=user_id,
            session_id=session_id,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            cached_tokens=usage.get("cached_tokens", 0),
            total_tokens=total_tokens
        )

        # 5. Deduct from user balance
        user.add_token_usage(total_tokens)

        # 6. Commit transaction
        try:
            db.add(record)
            db.commit()
            db.refresh(user)
        except exc.IntegrityError as e:
            db.rollback()
            logger.error(f"Failed to commit deduction: {e}")
            raise DeductionError("Database error during deduction")

        # 7. Return result
        return TokenDeductionResult(
            tokens_deducted=total_tokens,
            remaining_tokens=user.remaining_tokens,
            usage_record_id=record.id
        )
```

### Pattern 2: Post-Grading Deduction Integration

**What:** Deduct tokens only after grading completes successfully in the background task

**When to use:** API grading completion flow

**Example:**
```python
# Source: Modified from src/api/app.py:1232-1284
async def grade_task():
    from db import SessionLocal, User
    from services.token_service import TokenDeductionService

    try:
        # 1. Run grading (existing code)
        await orchestrator.analyze_only()
        orchestrator.confirm_scale(orchestrator.question_scales)
        await orchestrator.grade_all(progress_callback=progress_callback)

        # 2. Deduct actual tokens (NEW)
        db = SessionLocal()
        try:
            deduction_svc = TokenDeductionService()
            result = deduction_svc.deduct_grading_usage(
                user_id=current_user.id,
                provider=orchestrator.ai,  # Has get_token_usage()
                session_id=session_id,
                db=db
            )

            logger.info(
                f"Deducted {result.tokens_deducted} tokens "
                f"from user {current_user.id} for session {session_id}"
            )

            # Reload user to get updated balance
            db_user = db.query(User).filter(User.id == current_user.id).first()

            # Broadcast completion with usage info
            await ws_manager.broadcast_event(session_id, PROGRESS_EVENT_SESSION_COMPLETE, {
                "average_score": avg,
                "total_copies": len(session.graded_copies),
                "tokens_used": result.tokens_deducted,
                "remaining_tokens": db_user.remaining_tokens
            })

        finally:
            db.close()

    except InsufficientTokensError as e:
        # Grading succeeded but user can't afford it
        await ws_manager.broadcast_event(session_id, PROGRESS_EVENT_SESSION_ERROR, {
            "error": "Insufficient tokens for grading",
            "tokens_required": e.tokens_required,
            "tokens_remaining": e.tokens_remaining
        })
    except Exception as e:
        logger.error(f"Grading error: {e}")
        # ... existing error handling ...
```

### Anti-Patterns to Avoid

- **Per-call database deduction:** Deducting tokens to database on every LLM call is slow and complex to rollback
- **Deduction before completion:** Never deduct tokens before grading completes—use post-grading deduction instead
- **Ignoring cached tokens:** Track prompt, completion, and cached tokens separately for accurate cost accounting

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Token tracking | Custom counter in app.py | `BaseProvider.get_token_usage()` | Already implemented, O(1) aggregation |
| Idempotency check | Custom Redis key | Database query on `usage_records.session_id` | Simpler, no new dependency |
| Transaction handling | Manual commit/rollback | SQLAlchemy session context manager | Automatic rollback on exception |

**Key insight:** The token tracking infrastructure already exists at the provider level. This phase is purely about wiring that data to the user balance with proper accounting.

## Common Pitfalls

### Pitfall 1: Double-Charging from Retries

**What goes wrong:** If grading task restarts or WebSocket reconnection triggers completion event again, tokens are deducted twice for the same session.

**Why it happens:** No idempotency check—deduction always runs regardless of previous attempts.

**How to avoid:** Query `usage_records` for existing `session_id` before creating new record. Return cached result if found.

**Warning signs:** Users report tokens deducted twice for same grading session, or `usage_records` has duplicate `session_id` entries.

### Pitfall 2: Race Condition on Token Balance

**What goes wrong:** Two concurrent grading requests both pass `can_use_tokens()` check, then both deduct. User exceeds monthly limit.

**Why it happens:** Read-then-write pattern without locking allows both requests to see the same balance.

**How to avoid:** Use `with_for_update()` to lock user row during deduction, ensuring check and deduct happen atomically.

**Warning signs:** Database shows `tokens_used_this_month > monthly_token_limit`.

### Pitfall 3: Aggregating Wrong Provider

**What goes wrong:** Single LLM mode aggregates tokens correctly, but dual-LLM mode only gets tokens from one provider.

**Why it happens:** Code assumes `orchestrator.ai` is always a `BaseProvider`, but dual mode uses `ComparisonProvider`.

**How to avoid:** Check provider type. If `ComparisonProvider`, it already aggregates both LLMs in `get_token_usage()`.

**Warning signs:** Dual-LLM grading shows suspiciously low token counts.

### Pitfall 4: Ignoring Cached Tokens

**What goes wrong:** Cached tokens from Gemini context caching are excluded from deduction, under-reporting actual costs.

**Why it happens:** Assuming only prompt + completion tokens count, but cached tokens also represent compute resources.

**How to avoid:** Always include `cached_tokens` in total calculation: `prompt + completion + cached`.

**Warning signs:** User balance decreases slower than expected compared to LLM API costs.

## Code Examples

Verified patterns from existing codebase:

### Get Token Usage from Provider

```python
# Source: src/ai/base_provider.py:273-282
def get_token_usage(self) -> Dict[str, int]:
    """Get total token usage from all calls (O(1))."""
    cached = getattr(self, '_total_cached_tokens', 0)
    return {
        "prompt_tokens": self._total_prompt_tokens,
        "completion_tokens": self._total_completion_tokens,
        "cached_tokens": cached,
        "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
        "calls": len(self.call_history)
    }
```

### Comparison Provider Aggregates Both LLMs

```python
# Source: src/ai/comparison_provider.py:184-205
def get_token_usage(self) -> Dict[str, Any]:
    """Get total token usage from all providers."""
    total_prompt = 0
    total_completion = 0
    total_cached = 0
    provider_usage = {}

    for name, provider in self.providers:
        if hasattr(provider, 'get_token_usage'):
            usage = provider.get_token_usage()
            provider_usage[name] = usage
            total_prompt += usage.get('prompt_tokens', 0)
            total_completion += usage.get('completion_tokens', 0)
            total_cached += usage.get('cached_tokens', 0)

    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "cached_tokens": total_cached,
        "total_tokens": total_prompt + total_completion,
        "by_provider": provider_usage
    }
```

### User Model Token Methods

```python
# Source: src/db/models.py:74-81
def add_token_usage(self, token_count: int) -> None:
    """Add token usage to the counter."""
    self._reset_usage_if_new_month()
    self.tokens_used_this_month += token_count

def can_use_tokens(self, token_count: int = 1) -> bool:
    """Check if user can use the specified number of tokens."""
    self._reset_usage_if_new_month()
    return self.tokens_used_this_month + token_count <= self.get_monthly_token_limit()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Deduct copy count | Deduct actual tokens | v1.1 (this phase) | Fixes billing bug |
| No audit trail | usage_records table | v1.1 (this phase) | Enables reconciliation |
| Non-idempotent | Idempotent by session_id | v1.1 (this phase) | Prevents double-charging |

**Deprecated/outdated:**
- `increment_usage(len(session.graded_copies))`: The buggy pattern being replaced

## Open Questions

None—all implementation details are resolved from existing research.

## Validation Architecture

> nyquist_validation is not enabled in .planning/config.json, skipping this section

## Sources

### Primary (HIGH confidence)

- **Internal codebase analysis** (2026-02-28)
  - `src/api/app.py:1268-1284` — Bug location and current grading flow
  - `src/ai/base_provider.py:216-283` — Token tracking implementation
  - `src/ai/comparison_provider.py:184-205` — Multi-provider aggregation
  - `src/db/models.py:20-126` — User model with token methods
  - `.planning/research/STACK_TOKENS.md` — Token deduction fix research
  - `.planning/research/ARCHITECTURE.md` — TokenDeductionService architecture
  - `.planning/research/PITFALLS.md` — Race condition and idempotency pitfalls

### Secondary (MEDIUM confidence)

- **FastAPI + SQLAlchemy transaction patterns** (2025-2026)
  - [FastAPI Best Practices Repository](https://gitee.com/fastapi-practices/fastapi_best_architecture/members) — Transaction patterns
  - [FastAPI Production-Ready Templates](https://github.com/xfstudio/skills) — Database session patterns

### Tertiary (LOW confidence)

- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - No new libraries needed, all existing
- Architecture: HIGH - Based on existing codebase patterns and research
- Pitfalls: HIGH - Documented from previous research and code analysis

**Research date:** 2026-02-28
**Valid until:** 2026-03-31 (30 days - stable domain, no external dependencies)
