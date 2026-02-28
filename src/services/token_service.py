"""
Token deduction service for grading operations.

This service provides token deduction functionality with:
- Idempotency based on session_id
- Token aggregation from AI providers
- Audit trail via usage_records table
- Transactional consistency with row locking
"""

from typing import Any, Dict
from uuid import uuid4

from loguru import logger
from sqlalchemy.exc import IntegrityError

from src.core.exceptions import AICorrectionError
from src.db.database import SessionLocal


# ==================== Custom Exceptions ====================

class InsufficientTokensError(AICorrectionError):
    """
    Raised when user doesn't have enough tokens for an operation.

    Attributes:
        tokens_required: Number of tokens needed
        tokens_remaining: Number of tokens user has left
    """

    def __init__(self, message: str, tokens_required: int = 0, tokens_remaining: int = 0):
        super().__init__(message, {
            'tokens_required': tokens_required,
            'tokens_remaining': tokens_remaining
        })
        self.tokens_required = tokens_required
        self.tokens_remaining = tokens_remaining


class UserNotFoundError(AICorrectionError):
    """
    Raised when a user cannot be found in the database.

    Attributes:
        user_id: The user ID that was not found
    """

    def __init__(self, user_id: str):
        super().__init__(f"User not found: {user_id}", {'user_id': user_id})
        self.user_id = user_id


class DeductionError(AICorrectionError):
    """
    Raised when token deduction fails due to database or system errors.

    This is a wrapper for technical errors during the deduction process.
    """
    pass


# ==================== Token Deduction Service ====================

class TokenDeductionService:
    """
    Service for deducting token usage from user balances.

    This service aggregates actual token usage from AI providers after grading
    completes and deducts from user balances with idempotency and audit trail.

    The service ensures:
    - No double-charging via session_id idempotency check
    - Accurate token counts from provider usage tracking
    - Sufficient balance validation before deduction
    - Full audit trail via usage_records table
    - Transactional consistency with database row locking
    """

    def deduct_grading_usage(
        self,
        user_id: str,
        provider: Any,
        session_id: str,
        db: SessionLocal
    ) -> Dict[str, Any]:
        """
        Deduct tokens used during a grading session.

        This method performs token deduction with the following steps:
        1. Check for existing usage record (idempotency)
        2. Aggregate token usage from provider
        3. Validate user has sufficient tokens
        4. Create usage record and deduct in single transaction

        Args:
            user_id: User ID to deduct tokens from
            provider: AI provider (BaseProvider or ComparisonProvider)
                      Must have get_token_usage() method
            session_id: Unique session identifier for idempotency
            db: Database session

        Returns:
            Dict with token deduction result:
            {
                "tokens_deducted": int,           # Number of tokens deducted
                "remaining_tokens": int,          # User's remaining balance
                "usage_record_id": str,           # ID of created/retrieved record
                "is_duplicate": bool              # True if this was a duplicate call
            }

        Raises:
            InsufficientTokensError: User doesn't have enough tokens
            UserNotFoundError: User doesn't exist in database
            DeductionError: Database or system error during deduction
        """
        # Import here to avoid circular dependencies
        from src.db.models import User, UsageRecord

        # 1. Check idempotency: already deducted for this session?
        existing = db.query(UsageRecord).filter(
            UsageRecord.session_id == session_id,
            UsageRecord.user_id == user_id
        ).first()

        if existing:
            logger.info(
                f"Session {session_id} already deducted for user {user_id}, "
                f"returning cached result (record: {existing.id})"
            )

            # Get fresh user data for accurate remaining count
            user = db.query(User).filter(User.id == user_id).first()

            return {
                "tokens_deducted": existing.total_tokens,
                "remaining_tokens": user.remaining_tokens if user else 0,
                "usage_record_id": existing.id,
                "is_duplicate": True
            }

        # 2. Get actual usage from provider
        usage = provider.get_token_usage()
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cached_tokens = usage.get("cached_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        logger.debug(
            f"Token usage for session {session_id}: "
            f"prompt={prompt_tokens}, completion={completion_tokens}, "
            f"cached={cached_tokens}, total={total_tokens}"
        )

        # 3. If no tokens used, return early
        if total_tokens == 0:
            logger.info(f"No tokens to deduct for session {session_id}")
            return {
                "tokens_deducted": 0,
                "remaining_tokens": 0,
                "usage_record_id": None,
                "is_duplicate": False
            }

        # 4. Lock user row and verify sufficient tokens
        user = db.query(User).filter(User.id == user_id).with_for_update().first()

        if not user:
            raise UserNotFoundError(user_id)

        if not user.can_use_tokens(total_tokens):
            raise InsufficientTokensError(
                f"Insufficient tokens for grading session {session_id}",
                tokens_required=total_tokens,
                tokens_remaining=user.remaining_tokens
            )

        # 5. Create usage record (audit trail)
        record_id = str(uuid4())
        record = UsageRecord(
            id=record_id,
            user_id=user_id,
            session_id=session_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens
        )

        # 6. Deduct from user balance
        user.add_token_usage(total_tokens)

        # 7. Commit transaction
        try:
            db.add(record)
            db.commit()
            db.refresh(user)

            logger.info(
                f"Deducted {total_tokens} tokens from user {user_id} "
                f"for session {session_id}, remaining: {user.remaining_tokens}"
            )

        except IntegrityError as e:
            db.rollback()
            logger.error(
                f"Database integrity error during deduction for user {user_id}, "
                f"session {session_id}: {e}"
            )
            raise DeductionError("Failed to commit token deduction") from e

        except Exception as e:
            db.rollback()
            logger.error(
                f"Unexpected error during deduction for user {user_id}, "
                f"session {session_id}: {e}"
            )
            raise DeductionError(f"Token deduction failed: {str(e)}") from e

        # 8. Return result
        return {
            "tokens_deducted": total_tokens,
            "remaining_tokens": user.remaining_tokens,
            "usage_record_id": record_id,
            "is_duplicate": False
        }
