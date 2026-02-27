"""Database models for La Corrigeuse."""

from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Boolean, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from db.database import Base
import enum
import uuid


class SubscriptionTier(str, enum.Enum):
    """Subscription tiers for users."""
    FREE = "free"              # Découverte - 10K tokens (~1 page)
    ESSENTIEL = "essentiel"    # 1.2M tokens (~120 pages)
    PRO = "pro"                # 6M tokens (~600 pages)
    MAX = "max"                # 24M tokens (~2400 pages)
    ADMIN = "admin"            # Admin - unlimited, access to settings


class User(Base):
    """User model for authentication and subscription tracking."""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Subscription
    subscription_tier = Column(
        SQLEnum(SubscriptionTier),
        default=SubscriptionTier.FREE,
        nullable=False
    )
    subscription_start = Column(DateTime, nullable=True)
    subscription_end = Column(DateTime, nullable=True)

    # Usage tracking (in tokens)
    tokens_used_this_month = Column(Integer, default=0)
    usage_month = Column(Integer, default=lambda: datetime.utcnow().month)
    usage_year = Column(Integer, default=lambda: datetime.utcnow().year)

    def get_monthly_token_limit(self) -> int:
        """Get the monthly token limit for the user's tier."""
        limits = {
            SubscriptionTier.FREE: 10_000,           # ~1 page
            SubscriptionTier.ESSENTIEL: 1_200_000,   # ~120 pages
            SubscriptionTier.PRO: 6_000_000,         # ~600 pages
            SubscriptionTier.MAX: 24_000_000,        # ~2400 pages
            SubscriptionTier.ADMIN: 999_999_999,     # Unlimited
        }
        return limits[self.subscription_tier]

    # Legacy aliases for backward compatibility
    def get_monthly_limit(self) -> int:
        """Alias for get_monthly_token_limit()."""
        return self.get_monthly_token_limit()

    def can_use_tokens(self, token_count: int = 1) -> bool:
        """Check if user can use the specified number of tokens."""
        self._reset_usage_if_new_month()
        return self.tokens_used_this_month + token_count <= self.get_monthly_token_limit()

    def can_grade_copies(self, estimated_tokens: int = 10_000) -> bool:
        """Legacy alias - checks if user has enough tokens for estimated usage."""
        return self.can_use_tokens(estimated_tokens)

    def add_token_usage(self, token_count: int) -> None:
        """Add token usage to the counter."""
        self._reset_usage_if_new_month()
        self.tokens_used_this_month += token_count

    def increment_usage(self, token_count: int = 10_000) -> None:
        """Legacy alias - increments token usage."""
        self.add_token_usage(token_count)

    def _reset_usage_if_new_month(self) -> None:
        """Reset usage counter if we're in a new month."""
        now = datetime.utcnow()
        if now.month != self.usage_month or now.year != self.usage_year:
            self.usage_month = now.month
            self.usage_year = now.year
            self.tokens_used_this_month = 0

    @property
    def remaining_tokens(self) -> int:
        """Get remaining tokens for the current month."""
        self._reset_usage_if_new_month()
        return max(0, self.get_monthly_token_limit() - self.tokens_used_this_month)

    @property
    def remaining_copies(self) -> int:
        """Legacy alias - returns remaining tokens."""
        return self.remaining_tokens

    def to_dict(self) -> dict:
        """Convert user to dictionary (without sensitive data)."""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "subscription_tier": self.subscription_tier.value,
            "tokens_used_this_month": self.tokens_used_this_month,
            "monthly_token_limit": self.get_monthly_token_limit(),
            "remaining_tokens": self.remaining_tokens,
            # Legacy aliases for backward compatibility
            "copies_used_this_month": self.tokens_used_this_month,
            "monthly_limit": self.get_monthly_token_limit(),
            "remaining_copies": self.remaining_tokens,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class PasswordResetToken(Base):
    """Password reset token for secure password recovery."""
    __tablename__ = "password_reset_tokens"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    token_hash = Column(String, nullable=False, unique=True, index=True)  # Hashed token
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    used = Column(Boolean, default=False)
