"""Database models."""

from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, Integer, ForeignKey, Enum as SQLEnum, UniqueConstraint
from sqlalchemy.orm import relationship
from db.database import Base
import enum
import uuid


class SubscriptionTier(str, enum.Enum):
    """Subscription tiers for users."""
    FREE = "free"              # Découverte - 100K tokens one-shot (~30 pages)
    ESSENTIEL = "essentiel"    # 1.2M tokens/month (~500 copies)
    PRO = "pro"                # 6M tokens/month (~2500 copies)
    MAX = "max"                # 24M tokens/month (~10000 copies)
    ADMIN = "admin"            # Admin - unlimited, access to settings


class User(Base):
    """User model for authentication and subscription tracking."""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

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
    usage_month = Column(Integer, default=lambda: datetime.now(timezone.utc).month)
    usage_year = Column(Integer, default=lambda: datetime.now(timezone.utc).year)

    def get_monthly_token_limit(self) -> int:
        """Get the monthly token limit for the user's tier."""
        limits = {
            SubscriptionTier.FREE: 100_000,           # One-shot (~30 pages)
            SubscriptionTier.ESSENTIEL: 1_200_000,    # ~500 copies/month
            SubscriptionTier.PRO: 6_000_000,          # ~2500 copies/month
            SubscriptionTier.MAX: 24_000_000,         # ~10000 copies/month
            SubscriptionTier.ADMIN: 999_999_999,      # Unlimited
        }
        return limits[self.subscription_tier]

    def can_use_tokens(self, token_count: int = 1) -> bool:
        """Check if user can use the specified number of tokens."""
        self._reset_usage_if_new_month()
        return self.tokens_used_this_month + token_count <= self.get_monthly_token_limit()

    def add_token_usage(self, token_count: int) -> None:
        """Add token usage to the counter."""
        self._reset_usage_if_new_month()
        self.tokens_used_this_month += token_count

    def has_monthly_reset(self) -> bool:
        """Check if user's tier resets monthly (FREE is one-shot only)."""
        return self.subscription_tier != SubscriptionTier.FREE

    def _reset_usage_if_new_month(self) -> None:
        """Reset usage counter if we're in a new month (except FREE tier)."""
        # FREE tier is one-shot, never reset
        if self.subscription_tier == SubscriptionTier.FREE:
            return

        now = datetime.now(timezone.utc)
        if now.month != self.usage_month or now.year != self.usage_year:
            self.usage_month = now.month
            self.usage_year = now.year
            self.tokens_used_this_month = 0

    @property
    def remaining_tokens(self) -> int:
        """Get remaining tokens for the current month."""
        self._reset_usage_if_new_month()
        return max(0, self.get_monthly_token_limit() - self.tokens_used_this_month)

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
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class UsageRecord(Base):
    """Token usage record for audit trail and reconciliation."""
    __tablename__ = "usage_records"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)

    # Token breakdown (stored for cost analysis)
    prompt_tokens = Column(Integer, nullable=False)
    completion_tokens = Column(Integer, nullable=False)
    cached_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, nullable=False)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", backref="usage_records")

    # Unique constraint for idempotency (one record per session per user)
    __table_args__ = (
        UniqueConstraint('user_id', 'session_id', name='uq_usage_record_session'),
    )
