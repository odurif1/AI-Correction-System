"""Public request context helpers for the backend-only audit surface."""

from fastapi import Depends
from sqlalchemy.orm import Session

from db import get_db, User, SubscriptionTier

PUBLIC_USER_ID = "public-user"
PUBLIC_USER_EMAIL = "public@example.com"


def ensure_public_user(db: Session) -> User:
    """Return the shared technical user used by the public backend."""
    user = db.query(User).filter(User.id == PUBLIC_USER_ID).first()
    if user is not None:
        if user.subscription_tier != SubscriptionTier.ADMIN:
            user.subscription_tier = SubscriptionTier.ADMIN
            db.commit()
            db.refresh(user)
        return user

    user = User(
        id=PUBLIC_USER_ID,
        email=PUBLIC_USER_EMAIL,
        password_hash="disabled",
        name="Public API",
        subscription_tier=SubscriptionTier.ADMIN,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


async def get_current_user(db: Session = Depends(get_db)) -> User:
    """Provide the shared public user for all API requests."""
    return ensure_public_user(db)


async def get_optional_user(db: Session = Depends(get_db)) -> User:
    """Return the same shared public user for compatibility."""
    return ensure_public_user(db)


async def get_admin_user(db: Session = Depends(get_db)) -> User:
    """The public backend exposes a single technical admin user."""
    return ensure_public_user(db)
