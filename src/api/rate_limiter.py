"""
Rate limiter module for the API.

Extracted from app.py to avoid circular imports when auth.py
needs to apply rate limits on login/register endpoints.
"""

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def get_user_id(request: Request) -> str:
    """
    Extract user ID from JWT token, fallback to IP.
    Used as the rate limit key function.
    """
    # Try to get user_id from request state (set by middleware)
    if hasattr(request.state, "user_id"):
        return f"user:{request.state.user_id}"
    # Fallback to IP for unauthenticated requests
    return f"ip:{get_remote_address(request)}"


# Create limiter at module level for use in decorators
limiter = Limiter(key_func=get_user_id)
