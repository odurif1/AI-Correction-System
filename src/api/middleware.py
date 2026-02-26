"""
Access control middleware for La Corrigeuse.

This module provides dependency injection functions for enforcing session
ownership and access control on protected API endpoints.

Usage in protected endpoints:

    @app.get("/api/sessions/{session_id}")
    async def get_session(
        session_id: str,
        user_id: str = Depends(verify_session_ownership),  # Ownership verified
        current_user: User = Depends(get_current_user)
    ):
        # user_id is guaranteed to equal current_user.id
        # Session is guaranteed to exist and belong to user
        store = SessionStore(session_id=session_id, user_id=user_id)
        return store.load_session()

For endpoints that create new sessions (no ownership check needed yet):

    @app.post("/api/sessions")
    async def create_session(
        current_user: User = Depends(get_current_user)
    ):
        session_id = str(uuid.uuid4())
        user_id = current_user.id
        store = SessionStore(session_id=session_id, user_id=user_id)
        store.create()
        return {"session_id": session_id}
"""

from pathlib import Path
from typing import str

from fastapi import Depends, HTTPException, status

from api.auth import get_current_user
from config.constants import DATA_DIR
from db import User


async def verify_session_ownership(
    session_id: str,
    current_user: User = Depends(get_current_user)
) -> str:
    """
    Verify that the current user owns the requested session.

    This function performs two critical security checks:
    1. The session exists in the user's directory (prevents unauthorized access)
    2. The path is within the user's directory (prevents path traversal attacks)

    Args:
        session_id: The session ID to access
        current_user: The authenticated user (injected by dependency)

    Returns:
        The user_id for path construction (guaranteed to match current_user.id)

    Raises:
        HTTPException 404: If session doesn't exist in user's directory
        HTTPException 403: If path traversal attempted or session belongs to another user

    Example:
        ```python
        @app.get("/api/sessions/{session_id}/status")
        async def get_session_status(
            session_id: str,
            user_id: str = Depends(verify_session_ownership)
        ):
            # Safe to proceed - user owns this session
            store = SessionStore(session_id=session_id, user_id=user_id)
            return {"status": store.load_session().status}
        ```
    """
    user_id = current_user.id

    # Construct the expected session path
    session_path = Path(DATA_DIR) / "sessions" / user_id / session_id

    # Check if session exists in user's directory
    if not session_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session non trouvée"
        )

    # Additional safety: Verify path is within user's directory
    # This prevents path traversal attacks like "../../../other_user/session"
    expected_base = Path(DATA_DIR) / "sessions" / user_id
    try:
        session_path.resolve().relative_to(expected_base.resolve())
    except ValueError:
        # Path is not within expected base - possible path traversal
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès non autorisé"
        )

    return user_id
