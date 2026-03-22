"""Anonymous session authentication helpers for the public correction API."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import uuid
from typing import Optional

from fastapi import Depends, HTTPException, Request, Response, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from config.settings import get_settings
from db import SessionLocal, SubscriptionTier, User, get_db

SESSION_COOKIE_NAME = "ai_correction_session"
ADMIN_API_KEY_NAME = "X-API-Key"
admin_api_key_header = APIKeyHeader(name=ADMIN_API_KEY_NAME, auto_error=False)


def _session_secret() -> str:
    settings = get_settings()
    if settings.session_secret:
        return settings.session_secret
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Session secret not configured",
    )


def _sign_payload(payload: dict[str, str]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    signature = hmac.new(
        _session_secret().encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload_b64}.{signature}"


def _unsign_payload(token: str) -> Optional[dict[str, str]]:
    try:
        payload_b64, signature = token.split(".", 1)
    except ValueError:
        return None

    expected = hmac.new(
        _session_secret().encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None

    padded = payload_b64 + "=" * (-len(payload_b64) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None
    return payload


def _ensure_user(
    db: Session,
    user_id: str,
    *,
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE,
    name: str = "Anonymous session",
    email: Optional[str] = None,
) -> User:
    user = db.query(User).filter(User.id == user_id).first()
    if user is not None:
        if subscription_tier == SubscriptionTier.ADMIN and user.subscription_tier != SubscriptionTier.ADMIN:
            user.subscription_tier = SubscriptionTier.ADMIN
            db.add(user)
            db.commit()
            db.refresh(user)
        return user

    user = User(
        id=user_id,
        email=email or f"{user_id}@anonymous.local",
        password_hash="disabled",
        name=name,
        subscription_tier=subscription_tier,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def extract_user_id_from_cookie_value(cookie_value: str | None) -> Optional[str]:
    if not cookie_value:
        return None
    payload = _unsign_payload(cookie_value)
    user_id = payload.get("user_id") if payload else None
    if not isinstance(user_id, str) or not user_id:
        return None
    return user_id


def extract_user_id_from_request(request: Request) -> Optional[str]:
    return extract_user_id_from_cookie_value(request.cookies.get(SESSION_COOKIE_NAME))


def _issue_session_cookie(response: Response, user_id: str) -> None:
    token = _sign_payload({"user_id": user_id})
    settings = get_settings()
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=settings.use_secure_session_cookies,
        max_age=60 * 60 * 24 * 30,
        path="/",
    )


async def get_current_user(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> User:
    """Return an anonymous per-browser user, persisted in a signed cookie."""
    user_id = extract_user_id_from_request(request)
    if user_id:
        user = db.query(User).filter(User.id == user_id).first()
        if user is not None:
            request.state.user_id = user_id
            return user

    user_id = str(uuid.uuid4())
    user = _ensure_user(db, user_id, subscription_tier=SubscriptionTier.FREE)
    request.state.user_id = user_id
    _issue_session_cookie(response, user_id)
    return user


async def get_optional_user(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> User:
    return await get_current_user(request=request, response=response, db=db)


async def get_admin_user(
    api_key: str = Security(admin_api_key_header),
) -> User:
    expected_key = get_settings().admin_api_key
    if not expected_key or api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin API key",
        )

    db = SessionLocal()
    try:
        return _ensure_user(
            db,
            "admin-api-user",
            subscription_tier=SubscriptionTier.ADMIN,
            name="Admin API user",
            email="admin-api-user@internal.local",
        )
    finally:
        db.close()
