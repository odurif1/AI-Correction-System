"""
Tests for API security and deployment-time invariants.
"""

from pathlib import Path
import re
import tomllib

import pytest
from fastapi import Response

from src.api.auth import _ensure_user, _issue_session_cookie
from src.config.settings import Settings
from db.models import SubscriptionTier, User


class QueryStub:
    def __init__(self, result):
        self.result = result

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self.result


class DBStub:
    def __init__(self, result=None):
        self.result = result
        self.added = []
        self.commit_count = 0
        self.refreshed = []

    def query(self, _model):
        return QueryStub(self.result)

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commit_count += 1

    def refresh(self, obj):
        self.refreshed.append(obj)


def _settings(**overrides) -> Settings:
    base = {
        "ai_provider": "openai",
        "openai_api_key": "test-key",
    }
    base.update(overrides)
    return Settings(**base)


def test_ensure_user_creates_free_anonymous_user_by_default():
    db = DBStub()

    user = _ensure_user(db, "anon-user")

    assert user.subscription_tier == SubscriptionTier.FREE
    assert db.commit_count == 1
    assert db.added[0].email == "anon-user@anonymous.local"


def test_ensure_user_promotes_admin_user_when_requested():
    existing = User(
        id="admin-api-user",
        email="admin-api-user@internal.local",
        password_hash="disabled",
        subscription_tier=SubscriptionTier.FREE,
    )
    db = DBStub(existing)

    user = _ensure_user(
        db,
        "admin-api-user",
        subscription_tier=SubscriptionTier.ADMIN,
    )

    assert user.subscription_tier == SubscriptionTier.ADMIN
    assert db.commit_count == 1


def test_validate_api_runtime_configuration_requires_session_secret():
    settings = _settings()

    with pytest.raises(ValueError, match="AI_CORRECTION_SESSION_SECRET"):
        settings.validate_api_runtime_configuration()


def test_validate_api_runtime_configuration_rejects_insecure_prod_cookies():
    settings = _settings(
        environment="production",
        session_secret="secret",
        session_cookie_secure=False,
    )

    with pytest.raises(ValueError, match="SESSION_COOKIE_SECURE"):
        settings.validate_api_runtime_configuration()


def test_validate_api_runtime_configuration_requires_external_database_in_production():
    settings = _settings(
        environment="production",
        session_secret="secret",
        session_cookie_secure=True,
    )

    with pytest.raises(ValueError, match="DATABASE_URL"):
        settings.validate_api_runtime_configuration()


def test_validate_worker_runtime_configuration_rejects_sqlite_in_production():
    settings = _settings(
        environment="production",
        database_url="sqlite:///data/app.db",
    )

    with pytest.raises(ValueError, match="must not use SQLite"):
        settings.validate_worker_runtime_configuration()


def test_secure_cookie_defaults_follow_environment():
    dev_settings = _settings(session_secret="secret", environment="development")
    prod_settings = _settings(session_secret="secret", environment="production")

    assert dev_settings.use_secure_session_cookies is False
    assert prod_settings.use_secure_session_cookies is True


def test_issue_session_cookie_marks_secure_in_production(monkeypatch):
    settings = _settings(session_secret="secret", environment="production")
    monkeypatch.setattr("src.api.auth.get_settings", lambda: settings)

    response = Response()
    _issue_session_cookie(response, "user-1")

    assert "Secure" in response.headers["set-cookie"]


def test_issue_session_cookie_is_not_secure_by_default_in_development(monkeypatch):
    settings = _settings(session_secret="secret", environment="development")
    monkeypatch.setattr("src.api.auth.get_settings", lambda: settings)

    response = Response()
    _issue_session_cookie(response, "user-1")

    assert "Secure" not in response.headers["set-cookie"]


def test_pyproject_declares_http_runtime_dependencies():
    deps = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    normalized = {
        re.split(r"[<>=!~\\[]", dep, maxsplit=1)[0].strip().lower().replace(".", "-")
        for dep in deps
    }

    expected = {
        "asgi-correlation-id",
        "google-genai",
        "openpyxl",
        "psycopg",
        "sentry-sdk",
        "slowapi",
        "sqlalchemy",
    }

    assert expected.issubset(normalized)
