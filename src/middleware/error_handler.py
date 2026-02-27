"""
Error tracking and global exception handling for La Corrigeuse.

Integrates Sentry for production error aggregation and debugging.
"""

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
import logging


def init_sentry(
    dsn: str,
    environment: str,
    sample_rate: float = 0.1,
    debug: bool = False
) -> None:
    """
    Initialize Sentry with FastAPI integration.

    Args:
        dsn: Sentry DSN from dashboard
        environment: 'development' or 'production'
        sample_rate: Traces sample rate (1.0 for dev, 0.1 for prod)
        debug: Enable debug mode (verbose logging)
    """
    if not dsn or dsn == "":
        logger.warning("SENTRY_DSN not configured - error tracking disabled")
        return

    # Configure sampling based on environment
    traces_sample_rate = 1.0 if environment == "development" else sample_rate

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        traces_sample_rate=traces_sample_rate,
        debug=debug,

        # FastAPI-specific integration
        integrations=[
            FastApiIntegration(),
            LoggingIntegration(
                level=logging.INFO,  # Capture info and above
                event_level=logging.ERROR  # Send errors as events
            )
        ],

        # Filter out health check noise
        before_send_transaction=lambda event: None if event.get("transaction", "").startswith("/health") else event,

        # Filter sensitive data
        before_send=_filter_sensitive_data
    )

    logger.info(f"Sentry initialized: environment={environment}, traces_sample_rate={traces_sample_rate}")


def _filter_sensitive_data(event, hint):
    """
    Filter sensitive data before sending to Sentry.

    Removes: Authorization headers, cookies, API keys, passwords.
    """
    if "request" in event:
        # Remove sensitive headers
        event["request"]["headers"] = {
            k: v for k, v in event["request"]["headers"].items()
            if k.lower() not in ["authorization", "cookie", "x-api-key"]
        }

    # Remove sensitive data from extra
    if "extra" in event:
        sensitive_keys = ["password", "token", "api_key", "secret", "jwt_secret"]
        for key in sensitive_keys:
            if key in event["extra"]:
                del event["extra"][key]

    return event


async def sentry_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler that captures errors in Sentry.

    Handles all uncaught exceptions, logs them, and returns
    generic error message to user (security best practice).
    """
    # Send to Sentry
    sentry_sdk.capture_exception(exc)

    # Log error
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={
            "request_path": request.url.path,
            "request_method": request.method
        }
    )

    # Return generic error to user
    return JSONResponse(
        status_code=500,
        content={"detail": "Une erreur est survenue. Nos équipes ont été notifiées."}
    )


def set_user_context(user_id: str, email: str = None, username: str = None) -> None:
    """
    Set user context in Sentry for error tracking.

    Call this after authentication to associate errors with users.

    Args:
        user_id: User ID
        email: User email (optional)
        username: User name (optional)
    """
    sentry_sdk.set_user({
        "id": user_id,
        "email": email,
        "username": username
    })
