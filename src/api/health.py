"""
Health check endpoint for La Corrigeuse.

Provides system health status for load balancers and monitoring.
"""

from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from db import SessionLocal
from loguru import logger

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint with database status.

    Returns:
        JSON with status, version, and database connection status.
        HTTP 200 if healthy, 503 if database disconnected.
    """
    health_status = {
        "status": "healthy",
        "version": "1.0.0",
        "database": "unknown"
    }

    # Check database connection
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        health_status["database"] = "connected"
        db.close()
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["database"] = f"disconnected: {str(e)}"
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=health_status
        )

    return health_status
