"""
API module for the AI correction system.

Provides FastAPI application and routes for web access.
"""

from api.app import create_app, app

__all__ = ['create_app', 'app']
