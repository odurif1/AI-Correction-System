"""
Configuration management for the AI correction system.

All configuration comes from environment variables or .env file.
No hardcoded defaults - .env is required.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AI_CORRECTION_",
        case_sensitive=False
    )

    # API Keys
    gemini_api_key: str
    openai_api_key: str = ""

    # Gemini Configuration (required)
    gemini_model: str
    gemini_vision_model: str
    gemini_embedding_model: str = "models/gemini-embedding-001"

    # AI Provider
    ai_provider: str = "gemini"

    # Comparison Mode (dual LLM) - configurable via ENV
    comparison_mode: bool = False
    llm1_provider: Optional[str] = None  # "gemini" or "openai"
    llm1_model: Optional[str] = None
    llm2_provider: Optional[str] = None  # "gemini" or "openai"
    llm2_model: Optional[str] = None

    # Thresholds
    confidence_auto: float = 0.85
    confidence_flag: float = 0.60

    # Storage
    data_dir: str = "data"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reload_settings() -> Settings:
    """Reload settings from environment."""
    get_settings.cache_clear()
    return get_settings()
