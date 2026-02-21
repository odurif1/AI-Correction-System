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

    # Grade agreement threshold (percentage of max_points)
    # Used to detect disagreements between LLMs
    # e.g., 0.10 means 10% of max_points
    grade_agreement_threshold: float = 0.10

    # Flip-flop detection threshold (percentage of max_points)
    # Used to detect when LLMs swap positions after verification
    # 0 = detect any position swap regardless of magnitude
    # 0.25 = only flag swaps where differences exceed 25% of max_points
    flip_flop_threshold: float = 0.0

    # Storage
    data_dir: str = "data"

    # CORS - allowed origins for web frontend
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reload_settings() -> Settings:
    """Reload settings from environment."""
    get_settings.cache_clear()
    return get_settings()
