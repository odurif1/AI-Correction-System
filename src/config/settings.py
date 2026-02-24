"""
Configuration management for the AI correction system.

All configuration comes from environment variables or .env file.
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

    # AI Provider (required)
    ai_provider: str  # "gemini", "openai", "glm", "openrouter"

    # API Keys
    gemini_api_key: str = ""
    openai_api_key: str = ""
    glm_api_key: str = ""
    openrouter_api_key: str = ""

    # Gemini Models
    gemini_model: Optional[str] = None
    gemini_vision_model: Optional[str] = None
    gemini_embedding_model: Optional[str] = None

    # OpenAI Models
    openai_model: Optional[str] = None
    openai_vision_model: Optional[str] = None
    openai_embedding_model: Optional[str] = None

    # GLM Models
    glm_model: Optional[str] = None
    glm_vision_model: Optional[str] = None

    # OpenRouter Models
    openrouter_model: Optional[str] = None
    openrouter_vision_model: Optional[str] = None

    # Comparison Mode (dual LLM)
    comparison_mode: bool = False
    llm1_provider: Optional[str] = None
    llm1_model: Optional[str] = None
    llm2_provider: Optional[str] = None
    llm2_model: Optional[str] = None

    # Thresholds
    confidence_auto: float = 0.85
    confidence_flag: float = 0.60
    grade_agreement_threshold: float = 0.10
    flip_flop_threshold: float = 0.0

    # Annotation (optional - uses main provider if not set)
    annotation_provider: Optional[str] = None
    annotation_model: Optional[str] = None

    # Storage
    data_dir: str = "data"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reload_settings() -> Settings:
    """Reload settings from environment."""
    get_settings.cache_clear()
    return get_settings()
