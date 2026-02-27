"""
Configuration management for the AI correction system.

All configuration comes from environment variables or .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator, ValidationError
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

    # Security (required)
    jwt_secret: str = Field(..., min_length=32)

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

    # Validators
    @field_validator('jwt_secret')
    @classmethod
    def reject_default_values(cls, v: str) -> str:
        """Reject default/weak JWT secrets."""
        forbidden = ['your-secret-key-change-in-production', 'secret', 'test', 'password', 'change-me', 'default-secret']
        if v.lower() in forbidden:
            raise ValueError("JWT_SECRET cannot be a default value. Generate with: openssl rand -base64 32")
        return v

    @field_validator('ai_provider')
    @classmethod
    def validate_ai_provider(cls, v: str) -> str:
        """Validate AI provider is a supported value."""
        valid_providers = ['gemini', 'openai', 'glm', 'openrouter']
        if v.lower() not in valid_providers:
            raise ValueError(f"ai_provider must be one of: {', '.join(valid_providers)}")
        return v.lower()

    @model_validator(mode='after')
    def validate_api_keys(self):
        """Ensure required API key is set for the configured provider."""
        provider = self.ai_provider.lower()
        api_key_field = f"{provider}_api_key"
        api_key = getattr(self, api_key_field, "")
        if not api_key:
            raise ValueError(f"AI_CORRECTION_{provider.upper()}_API_KEY required when provider={provider}")
        return self


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reload_settings() -> Settings:
    """Reload settings from environment."""
    get_settings.cache_clear()
    return get_settings()
