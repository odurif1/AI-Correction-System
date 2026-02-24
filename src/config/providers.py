"""
Provider registry and configuration.

All provider-specific settings in one place.
No hardcoded values in provider classes.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class ProviderConfig:
    """Configuration for a single AI provider."""
    api_key: str = ""
    base_url: Optional[str] = None
    model: Optional[str] = None
    vision_model: Optional[str] = None
    embedding_model: Optional[str] = None
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def effective_vision_model(self) -> Optional[str]:
        """Vision model falls back to text model."""
        return self.vision_model or self.model


# Provider metadata - defines how to create each provider
PROVIDER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gemini": {
        "api_key_attr": "gemini_api_key",
        "model_attr": "gemini_model",
        "vision_attr": "gemini_vision_model",
        "embedding_attr": "gemini_embedding_model",
        "requires_native": True,  # Uses google-genai, not OpenAI-compatible
    },
    "openai": {
        "api_key_attr": "openai_api_key",
        "model_attr": "openai_model",
        "vision_attr": "openai_vision_model",
        "embedding_attr": "openai_embedding_model",
        "base_url": None,  # Default OpenAI API
    },
    "glm": {
        "api_key_attr": "glm_api_key",
        "model_attr": "glm_model",
        "vision_attr": "glm_vision_model",
        "base_url": "https://api.z.ai/api/paas/v4",
    },
    "openrouter": {
        "api_key_attr": "openrouter_api_key",
        "model_attr": "openrouter_model",
        "vision_attr": "openrouter_vision_model",
        "base_url": "https://openrouter.ai/api/v1",
        "extra_headers": {
            "HTTP-Referer": "https://github.com/odurif1/AI-Correction-System",
            "X-Title": "AI-Correction"
        }
    },
}


def get_provider_config(provider_name: str, settings) -> ProviderConfig:
    """
    Build ProviderConfig from settings for a given provider.

    Args:
        provider_name: Name of provider (e.g., "gemini", "openai", "glm")
        settings: Settings instance

    Returns:
        ProviderConfig with all settings populated
    """
    registry = PROVIDER_REGISTRY.get(provider_name.lower())
    if not registry:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(PROVIDER_REGISTRY.keys())}")

    return ProviderConfig(
        api_key=getattr(settings, registry["api_key_attr"], "") or "",
        base_url=registry.get("base_url"),
        model=getattr(settings, registry["model_attr"], None),
        vision_model=getattr(settings, registry["vision_attr"], None),
        embedding_model=getattr(settings, registry.get("embedding_attr", ""), None),
        extra_headers=registry.get("extra_headers", {}),
        extra_kwargs=registry.get("extra_kwargs", {}),
    )
