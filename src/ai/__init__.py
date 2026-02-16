"""
AI provider implementations for correction system.

Supports multiple AI backends:
- OpenAI (GPT-4o)
- Google Gemini (3 Pro)
"""

# Lazy imports to avoid circular dependencies
__all__ = [
    "OpenAIProvider",
    "GeminiProvider",
    "create_ai_provider",
    "get_available_providers"
]


def OpenAIProvider(*args, **kwargs):
    from .openai_provider import OpenAIProvider as _OpenAIProvider
    return _OpenAIProvider(*args, **kwargs)


def GeminiProvider(*args, **kwargs):
    from .gemini_provider import GeminiProvider as _GeminiProvider
    return _GeminiProvider(*args, **kwargs)


def create_ai_provider(*args, **kwargs):
    from .provider_factory import create_ai_provider as _create_ai_provider
    return _create_ai_provider(*args, **kwargs)


def get_available_providers(*args, **kwargs):
    from .provider_factory import get_available_providers as _get_available_providers
    return _get_available_providers(*args, **kwargs)
