"""
AI provider implementations for correction system.

Supports multiple AI backends:
- OpenAI (GPT-4o)
- Google Gemini (3 Pro)

Usage:
    from ai import create_ai_provider

    # Auto-detect provider from environment
    provider = create_ai_provider()

    # Or specify provider
    provider = create_ai_provider(provider_type="gemini")

    # Use for grading
    result = provider.grade_with_vision(
        question_text="Q1: Explain photosynthesis",
        criteria="Check for key concepts",
        image_path="student_answer.png",
        max_points=5.0
    )
"""

# Lazy imports to avoid circular dependencies
__all__ = [
    "OpenAIProvider",
    "GeminiProvider",
    "ComparisonProvider",
    "create_ai_provider",
    "create_comparison_provider",
    "get_available_providers"
]


def OpenAIProvider(*args, **kwargs):
    """Create an OpenAI provider instance (lazy import)."""
    from .openai_provider import OpenAIProvider as _OpenAIProvider
    return _OpenAIProvider(*args, **kwargs)


def GeminiProvider(*args, **kwargs):
    """Create a Gemini provider instance (lazy import)."""
    from .gemini_provider import GeminiProvider as _GeminiProvider
    return _GeminiProvider(*args, **kwargs)


def ComparisonProvider(*args, **kwargs):
    """Create a comparison provider for dual-LLM grading (lazy import)."""
    from .comparison_provider import ComparisonProvider as _ComparisonProvider
    return _ComparisonProvider(*args, **kwargs)


def create_ai_provider(*args, **kwargs):
    """Create an AI provider based on configuration (lazy import)."""
    from .provider_factory import create_ai_provider as _create_ai_provider
    return _create_ai_provider(*args, **kwargs)


def create_comparison_provider(*args, **kwargs):
    """Create a comparison provider for dual-LLM mode (lazy import)."""
    from .provider_factory import create_comparison_provider as _create_comparison_provider
    return _create_comparison_provider(*args, **kwargs)


def get_available_providers(*args, **kwargs):
    """Get list of available providers based on configured API keys (lazy import)."""
    from .provider_factory import get_available_providers as _get_available_providers
    return _get_available_providers(*args, **kwargs)
