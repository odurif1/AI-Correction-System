"""
Factory for creating AI provider instances.

Supports switching between OpenAI and Gemini providers based on configuration.
Also supports comparison mode with dual LLMs.
"""

from typing import Optional, Union, List, Tuple, Any

from ai.openai_provider import OpenAIProvider

# Optional Gemini support
try:
    from ai.gemini_provider import GeminiProvider
    _gemini_available = True
except ImportError:
    _gemini_available = False
    GeminiProvider = None  # type: ignore

from ai.comparison_provider import ComparisonProvider
from config.settings import get_settings


def create_ai_provider(
    provider_type: str = None,
    mock_mode: bool = False
) -> Union[OpenAIProvider, "GeminiProvider"]:
    """
    Create an AI provider instance based on configuration.

    Args:
        provider_type: "openai" or "gemini" (default: from settings or "openai")
        mock_mode: If True, skip API key validation for testing

    Returns:
        An instance of OpenAIProvider or GeminiProvider

    Raises:
        ValueError: If provider_type is invalid or API key is missing
    """
    settings = get_settings()

    # Auto-detect provider type from available API keys
    # Gemini is preferred when available (better for French, more cost-effective)
    if provider_type is None:
        if settings.gemini_api_key and _gemini_available:
            provider_type = "gemini"
        elif settings.openai_api_key:
            provider_type = "openai"
        else:
            raise ValueError(
                "No AI provider API key configured. "
                "Set AI_CORRECTION_GEMINI_API_KEY (recommended) or AI_CORRECTION_OPENAI_API_KEY."
            )

    if provider_type.lower() == "gemini":
        if not _gemini_available:
            raise ValueError(
                "Gemini provider is not available. "
                "Install google-generativeai package to use Gemini."
            )
        return GeminiProvider(mock_mode=mock_mode)  # type: ignore
    elif provider_type.lower() == "openai":
        return OpenAIProvider(mock_mode=mock_mode)
    else:
        raise ValueError(
            f"Invalid provider_type: {provider_type}. "
            "Must be 'openai' or 'gemini'."
        )


def get_available_providers() -> list[str]:
    """
    Get list of available AI providers based on configured API keys.

    Returns:
        List of provider names that have API keys configured
    """
    settings = get_settings()
    available = []

    if settings.openai_api_key:
        available.append("openai")

    if settings.gemini_api_key and _gemini_available:
        available.append("gemini")

    return available


def _create_gemini_provider(model: str = None, mock_mode: bool = False):
    """Create a Gemini provider instance."""
    if not _gemini_available:
        raise ValueError("Gemini provider not available. Install google-genai package.")
    return GeminiProvider(model=model, mock_mode=mock_mode)


def _create_openai_provider(model: str = None, mock_mode: bool = False):
    """Create an OpenAI provider instance."""
    return OpenAIProvider(model=model, mock_mode=mock_mode)


# Provider factory registry - extensible for future providers
_PROVIDER_FACTORIES = {
    "gemini": _create_gemini_provider,
    "openai": _create_openai_provider,
    # Add more providers here as needed:
    # "anthropic": _create_anthropic_provider,
    # "ollama": _create_ollama_provider,
    # "mistral": _create_mistral_provider,
}


def create_comparison_provider(
    mock_mode: bool = False,
    disagreement_callback: callable = None,
    progress_callback: callable = None
) -> ComparisonProvider:
    """
    Create a comparison provider with two LLMs for dual grading.

    Configuration via environment variables (REQUIRED):
        AI_CORRECTION_LLM1_PROVIDER=gemini
        AI_CORRECTION_LLM1_MODEL=gemini-2.5-flash
        AI_CORRECTION_LLM2_PROVIDER=gemini
        AI_CORRECTION_LLM2_MODEL=gemini-3-pro

    Args:
        mock_mode: If True, skip API key validation for testing
        disagreement_callback: Optional async callback for when LLMs disagree
                               Signature: async def callback(question_id, question_text,
                                                             llm1_name, llm1_result,
                                                             llm2_name, llm2_result,
                                                             max_points) -> float
        progress_callback: Optional callback for progress updates
                           Signature: async def callback(event_type, data)

    Returns:
        ComparisonProvider instance with two providers

    Raises:
        ValueError: If LLM1_PROVIDER or LLM2_PROVIDER is not configured
    """
    settings = get_settings()
    providers: List[Tuple[str, Any]] = []

    # LLM1 - REQUIRED
    llm1_provider = settings.llm1_provider
    llm1_model = settings.llm1_model

    if llm1_provider is None:
        raise ValueError(
            "LLM1_PROVIDER is required for comparison mode. "
            "Set AI_CORRECTION_LLM1_PROVIDER"
        )

    factory1 = _PROVIDER_FACTORIES.get(llm1_provider.lower())
    if factory1 is None:
        available = list(_PROVIDER_FACTORIES.keys())
        raise ValueError(
            f"Unknown LLM1 provider: {llm1_provider}. "
            f"Available: {available}"
        )

    provider1_instance = factory1(model=llm1_model, mock_mode=mock_mode)
    # Use full model name for display
    llm1_display_name = llm1_model or llm1_provider
    providers.append((llm1_display_name, provider1_instance))

    # LLM2 - REQUIRED
    llm2_provider = settings.llm2_provider
    llm2_model = settings.llm2_model

    if llm2_provider is None:
        raise ValueError(
            "LLM2_PROVIDER is required for comparison mode. "
            "Set AI_CORRECTION_LLM2_PROVIDER"
        )

    factory2 = _PROVIDER_FACTORIES.get(llm2_provider.lower())
    if factory2 is None:
        available = list(_PROVIDER_FACTORIES.keys())
        raise ValueError(
            f"Unknown LLM2 provider: {llm2_provider}. "
            f"Available: {available}"
        )

    provider2_instance = factory2(model=llm2_model, mock_mode=mock_mode)
    # Use full model name for display
    llm2_display_name = llm2_model or llm2_provider
    providers.append((llm2_display_name, provider2_instance))

    return ComparisonProvider(
        providers,
        disagreement_callback=disagreement_callback,
        progress_callback=progress_callback
    )
