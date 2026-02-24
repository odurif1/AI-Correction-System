"""
Factory for creating AI provider instances.

Uses registry-based configuration for easy extensibility.
"""

from typing import Union, List, Tuple, Any, Optional

from ai.openai_provider import OpenAIProvider
from config.settings import get_settings
from config.providers import get_provider_config, PROVIDER_REGISTRY

# Optional Gemini support (requires google-genai)
try:
    from ai.gemini_provider import GeminiProvider
    _gemini_available = True
except ImportError:
    _gemini_available = False
    GeminiProvider = None  # type: ignore

from ai.comparison_provider import ComparisonProvider


def create_ai_provider(
    provider_type: str = None,
    model: str = None,
    mock_mode: bool = False
) -> Union[OpenAIProvider, "GeminiProvider"]:
    """
    Create an AI provider instance.

    Args:
        provider_type: Provider name (default: from settings)
        model: Override model name
        mock_mode: Skip API key validation for testing

    Returns:
        Provider instance
    """
    settings = get_settings()
    provider_type = (provider_type or settings.ai_provider).lower()

    if provider_type not in PROVIDER_REGISTRY:
        raise ValueError(f"Unknown provider: {provider_type}. Available: {list(PROVIDER_REGISTRY.keys())}")

    config = get_provider_config(provider_type, settings)

    # Override model if specified
    if model:
        config.model = model
        config.vision_model = config.vision_model or model

    # Validate API key
    if not config.api_key and not mock_mode:
        raise ValueError(f"API key required for {provider_type}. Set AI_CORRECTION_{provider_type.upper()}_API_KEY")

    # Validate model
    if not config.model and not mock_mode:
        raise ValueError(f"Model required for {provider_type}. Set AI_CORRECTION_{provider_type.upper()}_MODEL")

    # Gemini uses native client
    if provider_type == "gemini":
        if not _gemini_available:
            raise ValueError("Gemini provider not available. Install google-genai package.")
        return GeminiProvider(
            api_key=config.api_key,
            model=config.model,
            vision_model=config.vision_model,
            embedding_model=config.embedding_model,
            mock_mode=mock_mode
        )

    # All other providers use OpenAI-compatible API
    return OpenAIProvider(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        vision_model=config.effective_vision_model,
        embedding_model=config.embedding_model,
        name=f"{provider_type}/{config.model}",
        mock_mode=mock_mode,
        extra_headers=config.extra_headers,
        **config.extra_kwargs
    )


def get_available_providers() -> List[str]:
    """Get providers with configured API keys."""
    settings = get_settings()
    available = []

    for name in PROVIDER_REGISTRY:
        config = get_provider_config(name, settings)
        if config.api_key:
            if name == "gemini" and not _gemini_available:
                continue
            available.append(name)

    return available


def create_comparison_provider(
    mock_mode: bool = False,
    disagreement_callback: callable = None,
    progress_callback: callable = None
) -> ComparisonProvider:
    """
    Create a comparison provider with two LLMs.

    Requires AI_CORRECTION_LLM1_PROVIDER and AI_CORRECTION_LLM2_PROVIDER in .env.
    """
    settings = get_settings()
    providers: List[Tuple[str, Any]] = []

    for llm_num in [1, 2]:
        provider_name = getattr(settings, f"llm{llm_num}_provider")
        model = getattr(settings, f"llm{llm_num}_model")

        if not provider_name:
            raise ValueError(f"LLM{llm_num}_PROVIDER required for comparison mode")

        provider = create_ai_provider(provider_name, model=model, mock_mode=mock_mode)
        # Use LLM1/LLM2 prefix to distinguish providers even if same model
        model_name = model or provider_name
        display_name = f"LLM{llm_num}: {model_name}"
        providers.append((display_name, provider))

    return ComparisonProvider(
        providers,
        disagreement_callback=disagreement_callback,
        progress_callback=progress_callback
    )
