"""
LLM pricing tables and cost calculation.

Provider-specific pricing per million tokens (USD).
Prices as of 2025-02 - verify with provider documentation.
"""

from typing import Dict, Optional, Any


# Pricing tables (USD per million tokens)
# Source: Provider documentation (Claude, OpenAI, Gemini)
PRICING_TABLES: Dict[str, Dict[str, float]] = {
    # Claude models (Anthropic)
    "claude-sonnet-4-20250514": {
        "prompt": 3.00,
        "completion": 15.00,
        "cached": 0.30,  # 90% discount
    },
    "claude-opus-4-20250514": {
        "prompt": 15.00,
        "completion": 75.00,
        "cached": 1.50,
    },
    "claude-3.5-sonnet": {
        "prompt": 3.00,
        "completion": 15.00,
        "cached": 0.30,
    },

    # OpenAI models
    "gpt-4o": {
        "prompt": 2.50,
        "completion": 10.00,
        "cached": 1.25,  # 50% discount
    },
    "gpt-4o-mini": {
        "prompt": 0.15,
        "completion": 0.60,
        "cached": 0.075,
    },
    "gpt-4-turbo": {
        "prompt": 10.00,
        "completion": 30.00,
        "cached": 5.00,
    },

    # Gemini models (Google)
    "gemini-2.5-pro": {
        "prompt": 1.25,
        "completion": 10.00,
        "cached": 0.31,  # 75% discount (implicit caching)
    },
    "gemini-2.5-flash": {
        "prompt": 0.075,
        "completion": 0.30,
        "cached": 0.019,
    },
    "gemini-1.5-pro": {
        "prompt": 1.25,
        "completion": 5.00,
        "cached": 0.31,
    },
}


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0
) -> Dict[str, float]:
    """
    Calculate USD cost for token usage.

    Args:
        model: Model identifier (e.g., "claude-sonnet-4-20250514")
        prompt_tokens: Number of non-cached prompt tokens
        completion_tokens: Number of completion tokens
        cached_tokens: Number of cached prompt tokens (lower cost)

    Returns:
        Dictionary with cost breakdown:
        {
            "prompt_cost_usd": float,
            "completion_cost_usd": float,
            "cached_cost_usd": float,
            "total_cost_usd": float,
            "cached_savings_usd": float,
        }
    """
    # Get pricing table, fallback to gpt-4o pricing
    pricing = PRICING_TABLES.get(model, PRICING_TABLES.get("gpt-4o", {}))

    if not pricing:
        # Default pricing if model not found
        pricing = {"prompt": 2.50, "completion": 10.00, "cached": 1.25}

    # Calculate costs (tokens / 1,000,000 * price_per_million)
    prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]
    cached_cost = (cached_tokens / 1_000_000) * pricing["cached"]

    total_cost = prompt_cost + completion_cost + cached_cost

    # Calculate savings from cache
    # Without cache, all cached tokens would be charged at prompt rate
    savings_without_cache = ((prompt_tokens + cached_tokens) / 1_000_000) * pricing["prompt"]
    cached_savings = savings_without_cache - (prompt_cost + cached_cost)

    return {
        "prompt_cost_usd": round(prompt_cost, 4),
        "completion_cost_usd": round(completion_cost, 4),
        "cached_cost_usd": round(cached_cost, 4),
        "total_cost_usd": round(total_cost, 4),
        "cached_savings_usd": round(cached_savings, 4),
    }


def get_pricing_info(model: str) -> Optional[Dict[str, float]]:
    """Get pricing table for a model."""
    return PRICING_TABLES.get(model)
