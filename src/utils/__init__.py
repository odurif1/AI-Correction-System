"""
Utility functions for the AI correction system.
"""

from utils.confidence import choose_by_confidence
from utils.sorting import natural_sort_key, question_sort_key
from utils.rate_limiter import (
    RateLimiter,
    SlidingWindowRateLimiter,
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
    get_gemini_rate_limiter,
    get_openai_rate_limiter,
    get_gemini_circuit_breaker,
    get_openai_circuit_breaker,
    rate_limited,
    with_circuit_breaker,
)

__all__ = [
    'choose_by_confidence',
    'natural_sort_key',
    'question_sort_key',
    'RateLimiter',
    'SlidingWindowRateLimiter',
    'CircuitBreaker',
    'CircuitState',
    'CircuitOpenError',
    'get_gemini_rate_limiter',
    'get_openai_rate_limiter',
    'get_gemini_circuit_breaker',
    'get_openai_circuit_breaker',
    'rate_limited',
    'with_circuit_breaker',
]
