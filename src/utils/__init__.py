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
from utils.retry import (
    RetryConfig,
    RetryExhausted,
    retry_with_backoff,
    retry_with_config,
    API_RETRY_CONFIG,
    AGGRESSIVE_RETRY_CONFIG,
    CONSERVATIVE_RETRY_CONFIG,
)
from utils.type_guards import (
    is_dict_with_keys,
    is_grading_result,
    is_reading_result,
    is_question_detection,
    is_student_name_result,
    is_api_error,
    is_list_of,
    is_scale_detection,
    ensure_dict,
    ensure_list,
    ensure_str,
    ensure_float,
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
    'RetryConfig',
    'RetryExhausted',
    'retry_with_backoff',
    'retry_with_config',
    'API_RETRY_CONFIG',
    'AGGRESSIVE_RETRY_CONFIG',
    'CONSERVATIVE_RETRY_CONFIG',
    'is_dict_with_keys',
    'is_grading_result',
    'is_reading_result',
    'is_question_detection',
    'is_student_name_result',
    'is_api_error',
    'is_list_of',
    'is_scale_detection',
    'ensure_dict',
    'ensure_list',
    'ensure_str',
    'ensure_float',
]
