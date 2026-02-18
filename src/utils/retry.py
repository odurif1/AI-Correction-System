"""
Retry utilities for API calls.

Provides configurable retry logic with exponential backoff.
"""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar, ParamSpec, Sequence

P = ParamSpec('P')
T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Sequence[type[Exception]] = (Exception,)


def calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool
) -> float:
    """
    Calculate delay for retry attempt with exponential backoff.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter

    Returns:
        Delay in seconds
    """
    delay = base_delay * (exponential_base ** attempt)
    delay = min(delay, max_delay)

    if jitter:
        # Add random jitter between 0% and 25%
        jitter_factor = 1 + random.random() * 0.25
        delay *= jitter_factor

    return delay


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Sequence[type[Exception]] = (Exception,),
    on_retry: Callable[[int, Exception, float], None] | None = None,
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Multiplier for exponential backoff
        jitter: Add random jitter to prevent thundering herd
        retryable_exceptions: Exception types to retry on
        on_retry: Optional callback called on each retry (attempt, exception, delay)

    Usage:
        @retry_with_backoff(max_attempts=3, base_delay=1.0)
        def call_api():
            ...

        @retry_with_backoff(retryable_exceptions=[ConnectionError, TimeoutError])
        async def call_api_async():
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except tuple(retryable_exceptions) as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        delay = calculate_delay(
                            attempt, base_delay, max_delay,
                            exponential_base, jitter
                        )

                        if on_retry:
                            on_retry(attempt + 1, e, delay)
                        else:
                            logging.warning(
                                f"Retry {attempt + 1}/{max_attempts} for {func.__name__} "
                                f"after {e.__class__.__name__}: {e}. "
                                f"Waiting {delay:.1f}s"
                            )

                        time.sleep(delay)

            raise last_exception

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except tuple(retryable_exceptions) as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        delay = calculate_delay(
                            attempt, base_delay, max_delay,
                            exponential_base, jitter
                        )

                        if on_retry:
                            on_retry(attempt + 1, e, delay)
                        else:
                            logging.warning(
                                f"Retry {attempt + 1}/{max_attempts} for {func.__name__} "
                                f"after {e.__class__.__name__}: {e}. "
                                f"Waiting {delay:.1f}s"
                            )

                        await asyncio.sleep(delay)

            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class RetryExhausted(Exception):
    """Raised when all retry attempts have been exhausted."""
    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"All {attempts} retry attempts exhausted. "
            f"Last error: {last_exception}"
        )


def retry_with_config(config: RetryConfig):
    """
    Decorator using a RetryConfig object.

    Usage:
        config = RetryConfig(max_attempts=5, base_delay=2.0)
        @retry_with_config(config)
        def call_api():
            ...
    """
    return retry_with_backoff(
        max_attempts=config.max_attempts,
        base_delay=config.base_delay,
        max_delay=config.max_delay,
        exponential_base=config.exponential_base,
        jitter=config.jitter,
        retryable_exceptions=config.retryable_exceptions,
    )


# Common retry configurations
API_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True,
)

AGGRESSIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    base_delay=0.5,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
)

CONSERVATIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=2,
    base_delay=2.0,
    max_delay=10.0,
    exponential_base=2.0,
    jitter=True,
)
