"""
Rate limiting utilities for API calls.

Provides rate limiting to prevent API quota exhaustion.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Thread-safe and async-safe implementation.
    """
    requests_per_second: float = 10.0
    burst_size: int = 20

    _tokens: float = field(default=0.0, repr=False)
    _last_update: float = field(default_factory=time.time, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self):
        self._tokens = float(self.burst_size)

    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire (default: 1)
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._last_update = now

            # Refill tokens based on elapsed time
            self._tokens = min(
                self.burst_size,
                self._tokens + elapsed * self.requests_per_second
            )

            if self._tokens < tokens:
                # Calculate wait time
                deficit = tokens - self._tokens
                wait_time = deficit / self.requests_per_second
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= tokens

    def acquire_sync(self, tokens: int = 1) -> None:
        """
        Synchronous version of acquire for non-async contexts.

        Args:
            tokens: Number of tokens to acquire (default: 1)
        """
        now = time.time()
        elapsed = now - self._last_update
        self._last_update = now

        # Refill tokens based on elapsed time
        self._tokens = min(
            self.burst_size,
            self._tokens + elapsed * self.requests_per_second
        )

        if self._tokens < tokens:
            # Calculate wait time
            deficit = tokens - self._tokens
            wait_time = deficit / self.requests_per_second
            time.sleep(wait_time)
            self._tokens = 0
        else:
            self._tokens -= tokens


@dataclass
class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter for API calls.

    Tracks requests within a time window and blocks when limit is exceeded.
    """
    max_requests: int = 60
    window_seconds: float = 60.0

    _requests: deque = field(default_factory=deque, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def acquire(self) -> None:
        """Acquire permission to make a request, waiting if necessary."""
        async with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds

            # Remove old requests
            while self._requests and self._requests[0] < cutoff:
                self._requests.popleft()

            if len(self._requests) >= self.max_requests:
                # Wait until oldest request expires
                wait_time = self._requests[0] - cutoff + 0.1
                await asyncio.sleep(wait_time)
                # Clean up again after waiting
                cutoff = time.time() - self.window_seconds
                while self._requests and self._requests[0] < cutoff:
                    self._requests.popleft()

            self._requests.append(now)


# Global rate limiters for different API types
_gemini_limiter: RateLimiter | None = None
_openai_limiter: RateLimiter | None = None


def get_gemini_rate_limiter() -> RateLimiter:
    """Get or create the global Gemini rate limiter."""
    global _gemini_limiter
    if _gemini_limiter is None:
        # Gemini free tier: 15 RPM, 1M TPM
        _gemini_limiter = RateLimiter(
            requests_per_second=0.25,  # ~15 per minute
            burst_size=5
        )
    return _gemini_limiter


def get_openai_rate_limiter() -> RateLimiter:
    """Get or create the global OpenAI rate limiter."""
    global _openai_limiter
    if _openai_limiter is None:
        # OpenAI tier 1: 500 RPM
        _openai_limiter = RateLimiter(
            requests_per_second=8.0,  # ~500 per minute
            burst_size=20
        )
    return _openai_limiter


def rate_limited(limiter: RateLimiter):
    """
    Decorator to rate limit a function.

    Usage:
        @rate_limited(get_gemini_rate_limiter())
        async def call_api():
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            await limiter.acquire()
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            limiter.acquire_sync()
            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
