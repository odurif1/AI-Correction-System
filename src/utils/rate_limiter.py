"""
Rate limiting and circuit breaker utilities for API calls.

Provides rate limiting to prevent API quota exhaustion and
circuit breaker pattern to protect against failing APIs.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject all calls
    HALF_OPEN = "half_open"  # Testing if recovered


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


# ==================== CIRCUIT BREAKER ====================

@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern for API resilience.

    Protects against cascading failures by "tripping" when
    failure threshold is exceeded, then allowing periodic
    test requests to check if the service has recovered.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing, all requests rejected immediately
    - HALF_OPEN: Testing recovery, limited requests allowed
    """
    failure_threshold: int = 5
    recovery_timeout: float = 30.0  # seconds before trying half-open
    half_open_max_calls: int = 3

    _state: CircuitState = field(default=CircuitState.CLOSED, repr=False)
    _failure_count: int = field(default=0, repr=False)
    _last_failure_time: float = field(default=0.0, repr=False)
    _half_open_calls: int = field(default=0, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing)."""
        if self._state == CircuitState.OPEN:
            # Check if we should transition to half-open
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                return False  # Will transition on next call
        return self._state == CircuitState.OPEN

    async def can_execute(self) -> bool:
        """
        Check if a request can be executed.

        Returns:
            True if request should proceed, False if circuit is open
        """
        async with self._lock:
            now = time.time()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if now - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logging.info("Circuit breaker entering HALF_OPEN state")
                    return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open state
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

        return False

    async def record_success(self) -> None:
        """Record a successful request."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # Successful request in half-open means recovery
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
                logging.info("Circuit breaker recovered, now CLOSED")

    async def record_failure(self) -> None:
        """Record a failed request."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Failure in half-open means back to open
                self._state = CircuitState.OPEN
                logging.warning("Circuit breaker failed in HALF_OPEN, back to OPEN")

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logging.warning(
                        f"Circuit breaker tripped to OPEN after "
                        f"{self._failure_count} failures"
                    )

    def can_execute_sync(self) -> bool:
        """Synchronous version of can_execute."""
        now = time.time()

        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            if now - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logging.info("Circuit breaker entering HALF_OPEN state")
                return True
            return False

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

        return False

    def record_success_sync(self) -> None:
        """Synchronous version of record_success."""
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
            logging.info("Circuit breaker recovered, now CLOSED")

    def record_failure_sync(self) -> None:
        """Synchronous version of record_failure."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logging.warning("Circuit breaker failed in HALF_OPEN, back to OPEN")

        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logging.warning(
                    f"Circuit breaker tripped to OPEN after "
                    f"{self._failure_count} failures"
                )


# Global circuit breakers for different API types
_gemini_circuit: CircuitBreaker | None = None
_openai_circuit: CircuitBreaker | None = None


def get_gemini_circuit_breaker() -> CircuitBreaker:
    """Get or create the global Gemini circuit breaker."""
    global _gemini_circuit
    if _gemini_circuit is None:
        _gemini_circuit = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            half_open_max_calls=2
        )
    return _gemini_circuit


def get_openai_circuit_breaker() -> CircuitBreaker:
    """Get or create the global OpenAI circuit breaker."""
    global _openai_circuit
    if _openai_circuit is None:
        _openai_circuit = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            half_open_max_calls=3
        )
    return _openai_circuit


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


def with_circuit_breaker(circuit: CircuitBreaker):
    """
    Decorator to wrap function with circuit breaker.

    Raises CircuitOpenError when circuit is open.

    Usage:
        @with_circuit_breaker(get_gemini_circuit_breaker())
        async def call_api():
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not await circuit.can_execute():
                raise CircuitOpenError(
                    f"Circuit breaker is open for {func.__name__}"
                )
            try:
                result = await func(*args, **kwargs)
                await circuit.record_success()
                return result
            except Exception as e:
                await circuit.record_failure()
                raise

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not circuit.can_execute_sync():
                raise CircuitOpenError(
                    f"Circuit breaker is open for {func.__name__}"
                )
            try:
                result = func(*args, **kwargs)
                circuit.record_success_sync()
                return result
            except Exception as e:
                circuit.record_failure_sync()
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
