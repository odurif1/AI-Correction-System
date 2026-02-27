"""
Centralized logging configuration for the AI correction system.

Provides structured JSON logging with correlation ID support for production observability.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger


def setup_structured_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    include_modules: Optional[list[str]] = None
) -> None:
    """
    Configure structured JSON logging with Loguru.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file (for local development)
        include_modules: List of module names to enable debug logging for

    Returns:
        None (Loguru configures its own handlers)
    """
    # Remove default handler
    logger.remove()

    # Add stdout handler with JSON serialization for production
    # Note: serialize=True outputs full JSON with all extra fields in "record.extra"
    # The format string only affects the "text" field in the JSON output
    logger.add(
        sys.stdout,
        serialize=True,  # JSON format - includes correlation_id in record.extra
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name} | {message}",
        level=level,
        enqueue=True,    # Async logging (non-blocking)
        backtrace=True,  # Full stack trace on errors
        diagnose=True    # Variable values on errors
    )

    # File handler (optional - for local development)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_path,
            serialize=True,  # JSON format in file too
            level="DEBUG",    # Always debug to file
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )

    # Enable debug for specific modules
    if include_modules:
        for module in include_modules:
            logger.level(module, level="DEBUG")

    # Suppress noisy third-party loggers
    logger.disable("httpx")
    logger.disable("httpcore")
    logger.disable("urllib3")
    logger.disable("PIL")
    logger.disable("fitz")


def get_logger(name: str):
    """
    Get a logger for a specific module with correlation ID binding.

    Args:
        name: Module name (usually __name__)

    Returns:
        Logger instance with module binding
    """
    return logger.bind(module=name)


# ============================================================================
# Legacy stdlib logging interface (backward compatibility)
# ============================================================================

import logging

# Log format templates (for backward compatibility only)
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT = "%(levelname)s: %(message)s"
JSON_FORMAT = '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "detailed",
    include_modules: Optional[list[str]] = None
) -> logging.Logger:
    """
    Configure stdlib logging (LEGACY - use setup_structured_logging instead).

    This function is kept for backward compatibility with existing code.
    New code should use setup_structured_logging() which uses Loguru.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_format: Format style ("detailed", "simple", "json")
        include_modules: List of module names to enable debug logging for

    Returns:
        Root logger
    """
    import warnings
    warnings.warn(
        "setup_logging() is deprecated. Use setup_structured_logging() for JSON logging.",
        DeprecationWarning,
        stacklevel=2
    )

    # Select format
    formats = {
        "detailed": DETAILED_FORMAT,
        "simple": SIMPLE_FORMAT,
        "json": JSON_FORMAT,
    }
    fmt = formats.get(log_format, DETAILED_FORMAT)

    # Create formatter
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always debug to file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Enable debug for specific modules
    if include_modules:
        for module in include_modules:
            logging.getLogger(module).setLevel(logging.DEBUG)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("fitz").setLevel(logging.WARNING)

    return root_logger


def get_stdlib_logger(name: str) -> logging.Logger:
    """
    Get a stdlib logger for a specific module (LEGACY).

    This function is kept for backward compatibility.
    New code should use get_logger() which returns Loguru logger.

    Args:
        name: Module name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggingContext:
    """
    Context manager for temporary logging level changes.

    Usage:
        with LoggingContext(level="DEBUG"):
            # Debug logging enabled
            ...
    """

    def __init__(self, level: str = "DEBUG", logger: Optional[logging.Logger] = None):
        """
        Initialize logging context.

        Args:
            level: Temporary log level
            logger: Logger to modify (default: root logger)
        """
        self.level = getattr(logging, level.upper())
        self.logger = logger or logging.getLogger()
        self.original_level = None

    def __enter__(self):
        self.original_level = self.logger.level
        self.logger.setLevel(self.level)
        return self

    def __exit__(self, *args):
        if self.original_level is not None:
            self.logger.setLevel(self.original_level)


def log_function_call(func):
    """
    Decorator to log function calls (for debugging).

    Usage:
        @log_function_call
        def my_function(arg1, arg2):
            ...
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger_obj = get_logger(func.__module__)
        logger_obj.debug(f"Calling {func.__name__}(args={len(args)}, kwargs={list(kwargs.keys())})")
        try:
            result = func(*args, **kwargs)
            logger_obj.debug(f"{func.__name__} returned successfully")
            return result
        except Exception as e:
            logger_obj.error(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise

    return wrapper


def log_async_function_call(func):
    """
    Decorator to log async function calls (for debugging).

    Usage:
        @log_async_function_call
        async def my_async_function(arg1, arg2):
            ...
    """
    import functools

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger_obj = get_logger(func.__module__)
        logger_obj.debug(f"Calling async {func.__name__}(args={len(args)}, kwargs={list(kwargs.keys())})")
        try:
            result = await func(*args, **kwargs)
            logger_obj.debug(f"async {func.__name__} returned successfully")
            return result
        except Exception as e:
            logger_obj.error(f"async {func.__name__} raised {type(e).__name__}: {e}")
            raise

    return wrapper


# Default logger for convenience
default_logger = get_logger("ai_correction")
