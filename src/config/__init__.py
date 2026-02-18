"""
Configuration module for the AI correction system.

Provides settings, constants, and prompt templates.
"""

from config.settings import get_settings, Settings
from config.constants import (
    # Model Configuration
    DEFAULT_MODEL,
    DEFAULT_VISION_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    # Gemini Configuration
    GEMINI_MODEL_FLASH,
    GEMINI_MODEL_PRO,
    GEMINI_DEFAULT_MODEL,
    # Confidence Thresholds
    CONFIDENCE_THRESHOLD_AUTO,
    CONFIDENCE_THRESHOLD_FLAG,
    CONFIDENCE_THRESHOLD_ASK,
    # Grading Thresholds
    GRADE_DIFFERENCE_THRESHOLD,
    GRADE_THRESHOLD,
    # Similarity Thresholds
    ANSWER_SIMILARITY_THRESHOLD,
    NAME_SIMILARITY_THRESHOLD,
    SIMILARITY_THRESHOLD,
    # Processing Defaults
    DEFAULT_PARALLEL_COPIES,
    DEFAULT_PAGES_PER_STUDENT,
    # API Timeouts
    API_CONNECT_TIMEOUT,
    API_READ_TIMEOUT,
    API_EMBEDDING_TIMEOUT,
    # Storage
    DATA_DIR,
)

__all__ = [
    'get_settings',
    'Settings',
    'DEFAULT_MODEL',
    'DEFAULT_VISION_MODEL',
    'MAX_TOKENS',
    'TEMPERATURE',
    'GEMINI_MODEL_FLASH',
    'GEMINI_MODEL_PRO',
    'GEMINI_DEFAULT_MODEL',
    'CONFIDENCE_THRESHOLD_AUTO',
    'CONFIDENCE_THRESHOLD_FLAG',
    'CONFIDENCE_THRESHOLD_ASK',
    'GRADE_DIFFERENCE_THRESHOLD',
    'GRADE_THRESHOLD',
    'ANSWER_SIMILARITY_THRESHOLD',
    'NAME_SIMILARITY_THRESHOLD',
    'SIMILARITY_THRESHOLD',
    'DEFAULT_PARALLEL_COPIES',
    'DEFAULT_PAGES_PER_STUDENT',
    'API_CONNECT_TIMEOUT',
    'API_READ_TIMEOUT',
    'API_EMBEDDING_TIMEOUT',
    'DATA_DIR',
]
