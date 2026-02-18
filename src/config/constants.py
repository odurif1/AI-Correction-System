"""
Constants and configuration values for the AI correction system.

Defines thresholds, defaults, and system-wide constants.
"""

from typing import Dict, Final

# AI Model Configuration
DEFAULT_MODEL: Final[str] = "gpt-4o"
DEFAULT_VISION_MODEL: Final[str] = "gpt-4o"
MAX_TOKENS: Final[int] = 4096
TEMPERATURE: Final[float] = 0.3  # Lower temperature for consistent grading

# Gemini Model Configuration
GEMINI_MODEL_FLASH: Final[str] = "gemini-2.5-flash"
GEMINI_MODEL_PRO: Final[str] = "gemini-3-pro"
GEMINI_MODEL_EMBEDDING: Final[str] = "text-embedding-004"
GEMINI_DEFAULT_MODEL: Final[str] = GEMINI_MODEL_FLASH  # Use flash for testing
GEMINI_DEFAULT_VISION_MODEL: Final[str] = GEMINI_MODEL_FLASH
GEMINI_EMBEDDING_DIM: Final[int] = 768  # Dimension for text-embedding-004

# Confidence Thresholds
CONFIDENCE_THRESHOLD_AUTO: Final[float] = 0.85
CONFIDENCE_THRESHOLD_FLAG: Final[float] = 0.60
CONFIDENCE_THRESHOLD_ASK: Final[float] = 0.60

# Grading Thresholds
GRADE_DIFFERENCE_THRESHOLD: Final[float] = 0.5  # Difference to trigger cross-verification
GRADE_THRESHOLD: Final[float] = 0.5  # Minimum grade difference to consider significant
JACCARD_SIMILARITY_THRESHOLD: Final[float] = 0.3  # For reading compatibility
READING_LENGTH_RATIO_THRESHOLD: Final[float] = 1.5  # For choosing longer reading

# Similarity Thresholds
ANSWER_SIMILARITY_THRESHOLD: Final[float] = 0.8  # For considering answers "similar"
NAME_SIMILARITY_THRESHOLD: Final[float] = 0.85  # For name matching

# Processing Defaults
DEFAULT_PARALLEL_COPIES: Final[int] = 6  # Number of copies to process in parallel
DEFAULT_PAGES_PER_STUDENT: Final[int] = 2  # Default pages per student

# Clustering
EMBEDDING_MODEL: Final[str] = "text-embedding-3-small"
EMBEDDING_DIM: Final[int] = 1536
DBSCAN_EPS: Final[float] = 0.3
DBSCAN_MIN_SAMPLES: Final[int] = 2
SIMILARITY_THRESHOLD: Final[float] = 0.85  # For considering answers "similar"

# PDF Processing
PDF_DPI: Final[int] = 200
MAX_PAGE_SIZE: Final[int] = 10 * 1024 * 1024  # 10MB
SUPPORTED_IMAGE_FORMATS: Final[tuple] = (".png", ".jpg", ".jpeg")

# Storage - Simplified Architecture
DATA_DIR: Final[str] = "data"
SESSIONS_INDEX: Final[str] = "_index.json"

# File naming
SESSION_JSON: Final[str] = "session.json"
POLICY_JSON: Final[str] = "policy.json"
CLUSTERS_JSON: Final[str] = "clusters.json"
AUDIT_LOG: Final[str] = "audit.log"
GRADES_CSV: Final[str] = "grades.csv"
FULL_REPORT_JSON: Final[str] = "full_report.json"

# Grading Defaults
DEFAULT_MAX_SCORE: Final[float] = 20.0
DEFAULT_QUESTION_POINTS: Final[float] = 1.0

# Retry Configuration
MAX_RETRIES: Final[int] = 3
RETRY_DELAY_MS: Final[int] = 1000

# UI/CLI
PROGRESS_BAR_WIDTH: Final[int] = 50
CONFIDENCE_COLORS: Final[Dict[str, str]] = {
    "high": "green",
    "medium": "yellow",
    "low": "red"
}

# Annotation
ANNOTATION_FONT_SIZE: Final[int] = 10
ANNOTATION_COLOR_CORRECT: Final[tuple] = (0, 0.8, 0)  # Green
ANNOTATION_COLOR_PARTIAL: Final[tuple] = (1, 0.6, 0)  # Orange
ANNOTATION_COLOR_WRONG: Final[tuple] = (0.8, 0, 0)  # Red
ANNOTATION_ALPHA: Final[float] = 0.3

# API
API_HOST: Final[str] = "127.0.0.1"
API_PORT: Final[int] = 8000
API_WORKERS: Final[int] = 1

# API Timeouts (in seconds)
API_CONNECT_TIMEOUT: Final[float] = 30.0  # Connection timeout
API_READ_TIMEOUT: Final[float] = 120.0  # Read timeout for API calls
API_EMBEDDING_TIMEOUT: Final[float] = 60.0  # Timeout for embedding calls

# Prompt Configuration
DEFAULT_LANGUAGE: Final[str] = "fr"  # Default language for prompts
MAX_FEEDBACK_WORDS: Final[int] = 25  # Maximum words in student feedback
MIN_CONFIDENCE_FOR_AUTO: Final[float] = 0.6  # Minimum confidence for auto-grade
SUPPORTED_LANGUAGES: Final[tuple] = ("fr", "en", "es", "de", "it")

# Language Detection Keywords (for detect_language)
LANGUAGE_KEYWORDS: Final[Dict[str, list]] = {
    "fr": ["le", "la", "les", "des", "du", "de", "et", "est", "en", "un", "une"],
    "en": ["the", "a", "an", "is", "are", "of", "to", "in", "and", "or", "but"],
}
