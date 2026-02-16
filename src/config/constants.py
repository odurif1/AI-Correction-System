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
