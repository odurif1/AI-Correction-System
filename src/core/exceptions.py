"""
Custom exception hierarchy for the AI correction system.

Provides a consistent error handling approach across all modules.
"""


class AICorrectionError(Exception):
    """
    Base exception for all AI correction system errors.

    All custom exceptions should inherit from this class.
    """

    def __init__(self, message: str, details: dict | None = None):
        """
        Initialize the exception.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# ==================== Configuration Errors ====================

class ConfigurationError(AICorrectionError):
    """
    Error in system configuration.

    Raised when required configuration is missing or invalid.
    """
    pass


class MissingAPIKeyError(ConfigurationError):
    """Raised when a required API key is not configured."""
    pass


class InvalidModelError(ConfigurationError):
    """Raised when an invalid model name is specified."""
    pass


# ==================== Provider Errors ====================

class ProviderError(AICorrectionError):
    """
    Base error for AI provider issues.

    Raised when there's a problem with an AI provider.
    """
    pass


class APIConnectionError(ProviderError):
    """Raised when connection to AI API fails."""
    pass


class APITimeoutError(ProviderError):
    """Raised when an AI API call times out."""
    pass


class APIRateLimitError(ProviderError):
    """Raised when API rate limit is exceeded."""
    pass


class APIResponseError(ProviderError):
    """Raised when API returns an unexpected or invalid response."""
    pass


class ParsingError(ProviderError):
    """Raised when parsing AI response fails."""
    pass


# ==================== Session Errors ====================

class SessionError(AICorrectionError):
    """
    Base error for session-related issues.
    """
    pass


class SessionNotFoundError(SessionError):
    """Raised when a requested session doesn't exist."""
    pass


class SessionStateError(SessionError):
    """Raised when session is in an invalid state for the operation."""
    pass


class SessionLockError(SessionError):
    """Raised when session file lock cannot be acquired."""
    pass


# ==================== Grading Errors ====================

class GradingError(AICorrectionError):
    """
    Base error for grading-related issues.
    """
    pass


class ScaleDetectionError(GradingError):
    """Raised when grading scale cannot be detected."""
    pass


class QuestionNotFoundError(GradingError):
    """Raised when a question is not found in the document."""
    pass


class GradingDisagreementError(GradingError):
    """Raised when LLMs disagree and no resolution is possible."""
    pass


class StudentNameMismatchError(GradingError):
    """
    Raised when LLMs detect different student names for the same copy.

    This indicates the LLMs are not aligned on which copies they're grading
    and requires user intervention (--pages-per-copy or --auto-detect-structure).
    """

    def __init__(self, message: str, mismatches: list = None, llm1_only: list = None, llm2_only: list = None):
        super().__init__(message, {
            'mismatches': mismatches or [],
            'llm1_only': llm1_only or [],
            'llm2_only': llm2_only or []
        })
        self.mismatches = mismatches or []
        self.llm1_only = llm1_only or []
        self.llm2_only = llm2_only or []


# ==================== PDF Errors ====================

class PDFError(AICorrectionError):
    """
    Base error for PDF processing issues.
    """
    pass


class PDFReadError(PDFError):
    """Raised when PDF cannot be read."""
    pass


class PDFConversionError(PDFError):
    """Raised when PDF to image conversion fails."""
    pass


class InvalidPDFError(PDFError):
    """Raised when PDF is invalid or corrupted."""
    pass


# ==================== Storage Errors ====================

class StorageError(AICorrectionError):
    """
    Base error for storage-related issues.
    """
    pass


class FileAccessError(StorageError):
    """Raised when file access fails."""
    pass


class SerializationError(StorageError):
    """Raised when serialization/deserialization fails."""
    pass


class ValidationFailedError(StorageError):
    """Raised when data validation fails."""
    pass


# ==================== Export Errors ====================

class ExportError(AICorrectionError):
    """
    Base error for export-related issues.
    """
    pass


class UnsupportedFormatError(ExportError):
    """Raised when unsupported export format is requested."""
    pass


class ExportGenerationError(ExportError):
    """Raised when export generation fails."""
    pass
