"""
Core module for the AI correction system.

Exports key models, exceptions, and workflow state for easy access.
"""

from core.models import (
    GradingSession,
    GradedCopy,
    CopyDocument,
    GradingPolicy,
    TeacherDecision,
    SessionStatus,
    ConfidenceLevel,
    UncertaintyType,
    generate_id,
)

from core.workflow_state import (
    CorrectionState,
    WorkflowPhase,
)

from core.workflow import (
    CorrectionWorkflow,
    WorkflowCallbacks,
    WorkflowConfig,
)

from core.exceptions import (
    AICorrectionError,
    ConfigurationError,
    MissingAPIKeyError,
    InvalidModelError,
    ProviderError,
    APIConnectionError,
    APITimeoutError,
    APIRateLimitError,
    APIResponseError,
    ParsingError,
    SessionError,
    SessionNotFoundError,
    SessionStateError,
    SessionLockError,
    GradingError,
    ScaleDetectionError,
    QuestionNotFoundError,
    GradingDisagreementError,
    PDFError,
    PDFReadError,
    PDFConversionError,
    InvalidPDFError,
    StorageError,
    FileAccessError,
    SerializationError,
    ValidationFailedError,
    ExportError,
    UnsupportedFormatError,
    ExportGenerationError,
)

__all__ = [
    # Models
    'GradingSession',
    'GradedCopy',
    'CopyDocument',
    'GradingPolicy',
    'TeacherDecision',
    'SessionStatus',
    'ConfidenceLevel',
    'UncertaintyType',
    'generate_id',
    # Workflow State
    'CorrectionState',
    'WorkflowPhase',
    # Workflow
    'CorrectionWorkflow',
    'WorkflowCallbacks',
    'WorkflowConfig',
    # Exceptions
    'AICorrectionError',
    'ConfigurationError',
    'MissingAPIKeyError',
    'InvalidModelError',
    'ProviderError',
    'APIConnectionError',
    'APITimeoutError',
    'APIRateLimitError',
    'APIResponseError',
    'ParsingError',
    'SessionError',
    'SessionNotFoundError',
    'SessionStateError',
    'SessionLockError',
    'GradingError',
    'ScaleDetectionError',
    'QuestionNotFoundError',
    'GradingDisagreementError',
    'PDFError',
    'PDFReadError',
    'PDFConversionError',
    'InvalidPDFError',
    'StorageError',
    'FileAccessError',
    'SerializationError',
    'ValidationFailedError',
    'ExportError',
    'UnsupportedFormatError',
    'ExportGenerationError',
]
