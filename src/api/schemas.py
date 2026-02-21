"""
Pydantic schemas for API request/response validation.

These schemas define the structure and validation rules for all API endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime


# ============================================================================
# Session Schemas
# ============================================================================

class CreateSessionRequest(BaseModel):
    """Request to create a new grading session."""
    subject: Optional[str] = None
    topic: Optional[str] = None
    total_questions: int = 0
    question_weights: Dict[str, float] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    """Session information."""
    session_id: str
    status: str
    created_at: str
    copies_count: int
    graded_count: int
    average_score: Optional[float] = None
    subject: Optional[str] = None
    topic: Optional[str] = None


class SessionDetailResponse(SessionResponse):
    """Detailed session information including copies and grades."""
    copies: List[Dict[str, Any]] = Field(default_factory=list)
    graded_copies: List[Dict[str, Any]] = Field(default_factory=list)
    question_weights: Dict[str, float] = Field(default_factory=dict)


# ============================================================================
# Grading Schemas
# ============================================================================

class GradeResponse(BaseModel):
    """Grading result."""
    success: bool
    session_id: str
    graded_count: int
    total_count: int
    pending_review: int


class TeacherDecisionRequest(BaseModel):
    """Request to provide teacher guidance."""
    question_id: str
    copy_id: str
    teacher_guidance: str
    original_score: float
    new_score: float
    applies_to_all: bool = True


# ============================================================================
# Disagreement Schemas
# ============================================================================

class LLMGradeInfo(BaseModel):
    """Information about an LLM's grading decision."""
    provider: str
    model: str
    grade: float
    confidence: float = 0.5
    reasoning: Optional[str] = None
    reading: Optional[str] = None


class DisagreementResponse(BaseModel):
    """A disagreement between two LLMs."""
    copy_id: str
    copy_index: int
    student_name: Optional[str] = None
    question_id: str
    max_points: float
    llm1: LLMGradeInfo
    llm2: LLMGradeInfo
    resolved: bool = False


class ResolveDisagreementRequest(BaseModel):
    """Request to resolve a disagreement."""
    action: Literal["llm1", "llm2", "average", "custom"]
    custom_grade: Optional[float] = None
    teacher_guidance: Optional[str] = None


# ============================================================================
# Analytics Schemas
# ============================================================================

class AnalyticsResponse(BaseModel):
    """Analytics data."""
    mean_score: float
    median_score: float
    min_score: float
    max_score: float
    std_dev: float
    score_distribution: Dict[str, int]
    question_stats: Optional[Dict[str, Dict[str, float]]] = None


# ============================================================================
# Provider Schemas
# ============================================================================

class ProviderModel(BaseModel):
    """A model available from a provider."""
    id: str
    name: str
    context_window: Optional[int] = None
    pricing: Optional[Dict[str, float]] = None


class ProviderResponse(BaseModel):
    """A configured LLM provider."""
    id: str
    name: str
    type: str  # "gemini", "openai", "openrouter"
    models: List[ProviderModel] = Field(default_factory=list)
    configured: bool = False


# ============================================================================
# Settings Schemas
# ============================================================================

class SettingsResponse(BaseModel):
    """Application settings."""
    ai_provider: str
    comparison_mode: bool
    llm1_provider: Optional[str] = None
    llm1_model: Optional[str] = None
    llm2_provider: Optional[str] = None
    llm2_model: Optional[str] = None
    confidence_auto: float = 0.85
    confidence_flag: float = 0.60


class UpdateSettingsRequest(BaseModel):
    """Request to update settings."""
    comparison_mode: Optional[bool] = None
    llm1_provider: Optional[str] = None
    llm1_model: Optional[str] = None
    llm2_provider: Optional[str] = None
    llm2_model: Optional[str] = None
    confidence_auto: Optional[float] = None
    confidence_flag: Optional[float] = None


# ============================================================================
# Export Schemas
# ============================================================================

class ExportOptions(BaseModel):
    """Options for exporting graded copies."""
    format: Literal["pdf", "json", "csv", "all"] = "pdf"
    include_feedback: bool = True
    include_reasoning: bool = False
    include_analytics: bool = True


# ============================================================================
# Progress Event Schemas (for WebSocket)
# ============================================================================

class ProgressEvent(BaseModel):
    """A progress event sent via WebSocket."""
    type: str
    timestamp: Optional[str] = None

    # Common fields
    session_id: Optional[str] = None
    copy_index: Optional[int] = None

    # copy_start fields
    total_copies: Optional[int] = None
    copy_id: Optional[str] = None
    student_name: Optional[str] = None
    questions: Optional[List[str]] = None

    # question_done fields
    question_id: Optional[str] = None
    grade: Optional[float] = None
    max_points: Optional[float] = None
    method: Optional[str] = None
    agreement: Optional[bool] = None

    # copy_done fields
    total_score: Optional[float] = None
    max_score: Optional[float] = None
    confidence: Optional[float] = None

    # copy_error fields
    error: Optional[str] = None

    # session_complete fields
    average_score: Optional[float] = None
