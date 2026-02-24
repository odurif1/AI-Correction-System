"""
Core data models for the AI correction system.

This module defines all Pydantic models used throughout the system.
They represent the foundation of all data structures.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Literal, Any
from datetime import datetime
from enum import Enum
import uuid


class ConfidenceLevel(str, Enum):
    """Confidence level categories for grading decisions."""
    HIGH = "high"       # >= 0.85: Auto-grade
    MEDIUM = "medium"   # 0.60 - 0.85: Grade + flag for review
    LOW = "low"         # < 0.60: Pause, ask teacher


class UncertaintyType(str, Enum):
    """Types of uncertainty the AI might encounter."""
    NONE = "none"
    ALTERNATIVE_METHOD = "alternative_method"
    AMBIGUOUS = "ambiguous"
    UNEXPECTED_APPROACH = "unexpected_approach"
    INCOMPLETE = "incomplete"
    OTHER = "other"


class SessionStatus(str, Enum):
    """Status of a grading session."""
    ANALYZING = "analyzing"
    GRADING = "grading"
    CALIBRATING = "calibrating"
    COMPLETE = "complete"
    PAUSED = "paused"


def generate_id() -> str:
    """
    Generate a unique ID.

    Uses full UUID to avoid collision risks.
    For display purposes, callers can truncate to first 8 characters.
    """
    return str(uuid.uuid4())


class CopyDocument(BaseModel):
    """
    Represents a student's copy with content extracted via vision-language AI.

    The AI "sees" and understands the handwriting/content rather than
    performing raw OCR text extraction.
    """
    id: str = Field(default_factory=generate_id)
    pdf_path: str
    page_count: int = 1
    student_name: Optional[str] = None
    student_id: Optional[str] = None

    # Detected language of the copy (auto-detected from content)
    language: Optional[str] = None  # 'fr', 'en', 'es', etc.

    # Content extracted by AI (semantic understanding, not raw text)
    content_summary: Dict[str, str] = Field(default_factory=dict)
    # {question_id: summary of student's answer}

    # For clustering and similarity detection
    embedding: Optional[List[float]] = None
    cluster_id: Optional[int] = None

    # Raw images for vision processing
    page_images: List[str] = Field(default_factory=list)  # Paths to page images

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    processed: bool = False


class GradingPolicy(BaseModel):
    """
    The AI's "mental model" - evolves based on teacher interactions.

    This represents the AI's understanding of how to grade this specific
    assessment. It updates as the teacher provides guidance.
    """
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Grading criteria per question
    criteria: Dict[str, str] = Field(default_factory=dict)
    # {question_id: evaluation_guidance}

    # Rules learned from teacher decisions
    teacher_decisions: List[str] = Field(default_factory=list)

    # Confidence thresholds for auto-grading
    confidence_thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "auto": 0.85,
        "flag": 0.60,
        "ask": 0.60
    })

    # Weighting per question
    question_weights: Dict[str, float] = Field(default_factory=dict)

    # General subject context
    subject: Optional[str] = None
    topic: Optional[str] = None


class GradingSession(BaseModel):
    """
    Main session model - represents the global state of a grading session.

    Orchestrates the entire workflow from PDF input to corrected output.
    """
    session_id: str = Field(default_factory=generate_id)
    created_at: datetime = Field(default_factory=datetime.now)
    status: SessionStatus = SessionStatus.ANALYZING

    # Data
    copies: List[CopyDocument] = Field(default_factory=list)
    graded_copies: List['GradedCopy'] = Field(default_factory=list)
    policy: GradingPolicy = Field(default_factory=GradingPolicy)

    # Cross-copy analysis
    clusters: Dict[int, List[str]] = Field(default_factory=dict)
    # {cluster_id: [copy_ids]}
    class_map: Optional['ClassAnswerMap'] = None

    # Metadata
    total_questions: int = 0
    copies_processed: int = 0

    # Teacher interactions
    pending_decisions: List[str] = Field(default_factory=list)
    teacher_corrections: int = 0

    # Storage paths
    storage_path: Optional[str] = None

    # Individual reading mode (PDF pre-split)
    pages_per_copy: Optional[int] = None  # If set, activates individual mode


class GradedCopy(BaseModel):
    """
    Represents a corrected student copy.

    Contains grades, confidence scores, reasoning and student_feedback for each question.
    """
    copy_id: str
    graded_at: datetime = Field(default_factory=datetime.now)
    policy_version: int = 1
    # Used to know if re-calibration is needed

    # Grades
    grades: Dict[str, float] = Field(default_factory=dict)
    # {question_id: points_earned}

    total_score: float = 0.0
    max_score: float = 0.0

    # AI confidence and reasoning
    confidence: float = 0.0  # 0-1 overall confidence
    confidence_by_question: Dict[str, float] = Field(default_factory=dict)

    # Feedback fields
    reasoning: Dict[str, str] = Field(default_factory=dict)
    # {question_id: technical analysis for teachers}
    student_feedback: Dict[str, str] = Field(default_factory=dict)
    # {question_id: pedagogical feedback for students}

    # Readings (what the AI read from the student's answer)
    readings: Dict[str, str] = Field(default_factory=dict)
    # {question_id: text read from student's answer}

    # Detected max points per question (from document)
    max_points_by_question: Dict[str, float] = Field(default_factory=dict)
    # {question_id: max_points detected from document}

    # Unified grading audit (replaces deprecated llm_comparison)
    grading_audit: Optional[GradingAudit] = None

    uncertainty_type: UncertaintyType = UncertaintyType.NONE

    # Teacher review
    teacher_reviewed: bool = False
    adjustments: List['TeacherAdjustment'] = Field(default_factory=list)

    # Feedback
    feedback: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED GRADING AUDIT MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ProviderInfo(BaseModel):
    """Information about an LLM provider used in grading."""
    id: str  # "LLM1" or "LLM2"
    model: str
    tokens: Optional[Dict[str, int]] = None  # {"prompt": int, "completion": int}


class LLMResult(BaseModel):
    """Result from a single LLM for a question."""
    grade: float
    max_points: float  # Each LLM detects its own barème
    reading: str = ""
    reasoning: str = ""
    feedback: str = ""
    confidence: float = 0.8


class ResolutionInfo(BaseModel):
    """Resolution information for a question after LLM comparison."""
    final_grade: float
    final_max_points: float  # Barème retained after resolution
    method: str  # consensus, average, verification_consensus, ultimatum_average, etc.
    phases: List[str]  # ["initial"], ["verification"], ["verification", "ultimatum"], etc.
    agreement: Optional[bool] = None  # null for single LLM
    initial_reading_similarity: Optional[float] = None  # SequenceMatcher ratio between initial LLM readings


class QuestionAudit(BaseModel):
    """Audit information for a single question."""
    llm_results: Dict[str, LLMResult]  # "LLM1" -> {...}, "LLM2" -> {...}
    resolution: ResolutionInfo


class StudentDetectionAudit(BaseModel):
    """Audit information for student name detection."""
    final_name: str
    llm_results: Dict[str, str]  # "LLM1" -> "Jean Dupont", "LLM2" -> "J. Dupont"
    resolution: ResolutionInfo


class AuditSummary(BaseModel):
    """Summary statistics for a grading audit."""
    total_questions: int
    agreed_initial: int
    required_verification: int = 0
    required_ultimatum: int = 0
    final_agreement_rate: float


class GradingAudit(BaseModel):
    """
    Unified audit structure for all grading modes.

    Works for:
    - Single LLM mode (mode="single", one provider)
    - Dual LLM mode (mode="dual", two providers)
    - All verification modes (grouped, per-copy, per-question, none)
    """
    mode: Literal["single", "dual"]
    grading_method: Literal["batch", "individual", "hybrid"]
    verification_mode: Literal["grouped", "per-copy", "per-question", "none"]

    providers: List[ProviderInfo]
    questions: Dict[str, QuestionAudit]  # "Q1" -> {...}
    student_detection: Optional[StudentDetectionAudit] = None
    summary: AuditSummary


# ═══════════════════════════════════════════════════════════════════════════════
# TEACHER INTERACTION MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class TeacherDecision(BaseModel):
    """
    Represents a teacher's decision for propagation to similar copies.

    When a teacher adjusts a grade, the system extracts a generalizable rule
    and offers to apply it to similar answers.
    """
    decision_id: str = Field(default_factory=generate_id)
    timestamp: datetime = Field(default_factory=datetime.now)

    question_id: str
    source_copy_id: str

    # What the teacher said
    teacher_guidance: str

    # What the AI extracted from it
    extracted_rule: Optional[str] = None

    # Scope
    applies_to_all: bool = True
    # If True, propose to all similar copies
    similar_copy_ids: List[str] = Field(default_factory=list)

    # Original vs new
    original_score: Optional[float] = None
    new_score: Optional[float] = None


class TeacherAdjustment(BaseModel):
    """
    Records a teacher's adjustment to a specific copy.

    Unlike TeacherDecision, this is local to one copy.
    """
    question_id: str
    original_score: float
    new_score: float
    reason: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ClassAnswerMap(BaseModel):
    """
    Result of cross-copy analysis.

    Contains clusters, patterns, and outliers detected across all copies.
    """
    question_analyses: Dict[str, 'QuestionAnalysis'] = Field(default_factory=dict)
    clusters: Dict[str, List['AnswerCluster']] = Field(default_factory=dict)
    # {question_id: [clusters]}

    outliers: List['AnswerOutlier'] = Field(default_factory=list)

    # Timestamp
    analyzed_at: datetime = Field(default_factory=datetime.now)


class QuestionAnalysis(BaseModel):
    """Analysis of a question across all copies."""
    question_id: str
    question_text: Optional[str] = None

    # Patterns detected
    common_correct: List[str] = Field(default_factory=list)
    common_errors: List[str] = Field(default_factory=list)
    unique_approaches: List[str] = Field(default_factory=list)

    # Difficulty estimate (0-1)
    difficulty_estimate: float = 0.5

    # Answer distribution
    answer_distribution: Dict[str, int] = Field(default_factory=dict)


class AnswerCluster(BaseModel):
    """
    A cluster of similar answers.

    Uses embeddings and semantic similarity to group similar responses.
    """
    cluster_id: int
    question_id: str
    copy_ids: List[str] = Field(default_factory=list)

    # Representative answer for this cluster
    representative_description: str
    representative_answer: Optional[str] = None

    # Grading (once graded)
    avg_grade: Optional[float] = None
    grade_variance: Optional[float] = None

    # Centroid for similarity calculations
    centroid_embedding: Optional[List[float]] = None


class AnswerOutlier(BaseModel):
    """
    An answer that is notably different from others.

    May indicate: exceptional answer, misunderstanding, or need for attention.
    """
    copy_id: str
    question_id: str
    description: str

    # Why it's an outlier
    outlier_reason: str

    # Suggested action
    suggested_action: Literal["review", "auto_accept", "ask_teacher"] = "review"


class SimilarCopies(BaseModel):
    """
    Result of finding copies similar to a source for retroactive application.
    """
    already_graded: List[str] = Field(default_factory=list)
    # Copies that were already graded and need re-grading

    pending: List[str] = Field(default_factory=list)
    # Copies not yet graded (will use new rule)

    # Similarity scores
    similarity_scores: Dict[str, float] = Field(default_factory=dict)
    # {copy_id: similarity_score}


class InconsistencyReport(BaseModel):
    """
    Report of grading inconsistencies detected during calibration.

    Highlights where similar answers received different grades.
    """
    question_id: str
    copy_ids: List[str]

    # The inconsistent grades
    grades: Dict[str, float] = Field(default_factory=dict)
    # {copy_id: grade}

    # Difference
    max_difference: float

    # Suggested resolution
    suggested_grade: Optional[float] = None
    reasoning: Optional[str] = None


class AICallResult(BaseModel):
    """
    Result of an AI call with metadata for audit trail.
    """
    call_id: str = Field(default_factory=generate_id)
    timestamp: datetime = Field(default_factory=datetime.now)

    prompt_type: str
    # "grade", "extract_rule", "analyze", etc.

    input_summary: str
    response_summary: str

    # Timing
    duration_ms: Optional[float] = None

    # Tokens (if available)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class ExportOptions(BaseModel):
    """Options for exporting graded copies."""
    format: Literal["pdf", "json", "csv", "all"] = "pdf"
    include_feedback: bool = True
    include_reasoning: bool = False
    include_analytics: bool = True


class AnalyticsReport(BaseModel):
    """Analytics report for the class."""
    session_id: str

    # Score statistics
    mean_score: float
    median_score: float
    std_dev: float
    min_score: float
    max_score: float

    # Score distribution
    score_distribution: Dict[str, int] = Field(default_factory=dict)
    # {"0-5": count, "5-10": count, ...}

    # Per-question statistics
    question_stats: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    # {question_id: {mean, median, difficulty}}

    # Common patterns
    common_errors: List[str] = Field(default_factory=list)
    exceptional_answers: List[str] = Field(default_factory=list)

    # Generated at
    generated_at: datetime = Field(default_factory=datetime.now)
