"""
FastAPI application.

Provides the public grading API with WebSocket support for real-time progress.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends, Security, Request, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import shutil
import uuid
import re
import json
import logging
import os
import time
import unicodedata

# Rate limiting
from slowapi.errors import RateLimitExceeded
from api.rate_limiter import limiter

# Maximum file size for uploads (50 MB)
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

# Maximum batch size for PDF uploads
MAX_BATCH_SIZE = 50

# Import workflow state before defining constant
from core.workflow_state import CorrectionState
from core.models import (
    DocumentPageClassification,
    DocumentDecision,
    DocumentSegment,
    DocumentStatus,
    DocumentType,
    SessionDocument,
    SessionStatus,
)

# API runs in auto-mode (no CLI interaction)
API_WORKFLOW_STATE = CorrectionState(auto_mode=True)


from config.settings import get_settings
from pydantic import ValidationError
from core.session import GradingSessionOrchestrator
from core.services.document_preparation_service import DocumentPreparationService
from core.services.review_context_service import ReviewContextService
from storage.session_store import SessionStore
from api.websocket import manager as ws_manager
from api.schemas import (
    CreateSessionRequest, SessionResponse, SessionDetailResponse,
    GradeResponse, TeacherDecisionRequest,
    DisagreementResponse, ResolveDisagreementRequest,
    AnalyticsResponse, ProviderResponse, ProviderModel,
    SettingsResponse, UpdateSettingsRequest, ExportOptions,
    DetectionRequest, DetectionResponse, ConfirmDetectionRequest,
    ConfirmDetectionResponse, StudentInfoSchema, CandidateScale,
    StartGradingRequest, UpdateGradeRequest, UpdateGradeResponse, UpdateStudentNameRequest,
    UpdateSessionSubjectRequest, UpdateQuestionWeightRequest, UpdateQuestionWeightResponse,
    UpdateQuestionNameRequest, UpdateQuestionNameResponse,
    SessionDocumentResponse, UpdateSessionDocumentRequest, ConfirmSessionDocumentsResponse
)

# Import auth module
from api.auth import router as auth_router, get_current_user, get_admin_user, decode_token
from db import SessionLocal, User

# Import token deduction service
from services.token_service import TokenDeductionService
from services.token_service import InsufficientTokensError, UserNotFoundError, DeductionError

# Import health check router
from api.health import router as health_router

try:
    from api.subscription import router as subscription_router
except ModuleNotFoundError:
    subscription_router = None

# Import metrics collector
from utils.metrics import get_metrics_collector

# Import Sentry error handler
from middleware.error_handler import init_sentry, sentry_exception_handler, set_user_context

# Import correlation ID middleware
from asgi_correlation_id import CorrelationIdMiddleware

# Import structured logging
from loguru import logger
from config.logging_config import setup_structured_logging
from vision.pdf_reader import PDFReader

# Import stdlib logger for compatibility
stdlib_logger = logging.getLogger(__name__)

# ============================================================================
# WebSocket Progress Event Types
# ============================================================================

# Progress event types for real-time grading updates
PROGRESS_EVENT_COPY_START = "copy_start"
PROGRESS_EVENT_QUESTION_DONE = "question_done"
PROGRESS_EVENT_COPY_DONE = "copy_done"
PROGRESS_EVENT_COPY_ERROR = "copy_error"
PROGRESS_EVENT_SESSION_COMPLETE = "session_complete"
PROGRESS_EVENT_SESSION_ERROR = "session_error"

# API Key authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify API key from header."""
    expected_key = os.getenv("AI_CORRECTION_API_KEY", "")

    # If no API key is configured, allow all requests (development mode)
    if not expected_key:
        return "dev_mode"

    if not api_key or api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return api_key


def extract_websocket_token(websocket: WebSocket) -> Optional[str]:
    """Extract a bearer token from the WebSocket query string or headers."""
    token = websocket.query_params.get("token")
    if token:
        return token

    auth_header = websocket.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()

    return None


def verify_websocket_session_access(session_id: str, token: Optional[str]) -> Optional[str]:
    """Validate WebSocket access and return the authorized user_id."""
    if not token:
        return None

    payload = decode_token(token)
    if not payload:
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return None
    finally:
        db.close()

    store = SessionStore(session_id, user_id=user_id)
    if not store.exists():
        return None

    return user_id


def serialize_progress_for_client(progress: Dict[str, Any]) -> Dict[str, Any]:
    """Strip internal fields before sending progress state to the client."""
    return {
        "status": progress.get("status"),
        "copies_uploaded": progress.get("copies_uploaded", 0),
        "copies_graded": progress.get("copies_graded", 0),
        "grading_mode": progress.get("grading_mode", "dual"),
        "error": progress.get("error"),
    }


def get_session_temp_dir(session_id: str) -> Path:
    """Return the temporary upload directory for a session."""
    return Path("temp") / session_id


def list_uploaded_pdfs(session_id: str) -> List[Path]:
    """Return uploaded PDFs for a session in deterministic order."""
    upload_dir = get_session_temp_dir(session_id)
    if not upload_dir.exists():
        return []
    return sorted(upload_dir.glob("*.pdf"))


def normalize_document_text(value: str) -> str:
    """Normalize text for robust keyword matching."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_only.lower()


def score_document_text(text: str) -> Dict[DocumentType, float]:
    """Score document roles from extracted PDF text."""
    normalized = normalize_document_text(text)

    grading_keywords = (
        "bareme", "corrige", "correction", "solution", "points", "notation",
        "criteres", "elements de correction", "reponses attendues",
    )
    subject_keywords = (
        "exercice", "question", "consigne", "duree", "calculatrice",
        "repondre", "sujet", "partie", "annexe",
    )
    copy_keywords = (
        "nom", "prenom", "classe", "eleve", "copie", "reponse",
        "je pense", "mon calcul", "ma reponse",
    )

    scores = {
        DocumentType.STUDENT_COPIES: 0.0,
        DocumentType.SUBJECT_ONLY: 0.0,
        DocumentType.GRADING_SCHEME: 0.0,
        DocumentType.RANDOM_DOCUMENT: 0.0,
    }

    for keyword in grading_keywords:
        if keyword in normalized:
            scores[DocumentType.GRADING_SCHEME] += 2.0

    for keyword in subject_keywords:
        if keyword in normalized:
            scores[DocumentType.SUBJECT_ONLY] += 1.5

    for keyword in copy_keywords:
        if keyword in normalized:
            scores[DocumentType.STUDENT_COPIES] += 1.2

    if "note" in normalized and "/20" in normalized:
        scores[DocumentType.GRADING_SCHEME] += 1.5
    if "nom" in normalized and "prenom" in normalized:
        scores[DocumentType.STUDENT_COPIES] += 1.5
    if "exercice 1" in normalized or "question 1" in normalized:
        scores[DocumentType.SUBJECT_ONLY] += 1.0

    if max(scores.values()) == 0:
        scores[DocumentType.RANDOM_DOCUMENT] = 0.5

    return scores


def extract_document_evidence(text: str) -> List[str]:
    """Return a few human-readable cues found in the document text."""
    normalized = normalize_document_text(text)
    evidence: List[str] = []
    cue_map = [
        ("nom", "Présence de champs d'identité"),
        ("prenom", "Présence de champs d'identité"),
        ("classe", "Référence à une classe"),
        ("exercice", "Présence d'exercices"),
        ("question", "Présence de questions"),
        ("consigne", "Présence de consignes"),
        ("bareme", "Référence explicite à un barème"),
        ("corrige", "Référence à une correction"),
        ("solution", "Référence à des solutions"),
        ("points", "Référence à des points/notes"),
    ]

    for keyword, label in cue_map:
        if keyword in normalized and label not in evidence:
            evidence.append(label)
        if len(evidence) >= 4:
            break

    return evidence


def classify_text_snippet(text: str) -> tuple[DocumentType, float, List[str], Optional[str]]:
    """Classify a single text snippet, typically one page."""
    scores = score_document_text(text)
    evidence = extract_document_evidence(text)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_type, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    excerpt = text[:180].strip() or None

    if best_score <= 0:
        return DocumentType.UNCLEAR, 0.1, evidence, excerpt
    if best_score - second_score < 1.0:
        return DocumentType.UNCLEAR, min(0.45, max(0.2, best_score / 10)), evidence, excerpt
    return best_type, min(0.95, 0.35 + best_score / 10), evidence, excerpt


def build_document_segments(page_classifications: List[DocumentPageClassification]) -> List[DocumentSegment]:
    """Merge consecutive pages with the same detected type into segments."""
    if not page_classifications:
        return []

    segments: List[DocumentSegment] = []
    current_start = page_classifications[0].page_number
    current_end = current_start
    current_type = page_classifications[0].detected_type
    confidences = [page_classifications[0].confidence]

    for page in page_classifications[1:]:
        if page.detected_type == current_type and page.page_number == current_end + 1:
            current_end = page.page_number
            confidences.append(page.confidence)
            continue

        segments.append(DocumentSegment(
            start_page=current_start,
            end_page=current_end,
            detected_type=current_type,
            confidence=sum(confidences) / len(confidences),
            page_count=current_end - current_start + 1,
        ))
        current_start = page.page_number
        current_end = page.page_number
        current_type = page.detected_type
        confidences = [page.confidence]

    segments.append(DocumentSegment(
        start_page=current_start,
        end_page=current_end,
        detected_type=current_type,
        confidence=sum(confidences) / len(confidences),
        page_count=current_end - current_start + 1,
    ))
    return segments


def classify_document_pages(
    pdf_path: str,
) -> tuple[int, List[DocumentPageClassification], List[DocumentSegment], str, Dict[str, float]]:
    """Classify each page of a PDF and aggregate contiguous segments."""
    page_count = 0
    page_classifications: List[DocumentPageClassification] = []
    excerpt_parts: List[str] = []
    pages_with_text = 0
    total_text_chars = 0

    with PDFReader(pdf_path) as reader:
        page_count = reader.get_page_count()
        for page_num in range(page_count):
            page_text = reader.extract_text(page_num).strip()
            if page_text:
                pages_with_text += 1
                total_text_chars += len(page_text)
            detected_type, confidence, evidence, excerpt = classify_text_snippet(page_text)
            if excerpt and len(excerpt_parts) < 3:
                excerpt_parts.append(excerpt)
            page_classifications.append(
                DocumentPageClassification(
                    page_number=page_num + 1,
                    detected_type=detected_type,
                    confidence=confidence,
                    evidence=evidence,
                    text_excerpt=excerpt,
                )
            )

    segments = build_document_segments(page_classifications)
    combined_excerpt = "\n".join(excerpt_parts[:3]).strip()
    text_stats = {
        "pages_with_text": float(pages_with_text),
        "page_coverage": (pages_with_text / page_count) if page_count else 0.0,
        "total_text_chars": float(total_text_chars),
        "avg_chars_per_page": (total_text_chars / page_count) if page_count else 0.0,
    }
    return page_count, page_classifications, segments, combined_excerpt, text_stats


def score_document_filename(filename: str) -> Dict[DocumentType, float]:
    """Score document roles from the uploaded filename."""
    name = normalize_document_text(filename)

    grading_keywords = ("bar", "bareme", "corrige", "solution")
    subject_keywords = ("sujet", "enonce", "consigne", "instructions")
    copy_keywords = ("copie", "copies", "eleve", "student", "students", "devoir")

    has_grading = any(keyword in name for keyword in grading_keywords)
    has_subject = any(keyword in name for keyword in subject_keywords)
    has_copy = any(keyword in name for keyword in copy_keywords)

    scores = {
        DocumentType.STUDENT_COPIES: 1.2 if has_copy else 0.0,
        DocumentType.SUBJECT_ONLY: 1.3 if has_subject else 0.0,
        DocumentType.GRADING_SCHEME: 1.4 if has_grading else 0.0,
        DocumentType.RANDOM_DOCUMENT: 0.0,
    }

    return scores


def detect_document_type_from_pdf(
    pdf_path: str,
    filename: str,
) -> tuple[
    DocumentType,
    float,
    List[str],
    int,
    Optional[str],
    List[str],
    List[DocumentPageClassification],
    List[DocumentSegment],
]:
    """Classify an uploaded PDF from both filename and extracted text."""
    issues: List[str] = []
    evidence: List[str] = []
    page_count = 0
    extracted_text = ""
    page_classifications: List[DocumentPageClassification] = []
    segments: List[DocumentSegment] = []
    text_stats = {
        "pages_with_text": 0.0,
        "page_coverage": 0.0,
        "total_text_chars": 0.0,
        "avg_chars_per_page": 0.0,
    }

    filename_scores = score_document_filename(filename)
    content_scores = {
        DocumentType.STUDENT_COPIES: 0.0,
        DocumentType.SUBJECT_ONLY: 0.0,
        DocumentType.GRADING_SCHEME: 0.0,
        DocumentType.RANDOM_DOCUMENT: 0.0,
    }

    try:
        page_count, page_classifications, segments, extracted_text, text_stats = classify_document_pages(pdf_path)
    except Exception as exc:
        issues.append(f"Lecture du PDF incomplète: {exc}")

    weak_text_signal = (
        page_count > 0
        and (
            text_stats["page_coverage"] < 0.5
            or text_stats["avg_chars_per_page"] < 40
        )
    )

    if extracted_text:
        content_scores = score_document_text(extracted_text)
        evidence = extract_document_evidence(extracted_text)
        if not evidence:
            evidence = list(
                dict.fromkeys(
                    item
                    for page in page_classifications
                    for item in page.evidence
                )
            )[:4]
        if weak_text_signal:
            issues.append("Extraction textuelle partielle: la qualification doit être confirmée sur document scanné.")
    else:
        issues.append("Le contenu de ce PDF est difficile à lire sans IA.")

    dominant_page_types = {
        page.detected_type
        for page in page_classifications
        if page.detected_type not in (DocumentType.UNCLEAR, DocumentType.RANDOM_DOCUMENT)
    }
    if len(dominant_page_types) > 1:
        issues.append("Classification ambiguë: document mixte détecté avec plusieurs rôles selon les pages.")
        excerpt = extracted_text[:220].strip() or None
        return (
            DocumentType.UNCLEAR,
            0.35,
            issues,
            page_count,
            excerpt,
            evidence,
            page_classifications,
            segments,
        )

    content_weight = 1.5 if weak_text_signal else 2.5
    combined_scores = {
        doc_type: (content_scores.get(doc_type, 0.0) * content_weight) + filename_scores.get(doc_type, 0.0)
        for doc_type in (
            DocumentType.STUDENT_COPIES,
            DocumentType.SUBJECT_ONLY,
            DocumentType.GRADING_SCHEME,
            DocumentType.RANDOM_DOCUMENT,
        )
    }

    ranked = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    best_type, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    filename_only_signal = (
        not extracted_text
        and not dominant_page_types
        and filename_scores.get(best_type, 0.0) > 0
    )

    if best_score <= 0:
        issues.append("Type de document non déterminé automatiquement.")
        return DocumentType.UNCLEAR, 0.1, issues, page_count, None, evidence, page_classifications, segments

    if best_score - second_score < 1.0:
        issues.append("Classification ambiguë: plusieurs rôles possibles après analyse du contenu.")
        excerpt = extracted_text[:220].strip() or None
        return (
            DocumentType.UNCLEAR,
            min(0.45, max(0.2, best_score / 10)),
            issues,
            page_count,
            excerpt,
            evidence,
            page_classifications,
            segments,
        )

    confidence = min(0.95, 0.35 + best_score / 10)
    if filename_only_signal:
        issues.append("Classification basée principalement sur le nom du fichier: confirmation manuelle recommandée.")
        confidence = min(confidence, 0.3)
    elif weak_text_signal:
        confidence = min(confidence, 0.55)
    excerpt = extracted_text[:220].strip() or None
    return best_type, confidence, issues, page_count, excerpt, evidence, page_classifications, segments


def serialize_session_document(document: SessionDocument) -> SessionDocumentResponse:
    """Convert an internal document model to API response schema."""
    return SessionDocumentResponse(
        document_id=document.id,
        filename=document.filename,
        storage_path=document.storage_path,
        page_count=document.page_count,
        status=document.status.value,
        detected_type=document.detected_type.value,
        confidence=document.confidence,
        issues=document.issues,
        evidence=document.evidence,
        text_excerpt=document.text_excerpt,
        page_classifications=[item.model_dump(mode="json") for item in document.page_classifications],
        segments=[item.model_dump(mode="json") for item in document.segments],
        user_decision=document.user_decision.value,
        usable=document.usable,
    )


def get_disagreement_answer_excerpt(
    question_audit: Any,
    copy: Optional[Any],
) -> tuple[Optional[str], Optional[str]]:
    """Return a central excerpt only when the system has a legitimate final reading."""
    resolution = getattr(question_audit, "resolution", None)
    if resolution and resolution.reading_consensus and resolution.final_reading:
        return resolution.final_reading.strip(), "lecture_finale_consensuelle"

    return None, None


def build_disagreement_response(
    session: Any,
    graded: Any,
    copy: Optional[Any],
    copy_index: int,
    question_id: str,
    question_audit: Any,
    provider1: Optional[Any],
    provider2: Optional[Any],
    llm1_result: Any,
    llm2_result: Any,
    resolved: bool,
) -> DisagreementResponse:
    """Build a professor-facing disagreement with the minimum useful context."""
    answer_excerpt, answer_excerpt_source = get_disagreement_answer_excerpt(
        question_audit,
        copy,
    )
    grade_gap = abs(getattr(llm1_result, "grade", 0) - getattr(llm2_result, "grade", 0))
    reading_similarity = getattr(question_audit.resolution, "initial_reading_similarity", None)
    has_reading_disagreement = (
        question_audit.resolution.reading_consensus is False
        or (reading_similarity is not None and reading_similarity < 0.8)
    )
    has_grade_disagreement = grade_gap > 0.5 or question_audit.resolution.agreement is False
    if has_grade_disagreement and has_reading_disagreement:
        disagreement_type = "grade+reading"
    elif has_reading_disagreement:
        disagreement_type = "reading"
    else:
        disagreement_type = "grade"

    pdf_question_text = review_context_service.extract_question_text_from_pdf(
        copy.pdf_path if copy else None,
        question_hint=session.policy.question_names.get(question_id) or question_id,
    )
    question_text_candidates = []
    if isinstance(pdf_question_text, str) and pdf_question_text.strip():
        question_text_candidates.append(pdf_question_text.strip())
    for candidate in (
        getattr(llm1_result, "question_text", None),
        getattr(llm2_result, "question_text", None),
    ):
        if isinstance(candidate, str):
            normalized = candidate.strip()
            if normalized:
                question_text_candidates.append(normalized)

    question_text = max(question_text_candidates, key=len) if question_text_candidates else None
    review_context = review_context_service.build_context(
        question_text=question_text,
        llm_reasonings=[
            getattr(llm1_result, "reasoning", "") or "",
            getattr(llm2_result, "reasoning", "") or "",
        ],
    )

    return DisagreementResponse(
        copy_id=graded.copy_id,
        copy_index=copy_index,
        student_name=copy.student_name if copy else None,
        question_id=question_id,
        question_label=session.policy.question_names.get(question_id) or question_id,
        question_text=question_text,
        review_context=review_context,
        disagreement_type=disagreement_type,
        answer_excerpt=answer_excerpt,
        answer_excerpt_source=answer_excerpt_source,
        start_page=copy.start_page if copy else None,
        end_page=copy.end_page if copy else None,
        max_points=question_audit.resolution.final_max_points,
        llm1={
            "provider": provider1.model if provider1 else "unknown",
            "model": provider1.model if provider1 else "unknown",
            "grade": llm1_result.grade,
            "confidence": llm1_result.confidence,
            "reasoning": llm1_result.reasoning,
            "reading": llm1_result.reading,
        },
        llm2={
            "provider": provider2.model if provider2 else "unknown",
            "model": provider2.model if provider2 else "unknown",
            "grade": llm2_result.grade,
            "confidence": llm2_result.confidence,
            "reasoning": llm2_result.reasoning,
            "reading": llm2_result.reading,
        },
        resolved=resolved,
    )


document_preparation_service = DocumentPreparationService()
review_context_service = ReviewContextService()


def get_session_copy_documents(session) -> List[SessionDocument]:
    """Return documents that should be used as student copies for the session."""
    confirmed = document_preparation_service.get_copy_documents(session)
    if confirmed:
        return confirmed

    classified = [
        doc for doc in session.documents
        if doc.detected_type == DocumentType.STUDENT_COPIES
        and doc.user_decision != DocumentDecision.EXCLUDE
    ]
    return classified


def estimate_session_token_budget(session) -> int:
    """Estimate token usage for a grading run using the session's copy pages."""
    estimated_tokens_per_page = 10_000  # Conservative sizing for session-level budgeting
    copy_documents = get_session_copy_documents(session) if session else []
    total_pages = sum(max(1, doc.page_count or 0) for doc in copy_documents)

    if total_pages == 0 and session:
        copies = getattr(session, "copies", []) or []
        total_pages = sum(max(1, getattr(copy, "page_count", 0) or 0) for copy in copies)

    if total_pages == 0 and session:
        total_pages = max(1, getattr(session, "copies_count", 0) or 0)

    return total_pages * estimated_tokens_per_page


def estimate_pages_from_tokens(token_count: int) -> int:
    """Convert internal token capacity to a user-facing page estimate."""
    estimated_tokens_per_page = 10_000
    if token_count <= 0:
        return 0
    return max(1, round(token_count / estimated_tokens_per_page))


def format_capacity_error(required_tokens: int, remaining_tokens: int, action: str) -> str:
    """Return a user-facing capacity message without exposing token jargon."""
    required_pages = estimate_pages_from_tokens(required_tokens)
    remaining_pages = estimate_pages_from_tokens(remaining_tokens)
    return (
        f"Capacité insuffisante pour {action}. "
        f"Cette opération est estimée à environ {required_pages} page"
        f"{'s' if required_pages > 1 else ''}. "
        f"Il vous reste environ {remaining_pages} page"
        f"{'s' if remaining_pages > 1 else ''} dans votre forfait."
    )


def get_session_reference_documents(session) -> List[SessionDocument]:
    """Return confirmed reference documents for subject/barème."""
    return document_preparation_service.get_reference_documents(session)


def confirm_session_documents(session) -> tuple[List[str], List[str], List[str], List[str]]:
    """Apply user decisions and validate the resulting document configuration."""
    prepared = document_preparation_service.apply_document_decisions(session)
    issues = [question.message for question in prepared.questions_for_user]
    return (
        prepared.copy_document_ids,
        prepared.reference_document_ids,
        prepared.excluded_document_ids,
        issues,
    )


def delete_session_artifacts(store: SessionStore, session_id: str) -> bool:
    """Delete persistent and temporary artifacts for a session."""
    deleted = store.delete()
    temp_dir = get_session_temp_dir(session_id)
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    return deleted


# ============================================================================
# Security Headers Middleware
# ============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # HSTS only for HTTPS (skip in development)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with correlation ID, method, path, status, latency."""

    async def dispatch(self, request: Request, call_next):
        # Extract correlation ID and user ID
        correlation_id = request.headers.get("X-Request-ID", "unknown")
        user_id = getattr(request.state, "user_id", None)

        # Start timer
        start_time = time.time()

        # Log request start
        with logger.contextualize(correlation_id=correlation_id, user_id=user_id):
            logger.info(
                f"{request.method} {request.url.path}",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "user_id": user_id
                }
            )

            # Process request
            response = await call_next(request)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Record metrics
            if hasattr(request.app.state, 'metrics_collector'):
                request.app.state.metrics_collector.record_request(
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    latency_ms=latency_ms
                )

            # Log response
            logger.info(
                f"{request.method} {request.url.path} -> {response.status_code}",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": round(latency_ms, 2),
                    "user_id": user_id
                }
            )

            return response


# ============================================================================
# Application Factory
# ============================================================================

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI application instance
    """
    settings = get_settings()

    app = FastAPI(
        title="AI Correction System",
        description="API de correction automatique de copies par IA",
        version="1.0.0"
    )

    # Configure rate limiting (use module-level limiter)
    app.state.limiter = limiter

    # Rate limit exception handler
    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Trop de requêtes. Réessayez plus tard."},
            headers={"Retry-After": str(exc.retry_after)}
        )

    # Global exception handler for Sentry
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return await sentry_exception_handler(request, exc)

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    )

    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)

    # Add request logging middleware (logs all requests with correlation ID)
    app.add_middleware(RequestLoggingMiddleware)

    # Add correlation ID middleware (generates/reads X-Request-ID header)
    app.add_middleware(CorrelationIdMiddleware, validator=lambda x: True)

    # Initialize database on startup
    @app.on_event("startup")
    async def startup_event():
        """Fail fast if critical security settings are invalid."""
        try:
            settings = get_settings()
            # This will raise ValidationError if JWT_SECRET or API keys are invalid
            stdlib_logger.info(f"Security configuration validated. Provider: {settings.ai_provider}")
        except ValidationError as e:
            stdlib_logger.error(f"Configuration error: {e}")
            raise SystemExit(1)

        # Initialize structured logging (Loguru JSON)
        setup_structured_logging(level="INFO")
        logger.info("Structured logging initialized with correlation ID support")

        # Initialize Sentry error tracking
        init_sentry(
            dsn=settings.sentry_dsn,
            environment=settings.sentry_environment,
            sample_rate=settings.sentry_traces_sample_rate,
            debug=(settings.sentry_environment == "development")
        )

        # Initialize metrics collector
        metrics_collector = get_metrics_collector()
        app.state.metrics_collector = metrics_collector
        logger.info("Metrics collector initialized")

        from db import init_db
        init_db()
        stdlib_logger.info("Database initialized")

    # Include auth router
    app.include_router(auth_router, prefix="/api")

    # Include health check router
    app.include_router(health_router, tags=["health"])

    if subscription_router is not None:
        app.include_router(subscription_router, prefix="/api")

    # Session storage (in-memory for active grading)
    active_sessions: Dict[str, GradingSessionOrchestrator] = {}
    session_progress: Dict[str, Dict[str, Any]] = {}

    # Max sessions kept in memory before cleanup of completed ones
    MAX_ACTIVE_SESSIONS = 100

    def _cleanup_completed_sessions():
        """Remove completed/error sessions from memory to prevent leaks."""
        if len(active_sessions) <= MAX_ACTIVE_SESSIONS:
            return
        to_remove = [
            sid for sid, orch in active_sessions.items()
            if orch.session.status in (SessionStatus.COMPLETE, SessionStatus.ERROR)
        ]
        for sid in to_remove:
            active_sessions.pop(sid, None)
            session_progress.pop(sid, None)

    # ============================================================================
    # WebSocket Endpoint
    # ============================================================================

    @app.websocket("/api/sessions/{session_id}/ws")
    async def websocket_progress(websocket: WebSocket, session_id: str):
        """
        WebSocket for real-time grading progress with reconnection support.

        Sends events:
        - copy_start: When a copy starts grading (copy_index, total_copies, student_name, stage)
        - question_done: When a question is graded (copy_index, question_id, grade, max_points, agreement)
        - copy_done: When a copy is fully graded (copy_index, student_name, total_score, max_score, confidence)
        - copy_error: When grading fails for a copy (copy_index, error)
        - session_complete: When the session is complete (average_score, total_copies)
        - session_error: When session-level error occurs (error)
        - progress_sync: Current progress state (status, copies_uploaded, copies_graded, grading_mode)
        """
        token = extract_websocket_token(websocket)
        user_id = verify_websocket_session_access(session_id, token)
        if not user_id:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        await ws_manager.connect(websocket, session_id)
        try:
            # Send current progress state on connect (for reconnection)
            if session_id in session_progress:
                progress = session_progress[session_id]
                if progress.get("user_id") != user_id:
                    await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                    return
                await websocket.send_json({
                    "type": "progress_sync",
                    "data": serialize_progress_for_client(progress)
                })

            # Keep connection alive and handle client messages
            while True:
                data = await websocket.receive_text()
                # Handle ping/pong for connection health
                if data == "ping":
                    await websocket.send_text("pong")
                # Handle client requests for current state
                elif data == "sync":
                    if session_id in session_progress:
                        progress = session_progress[session_id]
                        if progress.get("user_id") != user_id:
                            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                            return
                        await websocket.send_json({
                            "type": "progress_sync",
                            "data": serialize_progress_for_client(progress)
                        })
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket, session_id)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            ws_manager.disconnect(websocket, session_id)

    # ============================================================================
    # Routes
    # ============================================================================

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "AI Correction System",
            "version": "1.0.0",
            "docs": "/docs"
        }

    @app.post("/api/sessions", response_model=SessionResponse)
    async def create_session(
        request: CreateSessionRequest,
        background_tasks: BackgroundTasks,
        current_user = Depends(get_current_user)
    ):
        """
        Create a new grading session.

        Upload PDFs via /api/sessions/{id}/upload
        """
        user_id = current_user.id

        # Create orchestrator WITHOUT session_id - it will generate a new one
        orchestrator = GradingSessionOrchestrator(
            user_id=user_id,
            workflow_state=API_WORKFLOW_STATE
        )

        # Set policy from request
        if request.subject:
            orchestrator.session.policy.subject = request.subject
        if request.topic:
            orchestrator.session.policy.topic = request.topic
        if request.question_weights:
            orchestrator.session.policy.question_weights = request.question_weights

        # Save initial session
        orchestrator._save_sync()

        session_id = orchestrator.session_id
        _cleanup_completed_sessions()
        active_sessions[session_id] = orchestrator
        session_progress[session_id] = {
            "status": "created",
            "copies_uploaded": 0,
            "copies_graded": 0,
            "disagreements": [],
            "user_id": user_id
        }

        # Record active session
        if hasattr(app.state, 'metrics_collector'):
            app.state.metrics_collector.record_active_session(session_id)

        return SessionResponse(
            session_id=session_id,
            status=orchestrator.session.status,
            created_at=str(orchestrator.session.created_at),
            copies_count=0,
            graded_count=0,
            subject=request.subject,
            topic=request.topic
        )

    @app.post("/api/sessions/{session_id}/upload")
    async def upload_copies(
        session_id: str,
        files: List[UploadFile] = File(...),
        current_user = Depends(get_current_user)
    ):
        """
        Upload PDF copies to a session.

        Accepts up to 50 PDF files per batch.
        """
        from db import SessionLocal, User

        user_id = current_user.id

        # Validate batch size
        if len(files) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Trop de fichiers. Maximum {MAX_BATCH_SIZE} PDFs par lot."
            )

        store = SessionStore(session_id, user_id=user_id)
        session = store.load_session()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        active_sessions.pop(session_id, None)

        # Save uploaded files
        upload_dir = get_session_temp_dir(session_id)
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Resolve to absolute path for validation
        upload_dir_resolved = upload_dir.resolve()

        pdf_paths = []
        session_documents: List[SessionDocument] = []
        validation_results = []

        for file in files:
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                validation_results.append({
                    "filename": file.filename or "unknown",
                    "success": False,
                    "error": "Not a PDF file"
                })
                continue

            # Sanitize filename: remove path separators and dangerous characters
            safe_filename = re.sub(r'[^\w\-.]', '_', Path(file.filename).stem) + ".pdf"

            # Use UUID to ensure uniqueness and prevent collisions
            unique_filename = f"{uuid.uuid4().hex[:8]}_{safe_filename}"
            file_path = upload_dir / unique_filename

            # Validate the resolved path is within upload_dir (prevent path traversal)
            try:
                resolved_path = file_path.resolve()
                if not str(resolved_path).startswith(str(upload_dir_resolved)):
                    validation_results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": "Invalid file path"
                    })
                    continue
            except (OSError, ValueError):
                validation_results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Invalid file path"
                })
                continue

            # Check file size
            file.file.seek(0, 2)  # Seek to end
            file_size = file.file.tell()
            file.file.seek(0)  # Reset to beginning

            if file_size > MAX_UPLOAD_SIZE:
                validation_results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)} MB"
                })
                continue

            try:
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)
                pdf_paths.append(str(file_path))
                (
                    detected_type,
                    confidence,
                    issues,
                    page_count,
                    text_excerpt,
                    evidence,
                    page_classifications,
                    segments,
                ) = detect_document_type_from_pdf(
                    str(file_path),
                    file.filename,
                )
                session_documents.append(SessionDocument(
                    filename=file.filename,
                    storage_path=str(file_path),
                    page_count=page_count,
                    detected_type=detected_type,
                    confidence=confidence,
                    issues=issues,
                    evidence=evidence,
                    text_excerpt=text_excerpt,
                    page_classifications=page_classifications,
                    segments=segments,
                    status=DocumentStatus.CLASSIFIED,
                    user_decision=DocumentDecision.PENDING,
                    usable=False,
                ))
                validation_results.append({
                    "filename": file.filename,
                    "success": True,
                    "path": str(file_path),
                    "detected_type": detected_type.value,
                    "confidence": confidence,
                    "issues": issues,
                    "evidence": evidence,
                    "text_excerpt": text_excerpt,
                    "page_classifications": [item.model_dump(mode="json") for item in page_classifications],
                    "segments": [item.model_dump(mode="json") for item in segments],
                })
            except Exception as e:
                validation_results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })

        if not pdf_paths:
            raise HTTPException(
                status_code=400,
                detail="Aucun PDF valide n'a été envoyé."
            )

        store.clear_processing_artifacts()
        session.status = SessionStatus.DIAGNOSTIC
        session.documents = session_documents
        session.prepared_correction = None
        session.copies = []
        session.graded_copies = []
        session.class_map = None
        session.copies_processed = 0
        store.save_session(session)

        # Update session progress
        session_progress[session_id] = {
            "status": "uploaded",
            "copies_uploaded": len(pdf_paths),
            "copies_graded": 0,
            "disagreements": [],
            "user_id": user_id,
        }

        return {
            "session_id": session_id,
            "uploaded_count": len(pdf_paths),
            "paths": pdf_paths,
            "validation_results": validation_results,
            "documents": [serialize_session_document(doc).model_dump() for doc in session_documents],
        }

    @app.get("/api/sessions/{session_id}/documents", response_model=List[SessionDocumentResponse])
    async def list_session_documents(session_id: str, current_user = Depends(get_current_user)):
        """List uploaded documents and their current classification state."""
        store = SessionStore(session_id, user_id=current_user.id)
        session = store.load_session()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return [serialize_session_document(doc) for doc in session.documents]

    @app.patch("/api/sessions/{session_id}/documents/{document_id}", response_model=SessionDocumentResponse)
    async def update_session_document(
        session_id: str,
        document_id: str,
        request: UpdateSessionDocumentRequest,
        current_user = Depends(get_current_user)
    ):
        """Update the explicit role of a session document."""
        store = SessionStore(session_id, user_id=current_user.id)
        session = store.load_session()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        document = next((doc for doc in session.documents if doc.id == document_id), None)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        document.user_decision = DocumentDecision(request.user_decision)
        document.usable = document.user_decision != DocumentDecision.EXCLUDE
        document.status = (
            DocumentStatus.REJECTED
            if document.user_decision == DocumentDecision.EXCLUDE
            else DocumentStatus.CLASSIFIED
        )
        session.prepared_correction = None
        store.save_session(session)
        return serialize_session_document(document)

    @app.post("/api/sessions/{session_id}/documents/confirm", response_model=ConfirmSessionDocumentsResponse)
    async def confirm_session_documents_endpoint(
        session_id: str,
        current_user = Depends(get_current_user)
    ):
        """Confirm uploaded documents and validate the session configuration."""
        store = SessionStore(session_id, user_id=current_user.id)
        session = store.load_session()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        copy_ids, reference_ids, excluded_ids, issues = confirm_session_documents(session)
        store.save_session(session)

        return ConfirmSessionDocumentsResponse(
            success=not issues,
            copies_document_ids=copy_ids,
            reference_document_ids=reference_ids,
            excluded_document_ids=excluded_ids,
            issues=issues,
            prepared_correction=(
                session.prepared_correction.model_dump(mode="json")
                if session.prepared_correction
                else None
            ),
        )

    @app.get("/api/sessions/{session_id}/documents/{document_id}/pdf")
    async def get_session_document_pdf(
        session_id: str,
        document_id: str,
        current_user = Depends(get_current_user)
    ):
        """Stream an uploaded session document PDF for manual verification."""
        store = SessionStore(session_id, user_id=current_user.id)
        session = store.load_session()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        document = next((doc for doc in session.documents if doc.id == document_id), None)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        pdf_path = Path(document.storage_path)
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="PDF not found")

        def iterfile():
            with open(pdf_path, "rb") as f:
                yield from f

        return StreamingResponse(
            iterfile(),
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{pdf_path.name}"'},
        )

    @app.post("/api/sessions/{session_id}/detect", response_model=DetectionResponse)
    async def detect_session(
        session_id: str,
        request: DetectionRequest,
        background_tasks: BackgroundTasks,
        current_user = Depends(get_current_user)
    ):
        """
        Detect PDF structure and grading scale.

        This detection is performed before grading to:
        - Validate the PDF contains student copies
        - Detect document structure (one student or multiple per PDF)
        - Detect grading scale / barème (with multiple candidate scales if uncertain)
        - Identify blocking issues
        """
        from analysis.detection import Detector

        user_id = current_user.id

        # Check session exists
        store = SessionStore(session_id, user_id=user_id)
        session = store.load_session()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if session.documents and session.prepared_correction and not session.prepared_correction.ready_to_grade:
            blocking_messages = [question.message for question in session.prepared_correction.questions_for_user]
            raise HTTPException(status_code=400, detail=" ".join(blocking_messages))

        copy_documents = get_session_copy_documents(session)
        if session.documents and not copy_documents:
            raise HTTPException(
                status_code=400,
                detail="Aucun document n'est actuellement utilisable comme copie élève. Confirmez les documents d'abord."
            )

        if copy_documents:
            pdf_path = copy_documents[0].storage_path
        else:
            pdf_files = list_uploaded_pdfs(session_id)
            if not pdf_files:
                raise HTTPException(status_code=400, detail="No PDF file found")
            pdf_path = str(pdf_files[0])

        # Check tier-based limits for re-detection (force_refresh)
        if request.force_refresh:
            from db import SubscriptionTier
            tier = current_user.subscription_tier

            # FREE tier cannot re-detect (consumes ~5-7K tokens)
            if tier == SubscriptionTier.FREE:
                raise HTTPException(
                    status_code=403,
                    detail="Re-détection non disponible sur le plan FREE. Passez à ESSENTIEL."
                )

            # ESSENTIEL can only re-detect once per session
            if tier == SubscriptionTier.ESSENTIEL:
                existing = store.load_detection()
                if existing:
                    raise HTTPException(
                        status_code=403,
                        detail="Re-détection limitée à 1x par session sur ESSENTIEL. Passez à PRO pour illimité."
                    )

        # Update session progress
        if session_id in session_progress:
            session_progress[session_id]["status"] = "detection"

        # Update session file status
        if session:
            session.status = SessionStatus.DIAGNOSTIC
            store.save_session(session)

        try:
            # Run detection
            detector = Detector(
                user_id=user_id,
                session_id=session_id,
                language="fr"  # TODO: Get from user preferences
            )

            result = detector.detect(pdf_path, mode=request.mode, force_refresh=request.force_refresh)

            if len(copy_documents) > 1:
                result.warnings.append(
                    "Plusieurs documents de copies ont été fournis. La détection initiale utilise le premier document confirmé; la correction utilisera tout le lot confirmé."
                )

            # Debug log for exam_name
            logger.info(f"Detection result exam_name: {result.exam_name}")

            # Deduct actual tokens used by detection, independently from grading.
            detection_usage_key = f"{session_id}:detection:{result.detection_id}"
            db = SessionLocal()
            try:
                deduction_svc = TokenDeductionService()
                deduction_svc.deduct_grading_usage(
                    user_id=user_id,
                    provider=detector.provider,
                    session_id=detection_usage_key,
                    db=db,
                )
            except InsufficientTokensError as e:
                raise HTTPException(
                    status_code=402,
                    detail=format_capacity_error(
                        e.tokens_required,
                        e.tokens_remaining,
                        "enregistrer cette détection",
                    ),
                )
            except (UserNotFoundError, DeductionError) as e:
                logger.error(f"Detection token deduction failed for session {session_id}: {e}")
                raise HTTPException(status_code=500, detail="Failed to record detection token usage.")
            finally:
                db.close()

            # Store result in session for later use
            store.save_detection(result)

            # Convert students to schema
            students = [
                StudentInfoSchema(
                    index=s.index,
                    name=s.name,
                    start_page=s.start_page,
                    end_page=s.end_page,
                    confidence=s.confidence
                )
                for s in result.students
            ]

            # Convert candidate_scales to schema
            candidate_scales = []
            for candidate in result.candidate_scales:
                candidate_scales.append(CandidateScale(
                    scale=candidate.get("scale", {}),
                    confidence=candidate.get("confidence", 0.0)
                ))

            # Update progress
            if session_id in session_progress:
                session_progress[session_id]["status"] = "detection"

            # Update session file status
            session = store.load_session()
            if session:
                session.status = SessionStatus.DIAGNOSTIC
                store.save_session(session)

            return DetectionResponse(
                detection_id=result.detection_id,
                mode=result.mode,
                is_valid_pdf=result.is_valid_pdf,
                page_count=result.page_count,
                document_type=result.document_type.value,
                confidence_document_type=result.confidence_document_type,
                structure=result.structure.value,
                subject_integration=result.subject_integration.value,
                num_students_detected=result.num_students_detected,
                students=students,
                grading_scale=result.grading_scale,
                confidence_grading_scale=result.confidence_grading_scale,
                candidate_scales=candidate_scales,
                questions_detected=result.questions_detected,
                blocking_issues=result.blocking_issues,
                has_blocking_issues=result.has_blocking_issues,
                warnings=result.warnings,
                quality_issues=result.quality_issues,
                overall_quality_score=result.overall_quality_score,
                detected_language=result.detected_language,
                detection_duration_ms=result.detection_duration_ms,
                exam_name=result.exam_name
            )

        except Exception as e:
            logger.error(f"Detection error: {e}")
            if session_id in session_progress:
                session_progress[session_id]["status"] = "error"
                session_progress[session_id]["error"] = str(e)
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    @app.post("/api/sessions/{session_id}/confirm-detection", response_model=ConfirmDetectionResponse)
    async def confirm_detection(
        session_id: str,
        request: ConfirmDetectionRequest,
        background_tasks: BackgroundTasks,
        current_user = Depends(get_current_user)
    ):
        """
        Confirm detection and automatically start grading.

        Allows:
        - Selecting from multiple candidate grading scales
        - Adjusting the detected grading scale
        - Overriding detected student names

        After confirmation, grading starts automatically in background.
        """
        user_id = current_user.id

        # Check session exists
        logger.info(f"Confirm detection: session_id={session_id}, user_id={user_id}")
        store = SessionStore(session_id, user_id=user_id)
        logger.info(f"Session dir: {store.session_dir}, exists: {store.exists()}")
        if not store.exists():
            logger.warning(f"Session not found: {session_id} for user {user_id}")
            raise HTTPException(status_code=404, detail="Session not found")

        # Load detection result
        detection = store.load_detection()
        if not detection:
            raise HTTPException(status_code=400, detail="No detection found. Run /detect first.")

        # Check for blocking issues
        if detection.has_blocking_issues:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot confirm: blocking issues detected: {detection.blocking_issues}"
            )

        # Start with detected grading scale
        grading_scale = dict(detection.grading_scale)

        # Apply selected_scale_index if provided
        if request.selected_scale_index is not None:
            candidate_scales = detection.candidate_scales
            if not candidate_scales:
                raise HTTPException(
                    status_code=400,
                    detail="No candidate scales available. Cannot select by index."
                )
            if request.selected_scale_index < 0 or request.selected_scale_index >= len(candidate_scales):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid scale index {request.selected_scale_index}. Available: 0-{len(candidate_scales)-1}"
                )
            # Use the selected candidate scale
            selected = candidate_scales[request.selected_scale_index]
            grading_scale = dict(selected.get("scale", {}))

        # Apply adjustments if provided
        if request.adjustments:
            # Apply grading scale overrides
            if "grading_scale" in request.adjustments:
                grading_scale.update(request.adjustments["grading_scale"])

            # Apply student name overrides
            if "student_names" in request.adjustments:
                # Update student names in detection result
                student_names = request.adjustments["student_names"]
                for student_idx, name in student_names.items():
                    # Find the student by index and update name
                    for student in detection.students:
                        if student.index == student_idx:
                            student.name = name
                            break

        # Persist the adjusted detection result even if grading cannot start yet.
        detection.grading_scale = grading_scale
        detection.questions_detected = list(grading_scale.keys())
        store.save_detection(detection)

        session = store.load_session()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        copy_document_ids, reference_document_ids, excluded_document_ids, document_issues = confirm_session_documents(session)
        if document_issues:
            store.save_session(session)
            raise HTTPException(status_code=400, detail=" ".join(document_issues))

        # Persist confirmed grading settings before checking quota so the teacher never loses edits.
        session.policy.question_weights = grading_scale

        # Propagate exam_name to session.policy.subject if not already set
        if detection.exam_name and not session.policy.subject:
            session.policy.subject = detection.exam_name

        if not session.storage_path:
            session.storage_path = str(store.session_dir)

        store.save_session(session)

        user_id = current_user.id
        copy_documents = get_session_copy_documents(session) if session else []
        if not copy_documents:
            raise HTTPException(status_code=400, detail="Aucun document de copies confirmé pour cette session.")

        estimated_tokens = estimate_session_token_budget(session)
        if not current_user.can_use_tokens(estimated_tokens):
            raise HTTPException(
                status_code=402,
                detail=format_capacity_error(
                    estimated_tokens,
                    current_user.remaining_tokens,
                    "lancer cette correction",
                ),
            )

        # Update session with confirmed settings only once grading can actually start.
        session.status = SessionStatus.CORRECTION
        store.save_session(session)

        if session_id in session_progress:
            session_progress[session_id]["status"] = "correction"

        # Auto-start grading after confirmation
        if session_id not in active_sessions:
            orchestrator = GradingSessionOrchestrator(
                session_id=session_id,
                user_id=user_id,
                workflow_state=API_WORKFLOW_STATE,
                grading_mode=request.grading_mode,
                batch_verify=request.batch_verify
            )
            active_sessions[session_id] = orchestrator
        else:
            orchestrator = active_sessions[session_id]
            orchestrator._grading_mode = request.grading_mode
            orchestrator._batch_verify = request.batch_verify

        orchestrator.pdf_paths = [doc.storage_path for doc in copy_documents]

        # Create progress callback for WebSocket
        progress_callback = ws_manager.create_progress_callback(session_id)

        # Start grading in background
        async def auto_grade_task():
            from db import SessionLocal, User
            try:
                logger.info(f"Auto-grading task started for session {session_id}")

                # Update progress
                if session_id in session_progress:
                    session_progress[session_id]["status"] = "correction"

                # Re-create store for this task context
                task_store = SessionStore(session_id, user_id=user_id)

                logger.info(f"PDF paths for grading: {orchestrator.pdf_paths}")

                # Run analysis phase
                logger.info(f"Starting analyze_only for session {session_id}")
                await orchestrator.analyze_only()
                logger.info(f"analyze_only completed for session {session_id}")

                # Confirm scale with the validated grading scale
                orchestrator.confirm_scale(grading_scale)
                logger.info(f"Scale confirmed for session {session_id}")

                # Grade with progress callback
                logger.info(f"Starting grade_all for session {session_id}")
                await orchestrator.grade_all(progress_callback=progress_callback)
                logger.info(f"grade_all completed for session {session_id}")

                # Reload session to get graded copies
                session = task_store.load_session()

                # Record grading operation
                if hasattr(app.state, 'metrics_collector') and session and session.graded_copies:
                    token_usage = len(session.graded_copies) * 10000
                    app.state.metrics_collector.record_grading_operation(
                        session_id=session_id,
                        tokens_used=token_usage
                    )

                # Notify completion
                if session and session.graded_copies:
                    scores = [g.total_score for g in session.graded_copies]
                    avg = sum(scores) / len(scores)
                else:
                    avg = 0

                # Deduct actual tokens used
                result = {
                    'tokens_deducted': 0,
                    'remaining_tokens': 0,
                    'usage_record_id': None,
                    'is_duplicate': False
                }
                db_user = None

                try:
                    db = SessionLocal()
                    deduction_svc = TokenDeductionService()
                    result = deduction_svc.deduct_grading_usage(
                        user_id=user_id,
                        provider=orchestrator.ai,
                        session_id=session_id,
                        db=db
                    )
                    logger.info(f"Deducted {result['tokens_deducted']} tokens for session {session_id}")
                    db_user = db.query(User).filter(User.id == user_id).first()
                    db.close()
                except InsufficientTokensError as e:
                    await ws_manager.broadcast_event(session_id, PROGRESS_EVENT_SESSION_ERROR, {
                        "error": "Insufficient tokens",
                        "tokens_required": e.tokens_required,
                        "tokens_remaining": e.tokens_remaining
                    })
                    logger.error(f"Insufficient tokens for user {user_id}")
                except Exception as e:
                    logger.error(f"Token deduction error for session {session_id}: {e}")
                    # Continue without blocking - grading already done

                await ws_manager.broadcast_event(session_id, PROGRESS_EVENT_SESSION_COMPLETE, {
                    "average_score": avg,
                    "total_copies": len(session.graded_copies) if session else 0,
                    "tokens_used": result['tokens_deducted'],
                    "remaining_tokens": db_user.remaining_tokens if db_user else 0
                })

                # Update session status in database
                if session:
                    session.status = SessionStatus.COMPLETE
                    task_store.save_session(session)

                # Update progress
                if session_id in session_progress:
                    session_progress[session_id]["status"] = "complete"
                    session_progress[session_id]["copies_graded"] = len(session.graded_copies) if session else 0

            except Exception as e:
                logger.error(f"Auto-grading failed for session {session_id}: {e}")
                await ws_manager.broadcast_event(session_id, PROGRESS_EVENT_SESSION_ERROR, {
                    "error": str(e)
                })
                if session_id in session_progress:
                    session_progress[session_id]["status"] = "error"
                # Reset session status so it doesn't stay stuck on "correction"
                try:
                    error_store = SessionStore(session_id, user_id=user_id)
                    error_session = error_store.load_session()
                    if error_session and error_session.status == SessionStatus.CORRECTION:
                        error_session.status = SessionStatus.DIAGNOSTIC
                        error_store.save_session(error_session)
                except Exception as save_error:
                    logger.warning(f"Could not update session status: {save_error}")
            finally:
                # Always attempt to finalize session status if grading completed
                # This handles cases where the task was interrupted after grading
                # but before the final status update
                try:
                    task_store = SessionStore(session_id, user_id=user_id)
                    session = task_store.load_session()
                    if session:
                        # If we have graded copies, mark as complete
                        if session.graded_copies and len(session.graded_copies) > 0:
                            if session.status == SessionStatus.CORRECTION:
                                logger.info(f"Finalizing interrupted session {session_id}")
                                session.status = SessionStatus.COMPLETE
                                task_store.save_session(session)
                        # If still grading but no copies graded, reset to diagnostic
                        elif session.status == SessionStatus.CORRECTION:
                            logger.info(f"Resetting stuck session {session_id} to diagnostic")
                            session.status = SessionStatus.DIAGNOSTIC
                            task_store.save_session(session)
                except Exception as finalize_error:
                    logger.warning(f"Could not finalize session {session_id}: {finalize_error}")

        # Add background task
        background_tasks.add_task(auto_grade_task)

        return ConfirmDetectionResponse(
            success=True,
            session_id=session_id,
            status="correction",  # Status changed - grading started automatically
            grading_scale=grading_scale,
            num_students=detection.num_students_detected
        )

    @app.get("/api/sessions/{session_id}", response_model=SessionDetailResponse)
    async def get_session(session_id: str, current_user = Depends(get_current_user)):
        """
        Get detailed session information with complete dual-LLM data for review.

        Returns graded copies with:
        - grading_audit: Full dual-LLM comparison data
        - confidence_by_question: Per-question confidence scores
        - student_feedback: Per-question feedback for students
        - has_disagreements: Flag indicating if any LLM disagreements occurred
        """
        user_id = current_user.id
        logger.info(f"Get session: session_id={session_id}, user_id={user_id}")
        store = SessionStore(session_id, user_id=user_id)
        session = store.load_session()

        if not session:
            logger.warning(f"Session not found in get_session: {session_id} for user {user_id}")
            raise HTTPException(status_code=404, detail="Session not found")

        # Detect and fix stale "correction" status (orphaned after server restart/crash)
        if session.status == SessionStatus.CORRECTION:
            # If session is in correction but not in active_sessions, it's stale
            if session_id not in active_sessions:
                if session.graded_copies and len(session.graded_copies) > 0:
                    # Has graded copies -> mark as complete
                    logger.info(f"Recovering orphaned correction session {session_id} -> complete")
                    session.status = SessionStatus.COMPLETE
                    store.save_session(session)
                else:
                    # No graded copies -> reset to diagnostic
                    logger.info(f"Recovering orphaned correction session {session_id} -> diagnostic")
                    session.status = SessionStatus.DIAGNOSTIC
                    store.save_session(session)

        average = None
        max_score = None
        if session.graded_copies:
            scores = [g.total_score for g in session.graded_copies]
            average = sum(scores) / len(scores)
        # Get max_score from grading_scale (source of truth)
        if session.policy.question_weights:
            max_score = sum(session.policy.question_weights.values())

        # Build copies list
        copies = []
        for copy in session.copies:
            copy_info = {
                "id": copy.id,
                "student_name": copy.student_name,
                "page_count": copy.page_count,
                "processed": copy.processed
            }
            copies.append(copy_info)

        # Build graded copies list with full dual-LLM data
        graded_copies = []
        # Create a mapping from copy_id to student_name
        copy_student_names = {copy.id: copy.student_name for copy in session.copies}

        # Get subject from session.policy, or from detection exam_name
        subject = session.policy.subject
        if not subject:
            # Try to get exam_name from detection
            detection = store.load_detection()
            if detection and detection.exam_name:
                subject = detection.exam_name

        for graded in session.graded_copies:
            # Check for disagreements in grading_audit
            has_disagreements = False
            if graded.grading_audit:
                for qaudit in graded.grading_audit.questions.values():
                    if qaudit.resolution.agreement is False:
                        has_disagreements = True
                        break

            graded_info = {
                "copy_id": graded.copy_id,
                "student_name": copy_student_names.get(graded.copy_id),
                "total_score": graded.total_score,
                "max_score": graded.max_score,
                "confidence": graded.confidence,
                "grades": graded.grades,
                "max_points_by_question": graded.max_points_by_question,
                "confidence_by_question": graded.confidence_by_question,
                "student_feedback": graded.student_feedback,
                "grading_audit": graded.grading_audit.model_dump() if graded.grading_audit else None,
                "has_disagreements": has_disagreements
            }
            graded_copies.append(graded_info)

        return SessionDetailResponse(
            session_id=session_id,
            status=session.status,
            created_at=str(session.created_at),
            copies_count=len(session.copies),
            graded_count=len(session.graded_copies),
            average_score=average,
            max_score=max_score,
            subject=subject,
            topic=session.policy.topic,
            prepared_correction=(
                session.prepared_correction.model_dump(mode="json")
                if session.prepared_correction
                else None
            ),
            documents=[serialize_session_document(doc).model_dump() for doc in session.documents],
            copies=copies,
            graded_copies=graded_copies,
            question_weights=session.policy.question_weights,
            question_names=session.policy.question_names,
        )

    @app.patch("/api/sessions/{session_id}/copies/{copy_id}/grades", response_model=UpdateGradeResponse)
    async def update_grade(
        session_id: str,
        copy_id: str,
        request: UpdateGradeRequest,
        current_user = Depends(get_current_user)
    ):
        """
        Update a single question grade for a graded copy.

        Recalculates total_score automatically unless auto_recalc=False.
        Persists changes immediately to user-scoped session storage.
        """
        user_id = current_user.id
        store = SessionStore(session_id, user_id=user_id)
        session = store.load_session()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Find the graded copy
        graded = next((g for g in session.graded_copies if g.copy_id == copy_id), None)
        if not graded:
            raise HTTPException(status_code=404, detail="Graded copy not found")

        # Validate question_id exists
        if request.question_id not in graded.grades:
            raise HTTPException(status_code=400, detail="Question not found in graded copy")

        # Validate new_grade against max points for this question
        max_points = session.policy.question_weights.get(request.question_id)
        if max_points is not None and request.new_grade > max_points:
            raise HTTPException(
                status_code=400,
                detail=f"Grade {request.new_grade} exceeds max points {max_points} for {request.question_id}"
            )

        # Store old values
        old_grade = graded.grades[request.question_id]
        old_total = graded.total_score

        # Update grade
        graded.grades[request.question_id] = request.new_grade

        # Recalculate total
        if request.auto_recalc:
            graded.total_score = sum(graded.grades.values())

        # Persist immediately
        store.save_session(session)

        return UpdateGradeResponse(
            success=True,
            copy_id=copy_id,
            question_id=request.question_id,
            old_grade=old_grade,
            new_grade=request.new_grade,
            old_total=old_total,
            new_total=graded.total_score,
            max_score=graded.max_score
        )

    @app.patch("/api/sessions/{session_id}/copies/{copy_id}/student-name")
    async def update_student_name(
        session_id: str,
        copy_id: str,
        request: UpdateStudentNameRequest,
        current_user = Depends(get_current_user)
    ):
        """Update the student name for a copy."""
        user_id = current_user.id
        store = SessionStore(session_id, user_id=user_id)
        session = store.load_session()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Find the copy document
        copy = next((c for c in session.copies if c.id == copy_id), None)
        if not copy:
            raise HTTPException(status_code=404, detail="Copy not found")

        # Update student name
        old_name = copy.student_name
        copy.student_name = request.student_name

        # Persist immediately
        store.save_session(session)

        return {"success": True, "copy_id": copy_id, "old_name": old_name, "new_name": request.student_name}

    @app.patch("/api/sessions/{session_id}/subject")
    async def update_session_subject(
        session_id: str,
        request: UpdateSessionSubjectRequest,
        current_user = Depends(get_current_user)
    ):
        """Update the session subject (exam name)."""
        user_id = current_user.id
        store = SessionStore(session_id, user_id=user_id)
        session = store.load_session()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Update subject in session policy
        old_subject = session.policy.subject
        session.policy.subject = request.subject

        # Persist immediately
        store.save_session(session)

        return {"success": True, "session_id": session_id, "old_subject": old_subject, "new_subject": request.subject}

    @app.patch("/api/sessions/{session_id}/question-weights", response_model=UpdateQuestionWeightResponse)
    async def update_question_weight(
        session_id: str,
        request: UpdateQuestionWeightRequest,
        current_user = Depends(get_current_user)
    ):
        """
        Update the max points (weight) for a question.

        This will:
        1. Update the question weight in the session policy
        2. Recalculate max_score for all graded copies
        3. Cap any grades that exceed the new max points
        """
        user_id = current_user.id
        store = SessionStore(session_id, user_id=user_id)
        session = store.load_session()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Check if question exists in weights
        if request.question_id not in session.policy.question_weights:
            raise HTTPException(
                status_code=400,
                detail=f"Question {request.question_id} not found in grading scale"
            )

        # Store old values
        old_weight = session.policy.question_weights[request.question_id]
        old_max_score = sum(session.policy.question_weights.values())

        # Update weight
        session.policy.question_weights[request.question_id] = request.new_weight
        new_max_score = sum(session.policy.question_weights.values())

        # Update max_score and cap grades for all graded copies
        updated_copies = 0
        for graded in session.graded_copies:
            # Update max_points_by_question if it exists
            if graded.max_points_by_question and request.question_id in graded.max_points_by_question:
                graded.max_points_by_question[request.question_id] = request.new_weight

            # Cap grade if it exceeds new max
            if request.question_id in graded.grades:
                if graded.grades[request.question_id] > request.new_weight:
                    graded.grades[request.question_id] = request.new_weight
                    updated_copies += 1

            # Recalculate total score
            graded.total_score = sum(graded.grades.values())
            # Recalculate max_score from weights
            graded.max_score = sum(
                session.policy.question_weights.get(q, 0)
                for q in graded.grades.keys()
            )

        # Persist changes
        store.save_session(session)

        return UpdateQuestionWeightResponse(
            success=True,
            question_id=request.question_id,
            old_weight=old_weight,
            new_weight=request.new_weight,
            old_max_score=old_max_score,
            new_max_score=new_max_score,
            updated_copies=updated_copies
        )

    @app.patch("/api/sessions/{session_id}/question-names", response_model=UpdateQuestionNameResponse)
    async def update_question_name(
        session_id: str,
        request: UpdateQuestionNameRequest,
        current_user = Depends(get_current_user)
    ):
        """
        Update the display name for a question.

        This allows teachers to give custom names to questions (e.g., "Exercice 1 - Calcul").
        """
        user_id = current_user.id
        store = SessionStore(session_id, user_id=user_id)
        session = store.load_session()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Check if question exists in weights
        if request.question_id not in session.policy.question_weights:
            raise HTTPException(
                status_code=400,
                detail=f"Question {request.question_id} not found in grading scale"
            )

        # Store old name
        old_name = session.policy.question_names.get(request.question_id)

        # Update name
        session.policy.question_names[request.question_id] = request.new_name

        # Persist changes
        store.save_session(session)

        return UpdateQuestionNameResponse(
            success=True,
            question_id=request.question_id,
            old_name=old_name,
            new_name=request.new_name
        )

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str, current_user = Depends(get_current_user)):
        """Delete a session and all its data."""
        user_id = current_user.id
        store = SessionStore(session_id, user_id=user_id)

        if not store.exists():
            raise HTTPException(status_code=404, detail="Session not found")

        # Remove from active sessions if present
        if session_id in active_sessions:
            del active_sessions[session_id]
        if session_id in session_progress:
            del session_progress[session_id]

        if hasattr(app.state, 'metrics_collector'):
            app.state.metrics_collector.remove_active_session(session_id)

        # Delete persistent and temporary session artifacts
        success = delete_session_artifacts(store, session_id)

        return {"success": success}

    @app.post("/api/sessions/{session_id}/grade", response_model=GradeResponse)
    async def start_grading(
        session_id: str,
        request: StartGradingRequest,
        background_tasks: BackgroundTasks,
        current_user = Depends(get_current_user)
    ):
        """
        Start the grading process for a session.

        Progress updates are sent via WebSocket at /api/sessions/{session_id}/ws

        Args:
            session_id: The session to grade
            request: Grading configuration including mode (single or dual LLM)
            background_tasks: FastAPI background tasks
            current_user: Authenticated user
        """
        user_id = current_user.id
        store = SessionStore(session_id, user_id=user_id)
        session = store.load_session()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if session.documents:
            if session.prepared_correction and not session.prepared_correction.ready_to_grade:
                blocking_messages = [question.message for question in session.prepared_correction.questions_for_user]
                raise HTTPException(status_code=400, detail=" ".join(blocking_messages))
            copy_documents = get_session_copy_documents(session)
            if not copy_documents or any(doc.status != DocumentStatus.CONFIRMED for doc in copy_documents):
                raise HTTPException(
                    status_code=400,
                    detail="Les documents de la session doivent être qualifiés et confirmés avant de lancer la correction."
                )

        # Determine force_single_llm based on request
        force_single_llm = request.grading_mode == "single"

        # Get or create orchestrator with proper grading configuration
        if session_id not in active_sessions:
            orchestrator = GradingSessionOrchestrator(
                session_id=session_id,
                user_id=user_id,
                force_single_llm=force_single_llm,
                grading_mode=request.grading_method,
                batch_verify=request.batch_verify,
                workflow_state=API_WORKFLOW_STATE
            )
            active_sessions[session_id] = orchestrator
        else:
            orchestrator = active_sessions[session_id]
            # Update force_single_llm on existing orchestrator
            orchestrator._force_single_llm = force_single_llm
            orchestrator._grading_mode = request.grading_method
            orchestrator._batch_verify = request.batch_verify

        # Store selected grading mode in session_progress for UI reference
        if session_id in session_progress:
            session_progress[session_id]["grading_mode"] = request.grading_mode

        # Get uploaded PDF paths
        copy_documents = get_session_copy_documents(session)
        if copy_documents:
            orchestrator.pdf_paths = [doc.storage_path for doc in copy_documents]
        else:
            pdf_files = list_uploaded_pdfs(session_id)
            if pdf_files:
                orchestrator.pdf_paths = [str(p) for p in pdf_files]

        estimated_tokens = estimate_session_token_budget(session)
        if not current_user.can_use_tokens(estimated_tokens):
            raise HTTPException(
                status_code=402,
                detail=format_capacity_error(
                    estimated_tokens,
                    current_user.remaining_tokens,
                    "lancer cette correction",
                ),
            )

        # Create progress callback for WebSocket
        progress_callback = ws_manager.create_progress_callback(session_id)

        # Start grading in background
        async def grade_task():
            from db import SessionLocal, User

            try:
                # Run analysis phase
                await orchestrator.analyze_only()

                # Confirm scale (use detected or default)
                orchestrator.confirm_scale(orchestrator.question_scales)

                # Grade with progress callback
                # The orchestrator will send events through the callback:
                # - PROGRESS_EVENT_COPY_START: When starting each copy
                # - PROGRESS_EVENT_QUESTION_DONE: After each question
                # - PROGRESS_EVENT_COPY_DONE: When copy is complete
                # - PROGRESS_EVENT_COPY_ERROR: On error for a copy
                await orchestrator.grade_all(progress_callback=progress_callback)

                # Record grading operation with token usage
                if hasattr(app.state, 'metrics_collector'):
                    # Get token usage from session (simplified - tracks copy count)
                    token_usage = len(session.graded_copies) * 10000  # Estimate
                    app.state.metrics_collector.record_grading_operation(
                        session_id=session_id,
                        tokens_used=token_usage
                    )

                # Notify completion with proper event constant
                if session.graded_copies:
                    scores = [g.total_score for g in session.graded_copies]
                    avg = sum(scores) / len(scores)
                else:
                    avg = 0

                # Broadcast completion AFTER token deduction (moved below deduction block)

                # Deduct actual tokens used (not copy count)
                # Initialize with safe defaults for completion broadcast
                result = {
                    'tokens_deducted': 0,
                    'remaining_tokens': 0,
                    'usage_record_id': None,
                    'is_duplicate': False
                }
                db_user = None

                logger.info(f"[TOKEN_DEBUG] Starting token deduction for session {session_id}, user {user_id}")

                # Also write to file for debugging
                with open("/tmp/token_debug.log", "a") as f:
                    f.write(f"\n=== {session_id} ===\n")
                    f.write(f"User: {user_id}\n")

                db = SessionLocal()
                try:
                    # Get token usage from provider for debugging
                    provider_usage = orchestrator.ai.get_token_usage() if hasattr(orchestrator.ai, 'get_token_usage') else {}
                    logger.info(f"[TOKEN_DEBUG] Provider usage: {provider_usage}")

                    deduction_svc = TokenDeductionService()
                    result = deduction_svc.deduct_grading_usage(
                        user_id=user_id,
                        provider=orchestrator.ai,
                        session_id=session_id,
                        db=db
                    )

                    logger.info(
                        f"Deducted {result['tokens_deducted']} tokens "
                        f"from user {user_id} for session {session_id}"
                    )

                    # Reload user to get updated balance for WebSocket broadcast
                    db_user = db.query(User).filter(User.id == user_id).first()

                except InsufficientTokensError as e:
                    # Grading succeeded but user can't afford the tokens
                    await ws_manager.broadcast_event(session_id, PROGRESS_EVENT_SESSION_ERROR, {
                        "error": "Insufficient tokens for grading",
                        "tokens_required": e.tokens_required,
                        "tokens_remaining": e.tokens_remaining
                    })
                    logger.error(f"Insufficient tokens for user {user_id}: {e.tokens_remaining} remaining, {e.tokens_required} required")
                    return  # Exit early, don't broadcast success

                except UserNotFoundError as e:
                    await ws_manager.broadcast_event(session_id, PROGRESS_EVENT_SESSION_ERROR, {
                        "error": "User not found"
                    })
                    logger.error(f"User not found during token deduction: {e}")
                    return

                except DeductionError as e:
                    await ws_manager.broadcast_event(session_id, PROGRESS_EVENT_SESSION_ERROR, {
                        "error": "Failed to record token usage"
                    })
                    logger.error(f"Token deduction failed for session {session_id}: {e}")
                    return

                except Exception as e:
                    # Catch-all for unexpected errors in token deduction
                    # Log full traceback for debugging
                    logger.exception(
                        f"Unexpected error during token deduction for session {session_id}, user {user_id}: {e}"
                    )
                    # Broadcast error but allow grading to complete
                    await ws_manager.broadcast_event(session_id, PROGRESS_EVENT_SESSION_ERROR, {
                        "error": "Token deduction encountered an error",
                        "details": str(e)
                    })
                    # DO NOT return - let grading complete with zero tokens deducted

                finally:
                    db.close()

                # Update progress
                if session_id in session_progress:
                    session_progress[session_id]["status"] = "complete"
                    session_progress[session_id]["copies_graded"] = len(session.graded_copies)

                # Broadcast completion with token usage
                await ws_manager.broadcast_event(session_id, PROGRESS_EVENT_SESSION_COMPLETE, {
                    "average_score": avg,
                    "total_copies": len(session.graded_copies),
                    "tokens_used": result['tokens_deducted'],
                    "remaining_tokens": db_user.remaining_tokens if db_user else 0
                })

            except Exception as e:
                logger.error(f"Grading error: {e}")
                await ws_manager.broadcast_event(session_id, PROGRESS_EVENT_SESSION_ERROR, {
                    "error": str(e)
                })
                if session_id in session_progress:
                    session_progress[session_id]["status"] = "error"
                    session_progress[session_id]["error"] = str(e)

        background_tasks.add_task(grade_task)

        # Update progress
        if session_id in session_progress:
            session_progress[session_id]["status"] = "correction"

        return GradeResponse(
            success=True,
            session_id=session_id,
            graded_count=len(session.graded_copies),
            total_count=len(session.copies) if session.copies else len(orchestrator.pdf_paths),
            pending_review=0,
            grading_mode=request.grading_mode
        )

    @app.post("/api/sessions/{session_id}/decisions")
    async def submit_decision(session_id: str, decision: TeacherDecisionRequest, current_user = Depends(get_current_user)):
        """
        Submit a teacher's grading decision.

        The system will extract a generalizable rule and propose
        applying it to similar copies.
        """
        user_id = current_user.id
        store = SessionStore(session_id, user_id=user_id)
        session = store.load_session()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if session_id not in active_sessions:
            orchestrator = GradingSessionOrchestrator(
                session_id=session_id,
                user_id=user_id,
                workflow_state=API_WORKFLOW_STATE
            )
            active_sessions[session_id] = orchestrator
        else:
            orchestrator = active_sessions[session_id]

        result = await orchestrator.apply_teacher_decision(
            question_id=decision.question_id,
            copy_id=decision.copy_id,
            teacher_guidance=decision.teacher_guidance,
            original_score=decision.original_score,
            new_score=decision.new_score
        )

        return {
            "success": True,
            "updated_count": result["updated_count"],
            "extracted_rule": result["extracted_rule"]
        }

    @app.get("/api/sessions/{session_id}/analytics", response_model=AnalyticsResponse)
    async def get_analytics(session_id: str, current_user = Depends(get_current_user)):
        """Get analytics for a session."""
        user_id = current_user.id
        from export.analytics import AnalyticsGenerator

        store = SessionStore(session_id, user_id=user_id)
        session = store.load_session()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        report = AnalyticsGenerator(session).generate()
        analytics = {
            "mean_score": report.mean_score,
            "median_score": report.median_score,
            "std_dev": report.std_dev,
            "min_score": report.min_score,
            "max_score": report.max_score,
            "score_distribution": report.score_distribution,
            "question_stats": {q: {"mean": s.get("mean", 0)} for q, s in report.question_stats.items()},
        }

        return AnalyticsResponse(**analytics)

    @app.get("/api/sessions")
    async def list_sessions(current_user = Depends(get_current_user)):
        """List all sessions for the current user."""
        from storage.file_store import SessionIndex

        user_id = current_user.id
        index = SessionIndex(user_id=user_id)
        sessions = index.list_sessions()

        # Build detailed session list
        session_list = []
        for session_id in sessions:
            store = SessionStore(session_id, user_id=user_id)
            session = store.load_session()
            if session:
                avg = None
                max_score = None
                if session.graded_copies:
                    scores = [g.total_score for g in session.graded_copies]
                    avg = sum(scores) / len(scores)
                # Get max_score from grading_scale (source of truth)
                if session.policy.question_weights:
                    max_score = sum(session.policy.question_weights.values())

                # Get subject from session.policy, or from detection exam_name
                subject = session.policy.subject
                if not subject:
                    # Try to get exam_name from detection
                    detection = store.load_detection()
                    if detection and detection.exam_name:
                        subject = detection.exam_name

                session_list.append({
                    "session_id": session_id,
                    "status": session.status,
                    "created_at": str(session.created_at),
                    "copies_count": len(session.copies),
                    "graded_count": len(session.graded_copies),
                    "average_score": avg,
                    "max_score": max_score,
                    "subject": subject,
                    "topic": session.policy.topic
                })

        return {"sessions": session_list}

    # ============================================================================
    # Disagreement Endpoints
    # ============================================================================

    @app.get("/api/sessions/{session_id}/disagreements", response_model=List[DisagreementResponse])
    async def get_disagreements(session_id: str, current_user = Depends(get_current_user)):
        """Get all disagreements for a session."""
        user_id = current_user.id
        store = SessionStore(session_id, user_id=user_id)
        session = store.load_session()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        disagreements = []
        for graded in session.graded_copies:
            # Use new grading_audit structure
            if graded.grading_audit:
                audit = graded.grading_audit

                # Find the copy for student name and index
                copy = next((c for c in session.copies if c.id == graded.copy_id), None)
                copy_index = session.copies.index(copy) + 1 if copy else 0

                for q_id, qaudit in audit.questions.items():
                    llm_results = qaudit.llm_results

                    # Get LLM results
                    llm_ids = list(llm_results.keys())
                    if len(llm_ids) < 2:
                        continue  # Skip if not dual LLM

                    llm1_result = llm_results.get(llm_ids[0])
                    llm2_result = llm_results.get(llm_ids[1])

                    if not llm1_result or not llm2_result:
                        continue

                    # Check if there was a disagreement
                    grade1 = llm1_result.grade
                    grade2 = llm2_result.grade
                    diff = abs(grade1 - grade2)

                    # Consider it a disagreement if difference > 0.5 points or no agreement
                    if diff > 0.5 or not qaudit.resolution.agreement:
                        # Find provider info
                        provider1 = next((p for p in audit.providers if p.id == llm_ids[0]), None)
                        provider2 = next((p for p in audit.providers if p.id == llm_ids[1]), None)

                        disagreements.append(
                            build_disagreement_response(
                                session=session,
                                graded=graded,
                                copy=copy,
                                copy_index=copy_index,
                                question_id=q_id,
                                question_audit=qaudit,
                                provider1=provider1,
                                provider2=provider2,
                                llm1_result=llm1_result,
                                llm2_result=llm2_result,
                                resolved=qaudit.resolution.agreement or False,
                            )
                        )

        return disagreements

    @app.get("/api/sessions/{session_id}/copies/{copy_id}/pdf")
    async def get_copy_pdf(session_id: str, copy_id: str, current_user = Depends(get_current_user)):
        """Stream the original PDF for a specific copy."""
        user_id = current_user.id
        store = SessionStore(session_id, user_id=user_id)
        session = store.load_session()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        copy = next((c for c in session.copies if c.id == copy_id), None)
        if not copy or not copy.pdf_path:
            raise HTTPException(status_code=404, detail="Copy not found")

        pdf_path = Path(copy.pdf_path)
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="PDF not found")

        def iterfile():
            with open(pdf_path, "rb") as f:
                yield from f

        return StreamingResponse(
            iterfile(),
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{pdf_path.name}"'},
        )

    @app.post("/api/sessions/{session_id}/disagreements/{copy_id}/{question_id}/resolve")
    async def resolve_disagreement(
        session_id: str,
        copy_id: str,
        question_id: str,
        request: ResolveDisagreementRequest,
        current_user = Depends(get_current_user)
    ):
        """Resolve a disagreement by marking it as agreed and optionally updating the grade."""
        user_id = current_user.id
        store = SessionStore(session_id, user_id=user_id)
        session = store.load_session()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Find the graded copy
        graded = next((g for g in session.graded_copies if g.copy_id == copy_id), None)
        if not graded:
            raise HTTPException(status_code=404, detail="Graded copy not found")

        # Update the disagreement status in grading_audit
        if graded.grading_audit and question_id in graded.grading_audit.questions:
            graded.grading_audit.questions[question_id].resolution.agreement = True

            # Update the grade based on the decision
            if request.action == "llm1":
                llm1_key = list(graded.grading_audit.questions[question_id].llm_results.keys())[0]
                new_grade = graded.grading_audit.questions[question_id].llm_results[llm1_key].grade
            elif request.action == "llm2":
                keys = list(graded.grading_audit.questions[question_id].llm_results.keys())
                llm2_key = keys[1] if len(keys) > 1 else keys[0]
                new_grade = graded.grading_audit.questions[question_id].llm_results[llm2_key].grade
            elif request.action == "average":
                results = list(graded.grading_audit.questions[question_id].llm_results.values())
                new_grade = sum(r.grade for r in results) / len(results)
            elif request.action == "custom" and request.custom_grade is not None:
                new_grade = request.custom_grade
            else:
                new_grade = graded.grades.get(question_id, 0)

            # Update the grade
            graded.grades[question_id] = new_grade
            graded.total_score = sum(graded.grades.values())

            # Persist the session
            store.save_session(session)

        return {"success": True, "question_id": question_id, "action": request.action}

    # ============================================================================
    # Export Endpoints
    # ============================================================================

    @app.get("/api/sessions/{session_id}/export/{format}")
    async def export_session(session_id: str, format: str, current_user = Depends(get_current_user)):
        """Export session results in CSV, JSON, or Excel format."""
        from export.analytics import DataExporter

        user_id = current_user.id
        store = SessionStore(session_id, user_id=user_id)
        session = store.load_session()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        reports_dir = store.get_reports_dir()
        reports_dir.mkdir(parents=True, exist_ok=True)

        exporter = DataExporter(session, str(reports_dir))

        if format == "csv":
            path = exporter.export_csv()
            media_type = "text/csv"
        elif format == "json":
            path = exporter.export_json()
            media_type = "application/json"
        elif format == "excel":
            path = exporter.export_excel()
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {format}. Supported: csv, json, excel"
            )

        def iterfile():
            with open(path, "rb") as f:
                yield from f

        filename = Path(path).name
        return StreamingResponse(
            iterfile(),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

    # ============================================================================
    # Provider Endpoints
    # ============================================================================

    @app.get("/api/providers", response_model=List[ProviderResponse])
    async def list_providers():
        """
        List available LLM providers based on .env configuration.

        Returns the providers that have API keys configured and their
        configured models from environment variables.
        """
        from ai.provider_factory import get_available_providers

        settings = get_settings()
        providers = []

        # Provider display names
        provider_names = {
            "gemini": "Google Gemini",
            "openai": "OpenAI",
        }

        # Get list of providers that have API keys configured
        available = get_available_providers()

        for provider_id in available:
            models = []

            # Add configured models for this provider
            if provider_id == "gemini":
                if settings.gemini_model:
                    models.append(ProviderModel(id=settings.gemini_model, name=settings.gemini_model))
                if settings.gemini_vision_model and settings.gemini_vision_model != settings.gemini_model:
                    models.append(ProviderModel(id=settings.gemini_vision_model, name=settings.gemini_vision_model))
                if settings.gemini_embedding_model:
                    models.append(ProviderModel(id=settings.gemini_embedding_model, name=settings.gemini_embedding_model))

            elif provider_id == "openai":
                # OpenAI provider uses model from llm1_model/llm2_model in comparison mode
                if settings.comparison_mode:
                    if settings.llm1_provider == "openai" and settings.llm1_model:
                        models.append(ProviderModel(id=settings.llm1_model, name=settings.llm1_model))
                    if settings.llm2_provider == "openai" and settings.llm2_model:
                        if not any(m.id == settings.llm2_model for m in models):
                            models.append(ProviderModel(id=settings.llm2_model, name=settings.llm2_model))

            providers.append(ProviderResponse(
                id=provider_id,
                name=provider_names.get(provider_id, provider_id.title()),
                type=provider_id,
                models=models,
                configured=True
            ))

        return providers

    # ============================================================================
    # Settings Endpoints
    # ============================================================================

    @app.get("/api/settings", response_model=SettingsResponse)
    async def get_settings_api(admin_user = Depends(get_admin_user)):
        """Get application settings (admin only)."""
        settings = get_settings()

        return SettingsResponse(
            ai_provider=settings.ai_provider,
            comparison_mode=settings.comparison_mode,
            llm1_provider=settings.llm1_provider,
            llm1_model=settings.llm1_model,
            llm2_provider=settings.llm2_provider,
            llm2_model=settings.llm2_model,
            confidence_auto=settings.confidence_auto,
            confidence_flag=settings.confidence_flag
        )

    @app.put("/api/settings", response_model=SettingsResponse)
    async def update_settings_api(request: UpdateSettingsRequest, admin_user = Depends(get_admin_user)):
        """Update application settings (admin only).

        Note: This only updates runtime settings. For persistent changes,
        modify the .env file.
        """
        settings = get_settings()

        # Update in-memory settings (non-persistent)
        if request.comparison_mode is not None:
            settings.comparison_mode = request.comparison_mode
        if request.llm1_provider is not None:
            settings.llm1_provider = request.llm1_provider
        if request.llm1_model is not None:
            settings.llm1_model = request.llm1_model
        if request.llm2_provider is not None:
            settings.llm2_provider = request.llm2_provider
        if request.llm2_model is not None:
            settings.llm2_model = request.llm2_model
        if request.confidence_auto is not None:
            settings.confidence_auto = request.confidence_auto
        if request.confidence_flag is not None:
            settings.confidence_flag = request.confidence_flag

        return SettingsResponse(
            ai_provider=settings.ai_provider,
            comparison_mode=settings.comparison_mode,
            llm1_provider=settings.llm1_provider,
            llm1_model=settings.llm1_model,
            llm2_provider=settings.llm2_provider,
            llm2_model=settings.llm2_model,
            confidence_auto=settings.confidence_auto,
            confidence_flag=settings.confidence_flag
        )

    return app


# Create app instance
app = create_app()
