"""
FastAPI application for La Corrigeuse.

Provides web API for grading operations with WebSocket support for real-time progress.
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

# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


def get_user_id(request: Request) -> str:
    """
    Extract user ID from JWT token, fallback to IP.

    For authenticated requests, rate limit per user.
    For unauthenticated requests (e.g., login), rate limit per IP.
    """
    # Try to get user from request state (set by auth middleware)
    if hasattr(request.state, 'user_id'):
        return f"user:{request.state.user_id}"
    # Fallback to IP for unauthenticated requests
    return f"ip:{get_remote_address(request)}"


# Create limiter at module level for use in decorators
limiter = Limiter(key_func=get_user_id)

# Maximum file size for uploads (50 MB)
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

# Maximum batch size for PDF uploads
MAX_BATCH_SIZE = 50

# Import workflow state before defining constant
from core.workflow_state import CorrectionState
from core.models import SessionStatus

# API runs in auto-mode (no CLI interaction)
API_WORKFLOW_STATE = CorrectionState(auto_mode=True)


from config.settings import get_settings
from pydantic import ValidationError
from core.session import GradingSessionOrchestrator
from storage.session_store import SessionStore
from api.websocket import manager as ws_manager
from api.schemas import (
    CreateSessionRequest, SessionResponse, SessionDetailResponse,
    GradeResponse, TeacherDecisionRequest,
    DisagreementResponse, ResolveDisagreementRequest,
    AnalyticsResponse, ProviderResponse, ProviderModel,
    SettingsResponse, UpdateSettingsRequest, ExportOptions,
    PreAnalysisRequest, PreAnalysisResponse, ConfirmPreAnalysisRequest,
    ConfirmPreAnalysisResponse, StudentInfoSchema, CandidateScale,
    StartGradingRequest, UpdateGradeRequest, UpdateGradeResponse, UpdateStudentNameRequest
)

# Import auth module
from api.auth import router as auth_router, get_current_user, get_admin_user

# Import token deduction service
from services.token_service import TokenDeductionService
from services.token_service import InsufficientTokensError, UserNotFoundError, DeductionError

# Import health check router
from api.health import router as health_router

# Import subscription router
from api.subscription import router as subscription_router

# Import metrics collector
from utils.metrics import get_metrics_collector

# Import Sentry error handler
from middleware.error_handler import init_sentry, sentry_exception_handler, set_user_context

# Import correlation ID middleware
from asgi_correlation_id import CorrelationIdMiddleware

# Import structured logging
from loguru import logger
from config.logging_config import setup_structured_logging

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
        title="La Corrigeuse",
        description="Correction automatique par IA pour les professeurs de collège et lycée",
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

    # Include subscription router
    app.include_router(subscription_router, prefix="/api")

    # Session storage (in-memory for active grading)
    active_sessions: Dict[str, GradingSessionOrchestrator] = {}
    session_progress: Dict[str, Dict[str, Any]] = {}

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
        await ws_manager.connect(websocket, session_id)
        try:
            # Send current progress state on connect (for reconnection)
            if session_id in session_progress:
                progress = session_progress[session_id]
                await websocket.send_json({
                    "type": "progress_sync",
                    "data": {
                        "status": progress.get("status"),
                        "copies_uploaded": progress.get("copies_uploaded", 0),
                        "copies_graded": progress.get("copies_graded", 0),
                        "grading_mode": progress.get("grading_mode", "dual")
                    }
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
                        await websocket.send_json({
                            "type": "progress_sync",
                            "data": session_progress[session_id]
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
            "name": "La Corrigeuse",
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

        # Check quota
        db = SessionLocal()
        try:
            db_user = db.query(User).filter(User.id == user_id).first()
            if db_user:
                # Count PDF files
                pdf_count = sum(1 for f in files if f.filename and f.filename.lower().endswith(".pdf"))
                if not db_user.can_grade_copies(pdf_count):
                    raise HTTPException(
                        status_code=402,  # Payment Required
                        detail=f"Quota dépassé. Vous avez {db_user.remaining_copies} copies restantes sur {pdf_count} demandées."
                    )
        finally:
            db.close()

        if session_id not in active_sessions:
            # Try to load existing session
            store = SessionStore(session_id, user_id=user_id)
            if not store.exists():
                raise HTTPException(status_code=404, detail="Session not found")
            # Create orchestrator for existing session
            orchestrator = GradingSessionOrchestrator(
                session_id=session_id,
                workflow_state=API_WORKFLOW_STATE
            )
            active_sessions[session_id] = orchestrator

        orchestrator = active_sessions[session_id]

        # Save uploaded files
        upload_dir = Path(f"temp/{session_id}")
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Resolve to absolute path for validation
        upload_dir_resolved = upload_dir.resolve()

        pdf_paths = []
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
                validation_results.append({
                    "filename": file.filename,
                    "success": True,
                    "path": str(file_path)
                })
            except Exception as e:
                validation_results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })

        # Update session progress
        if session_id in session_progress:
            session_progress[session_id]["copies_uploaded"] = len(pdf_paths)
            session_progress[session_id]["status"] = "uploaded"

        return {
            "session_id": session_id,
            "uploaded_count": len(pdf_paths),
            "paths": pdf_paths,
            "validation_results": validation_results
        }

    @app.post("/api/sessions/{session_id}/pre-analyze", response_model=PreAnalysisResponse)
    async def pre_analyze_session(
        session_id: str,
        request: PreAnalysisRequest,
        background_tasks: BackgroundTasks,
        current_user = Depends(get_current_user)
    ):
        """
        Pre-analyze the uploaded PDF to detect structure and grading scale.

        This analysis is performed before grading to:
        - Validate the PDF contains student copies
        - Detect document structure (one student or multiple per PDF)
        - Detect grading scale / barème (with multiple candidate scales if uncertain)
        - Identify blocking issues

        Results are cached for the session.
        """
        from analysis.pre_analysis import PreAnalyzer

        user_id = current_user.id

        # Check session exists
        store = SessionStore(session_id, user_id=user_id)
        if not store.exists():
            raise HTTPException(status_code=404, detail="Session not found")

        # Get uploaded PDF
        upload_dir = Path(f"temp/{session_id}")
        if not upload_dir.exists():
            raise HTTPException(status_code=400, detail="No PDF uploaded yet")

        pdf_files = list(upload_dir.glob("*.pdf"))
        if not pdf_files:
            raise HTTPException(status_code=400, detail="No PDF file found")

        # Use the first PDF (we only support one PDF per session)
        pdf_path = str(pdf_files[0])

        # Check tier-based limits for re-diagnosis (force_refresh)
        if request.force_refresh:
            from db import SubscriptionTier
            tier = current_user.subscription_tier

            # FREE tier cannot re-diagnose (consumes ~5-7K tokens)
            if tier == SubscriptionTier.FREE:
                raise HTTPException(
                    status_code=403,
                    detail="Re-diagnostic non disponible sur le plan FREE. Passez à ESSENTIEL pour re-analyser."
                )

            # ESSENTIEL can only re-diagnose once per session
            if tier == SubscriptionTier.ESSENTIEL:
                existing = store.load_pre_analysis()
                if existing and not existing.cached:
                    raise HTTPException(
                        status_code=403,
                        detail="Re-diagnostic limité à 1x par session sur ESSENTIEL. Passez à PRO pour illimité."
                    )

        # Update session progress
        if session_id in session_progress:
            session_progress[session_id]["status"] = "diagnostic"

        # Update session file status
        session = store.load_session()
        if session:
            session.status = SessionStatus.DIAGNOSTIC
            store.save_session(session)

        try:
            # Run pre-analysis
            analyzer = PreAnalyzer(
                user_id=user_id,
                session_id=session_id,
                language="fr"  # TODO: Get from user preferences
            )

            result = analyzer.analyze(pdf_path, force_refresh=request.force_refresh)

            # Debug log for exam_name
            logger.info(f"Pre-analysis result exam_name: {result.exam_name}")

            # Store result in session for later use
            store.save_pre_analysis(result)

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
                session_progress[session_id]["status"] = "diagnostic"

            # Update session file status
            session = store.load_session()
            if session:
                session.status = SessionStatus.DIAGNOSTIC
                store.save_session(session)

            return PreAnalysisResponse(
                analysis_id=result.analysis_id,
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
                cached=result.cached,
                analysis_duration_ms=result.analysis_duration_ms,
                exam_name=result.exam_name
            )

        except Exception as e:
            logger.error(f"Pre-analysis error: {e}")
            if session_id in session_progress:
                session_progress[session_id]["status"] = "error"
                session_progress[session_id]["error"] = str(e)
            raise HTTPException(status_code=500, detail=f"Pre-analysis failed: {str(e)}")

    @app.post("/api/sessions/{session_id}/confirm-pre-analysis", response_model=ConfirmPreAnalysisResponse)
    async def confirm_pre_analysis(
        session_id: str,
        request: ConfirmPreAnalysisRequest,
        background_tasks: BackgroundTasks,
        current_user = Depends(get_current_user)
    ):
        """
        Confirm pre-analysis and automatically start grading.

        Allows:
        - Selecting from multiple candidate grading scales
        - Adjusting the detected grading scale
        - Overriding detected student names

        After confirmation, grading starts automatically in background.
        """
        user_id = current_user.id

        # Check session exists
        logger.info(f"Confirm pre-analysis: session_id={session_id}, user_id={user_id}")
        store = SessionStore(session_id, user_id=user_id)
        logger.info(f"Session dir: {store.session_dir}, exists: {store.exists()}")
        if not store.exists():
            logger.warning(f"Session not found: {session_id} for user {user_id}")
            raise HTTPException(status_code=404, detail="Session not found")

        # Load pre-analysis result
        pre_analysis = store.load_pre_analysis()
        if not pre_analysis:
            raise HTTPException(status_code=400, detail="No pre-analysis found. Run /pre-analyze first.")

        # Check for blocking issues
        if pre_analysis.has_blocking_issues:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot confirm: blocking issues detected: {pre_analysis.blocking_issues}"
            )

        # Start with detected grading scale
        grading_scale = dict(pre_analysis.grading_scale)

        # Apply selected_scale_index if provided
        if request.selected_scale_index is not None:
            candidate_scales = pre_analysis.candidate_scales
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
                # Update student names in pre_analysis result
                student_names = request.adjustments["student_names"]
                for student_idx, name in student_names.items():
                    # Find the student by index and update name
                    for student in pre_analysis.students:
                        if student.index == student_idx:
                            student.name = name
                            break
                # Save updated pre-analysis with new names
                store.save_pre_analysis(pre_analysis)

        # Update session with confirmed settings
        session = store.load_session()
        if session:
            # Set question weights from final grading scale
            session.policy.question_weights = grading_scale
            session.status = SessionStatus.CORRECTION  # Grading starts now

            # Propagate exam_name to session.policy.subject if not already set
            if pre_analysis.exam_name and not session.policy.subject:
                session.policy.subject = pre_analysis.exam_name

            # Store student name adjustments in session metadata for later use
            if not session.storage_path:
                session.storage_path = str(store.session_dir)

            store.save_session(session)

        # Update progress
        if session_id in session_progress:
            session_progress[session_id]["status"] = "correction"

        # Auto-start grading after confirmation
        user_id = current_user.id

        # Get or create orchestrator with grading configuration
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
            # Update grading configuration on existing orchestrator
            orchestrator._grading_mode = request.grading_mode
            orchestrator._batch_verify = request.batch_verify

        # Get uploaded PDF paths
        upload_dir = Path(f"temp/{session_id}")
        if upload_dir.exists():
            pdf_paths = [str(p) for p in upload_dir.glob("*.pdf")]
            orchestrator.pdf_paths = pdf_paths

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

                await ws_manager.broadcast_event(session_id, PROGRESS_EVENT_SESSION_COMPLETE, {
                    "average_score": avg,
                    "total_copies": len(session.graded_copies) if session else 0
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

        return ConfirmPreAnalysisResponse(
            success=True,
            session_id=session_id,
            status="correction",  # Status changed - grading started automatically
            grading_scale=grading_scale,
            num_students=pre_analysis.num_students_detected
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

        # Get subject from session.policy, or from pre-analysis exam_name
        subject = session.policy.subject
        if not subject:
            # Try to get exam_name from pre-analysis
            pre_analysis = store.load_pre_analysis()
            if pre_analysis and pre_analysis.exam_name:
                subject = pre_analysis.exam_name

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
            copies=copies,
            graded_copies=graded_copies,
            question_weights=session.policy.question_weights
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

        # Delete the session directory
        success = store.delete()

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
        upload_dir = Path(f"temp/{session_id}")
        if upload_dir.exists():
            pdf_paths = [str(p) for p in upload_dir.glob("*.pdf")]
            orchestrator.pdf_paths = pdf_paths

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
                db = SessionLocal()
                try:
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
        orchestrator = GradingSessionOrchestrator(
            session_id=session_id,
            user_id=user_id,
            workflow_state=API_WORKFLOW_STATE
        )
        analytics = orchestrator.get_analytics()

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

                # Get subject from session.policy, or from pre-analysis exam_name
                subject = session.policy.subject
                if not subject:
                    # Try to get exam_name from pre-analysis
                    pre_analysis = store.load_pre_analysis()
                    if pre_analysis and pre_analysis.exam_name:
                        subject = pre_analysis.exam_name

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

                        disagreements.append(DisagreementResponse(
                            copy_id=graded.copy_id,
                            copy_index=copy_index,
                            student_name=copy.student_name if copy else None,
                            question_id=q_id,
                            max_points=qaudit.resolution.final_max_points,
                            llm1={
                                "provider": provider1.model if provider1 else "unknown",
                                "model": provider1.model if provider1 else "unknown",
                                "grade": grade1,
                                "confidence": llm1_result.confidence,
                                "reasoning": llm1_result.reasoning,
                                "reading": llm1_result.reading
                            },
                            llm2={
                                "provider": provider2.model if provider2 else "unknown",
                                "model": provider2.model if provider2 else "unknown",
                                "grade": grade2,
                                "confidence": llm2_result.confidence,
                                "reasoning": llm2_result.reasoning,
                                "reading": llm2_result.reading
                            },
                            resolved=qaudit.resolution.agreement or False
                        ))

        return disagreements

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
