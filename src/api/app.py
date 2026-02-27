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
    ConfirmPreAnalysisResponse, StudentInfoSchema, CandidateScale
)

# Import auth module
from api.auth import router as auth_router, get_current_user, get_admin_user

# Import health check router
from api.health import router as health_router

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
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
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

        # One-time cleanup: Delete shared sessions (pre-multi-tenant)
        await _cleanup_legacy_sessions()


    async def _cleanup_legacy_sessions():
        """Delete legacy sessions in shared namespace (pre-user-isolation)."""
        import shutil
        from pathlib import Path

        sessions_dir = Path("data/sessions")

        # Check if sessions directory exists
        if not sessions_dir.exists():
            return

        # Find items in shared namespace (direct children, not user_id subdirs)
        legacy_items = []
        for item in sessions_dir.iterdir():
            if item.is_dir() and not item.name.startswith("__"):  # Exclude __pycache__ etc
                # If it doesn't look like a UUID, it's likely a user_id dir, keep it
                # If it looks like a session_id (UUID format), it's legacy
                if len(item.name) == 36 and item.name.count("-") == 4:  # UUID format
                    legacy_items.append(item)

        if legacy_items:
            logger.warning(f"Found {len(legacy_items)} legacy sessions in shared namespace, deleting...")
            for item in legacy_items:
                shutil.rmtree(item)
                logger.info(f"Deleted legacy session: {item.name}")
            logger.info("Legacy session cleanup complete. Fresh start enforced.")

    # Include auth router
    app.include_router(auth_router, prefix="/api")

    # Include health check router
    app.include_router(health_router, tags=["health"])

    # Session storage (in-memory for active grading)
    active_sessions: Dict[str, GradingSessionOrchestrator] = {}
    session_progress: Dict[str, Dict[str, Any]] = {}

    # ============================================================================
    # WebSocket Endpoint
    # ============================================================================

    @app.websocket("/api/sessions/{session_id}/ws")
    async def websocket_progress(websocket: WebSocket, session_id: str):
        """
        WebSocket for real-time grading progress.

        Sends events:
        - copy_start: When a copy starts grading
        - question_done: When a question is graded
        - copy_done: When a copy is fully graded
        - copy_error: When grading fails for a copy
        - session_complete: When the session is complete
        """
        await ws_manager.connect(websocket, session_id)
        try:
            # Keep connection alive and handle any client messages
            while True:
                data = await websocket.receive_text()
                # Handle ping/pong or other client messages
                if data == "ping":
                    await websocket.send_text("pong")
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
        orchestrator = GradingSessionOrchestrator(user_id=user_id)

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
            orchestrator = GradingSessionOrchestrator(session_id=session_id)
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
            session_progress[session_id]["status"] = "pre_analyzing"

        try:
            # Run pre-analysis
            analyzer = PreAnalyzer(
                user_id=user_id,
                session_id=session_id,
                language="fr"  # TODO: Get from user preferences
            )

            result = analyzer.analyze(pdf_path, force_refresh=request.force_refresh)

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
                session_progress[session_id]["status"] = "pre_analyzed"

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
                analysis_duration_ms=result.analysis_duration_ms
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
        current_user = Depends(get_current_user)
    ):
        """
        Confirm pre-analysis and prepare for grading.

        Allows:
        - Selecting from multiple candidate grading scales
        - Adjusting the detected grading scale
        - Overriding detected student names

        before starting the actual grading process.
        """
        user_id = current_user.id

        # Check session exists
        store = SessionStore(session_id, user_id=user_id)
        if not store.exists():
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
            session.status = "ready_for_grading"

            # Store student name adjustments in session metadata for later use
            if not session.storage_path:
                session.storage_path = store.get_storage_dir()

            store.save_session(session)

        # Update progress
        if session_id in session_progress:
            session_progress[session_id]["status"] = "ready_for_grading"

        return ConfirmPreAnalysisResponse(
            success=True,
            session_id=session_id,
            status="ready_for_grading",
            grading_scale=grading_scale,
            num_students=pre_analysis.num_students_detected
        )

    @app.get("/api/sessions/{session_id}", response_model=SessionDetailResponse)
    async def get_session(session_id: str, current_user = Depends(get_current_user)):
        """Get detailed session information."""
        user_id = current_user.id
        store = SessionStore(session_id, user_id=user_id)
        session = store.load_session()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        average = None
        if session.graded_copies:
            scores = [g.total_score for g in session.graded_copies]
            average = sum(scores) / len(scores)

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

        # Build graded copies list
        graded_copies = []
        for graded in session.graded_copies:
            graded_info = {
                "copy_id": graded.copy_id,
                "total_score": graded.total_score,
                "max_score": graded.max_score,
                "confidence": graded.confidence,
                "grades": graded.grades
            }
            graded_copies.append(graded_info)

        return SessionDetailResponse(
            session_id=session_id,
            status=session.status,
            created_at=str(session.created_at),
            copies_count=len(session.copies),
            graded_count=len(session.graded_copies),
            average_score=average,
            subject=session.policy.subject,
            topic=session.policy.topic,
            copies=copies,
            graded_copies=graded_copies,
            question_weights=session.policy.question_weights
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

        # Delete the session directory
        success = store.delete()

        return {"success": success}

    @app.post("/api/sessions/{session_id}/grade", response_model=GradeResponse)
    async def start_grading(session_id: str, background_tasks: BackgroundTasks, current_user = Depends(get_current_user)):
        """
        Start the grading process for a session.

        Progress updates are sent via WebSocket at /api/sessions/{session_id}/ws
        """
        user_id = current_user.id
        store = SessionStore(session_id, user_id=user_id)
        session = store.load_session()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get or create orchestrator
        if session_id not in active_sessions:
            orchestrator = GradingSessionOrchestrator(session_id=session_id, user_id=user_id)
            active_sessions[session_id] = orchestrator
        else:
            orchestrator = active_sessions[session_id]

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
                await orchestrator.grade_all(progress_callback=progress_callback)

                # Record grading operation with token usage
                if hasattr(app.state, 'metrics_collector'):
                    # Get token usage from session (simplified - tracks copy count)
                    token_usage = len(session.graded_copies) * 10000  # Estimate
                    app.state.metrics_collector.record_grading_operation(
                        session_id=session_id,
                        tokens_used=token_usage
                    )

                # Notify completion
                if session.graded_copies:
                    scores = [g.total_score for g in session.graded_copies]
                    avg = sum(scores) / len(scores)
                else:
                    avg = 0

                await ws_manager.broadcast_event(session_id, "session_complete", {
                    "average_score": avg,
                    "total_copies": len(session.graded_copies)
                })

                # Update progress
                if session_id in session_progress:
                    session_progress[session_id]["status"] = "complete"
                    session_progress[session_id]["copies_graded"] = len(session.graded_copies)

                # Increment user usage counter
                db = SessionLocal()
                try:
                    db_user = db.query(User).filter(User.id == user_id).first()
                    if db_user and len(session.graded_copies) > 0:
                        db_user.increment_usage(len(session.graded_copies))
                        db.commit()
                finally:
                    db.close()

            except Exception as e:
                logger.error(f"Grading error: {e}")
                await ws_manager.broadcast_event(session_id, "session_error", {
                    "error": str(e)
                })
                if session_id in session_progress:
                    session_progress[session_id]["status"] = "error"
                    session_progress[session_id]["error"] = str(e)

        background_tasks.add_task(grade_task)

        # Update progress
        if session_id in session_progress:
            session_progress[session_id]["status"] = "grading"

        return GradeResponse(
            success=True,
            session_id=session_id,
            graded_count=len(session.graded_copies),
            total_count=len(session.copies) if session.copies else len(orchestrator.pdf_paths),
            pending_review=0
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
            orchestrator = GradingSessionOrchestrator(session_id=session_id, user_id=user_id)
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
        orchestrator = GradingSessionOrchestrator(session_id=session_id, user_id=user_id)
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
                if session.graded_copies:
                    scores = [g.total_score for g in session.graded_copies]
                    avg = sum(scores) / len(scores)

                session_list.append({
                    "session_id": session_id,
                    "status": session.status,
                    "created_at": str(session.created_at),
                    "copies_count": len(session.copies),
                    "graded_count": len(session.graded_copies),
                    "average_score": avg,
                    "subject": session.policy.subject,
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

    @app.post("/api/sessions/{session_id}/disagreements/{question_id}/resolve")
    async def resolve_disagreement(
        session_id: str,
        question_id: str,
        request: ResolveDisagreementRequest,
        current_user = Depends(get_current_user)
    ):
        """Resolve a disagreement."""
        user_id = current_user.id
        store = SessionStore(session_id, user_id=user_id)
        session = store.load_session()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # This is a simplified resolution - in a real implementation,
        # you would update the actual graded copy
        # For now, just acknowledge the resolution

        return {"success": True, "question_id": question_id, "action": request.action}

    # ============================================================================
    # Export Endpoints
    # ============================================================================

    @app.get("/api/sessions/{session_id}/export/{format}")
    async def export_session(session_id: str, format: str, current_user = Depends(get_current_user)):
        """Export session results in the specified format."""
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
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

        def iterfile():
            with open(path, "rb") as f:
                yield from f

        return StreamingResponse(
            iterfile(),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={Path(path).name}"}
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
