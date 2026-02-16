"""
FastAPI application for the AI correction system.

Provides web API for grading operations.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from pathlib import Path
import shutil
import uuid

from config.settings import get_settings
from core.session import GradingSessionOrchestrator
from storage.session_store import SessionStore


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateSessionRequest(BaseModel):
    """Request to create a new grading session."""
    subject: Optional[str] = None
    topic: Optional[str] = None
    total_questions: int = 0
    question_weights: Dict[str, float] = {}


class SessionResponse(BaseModel):
    """Session information."""
    session_id: str
    status: str
    created_at: str
    copies_count: int
    graded_count: int
    average_score: Optional[float] = None


class TeacherDecisionRequest(BaseModel):
    """Request to provide teacher guidance."""
    question_id: str
    copy_id: str
    teacher_guidance: str
    original_score: float
    new_score: float
    applies_to_all: bool = True


class GradeResponse(BaseModel):
    """Grading result."""
    success: bool
    session_id: str
    graded_count: int
    total_count: int
    pending_review: int


class AnalyticsResponse(BaseModel):
    """Analytics data."""
    mean_score: float
    median_score: float
    min_score: float
    max_score: float
    std_dev: float
    score_distribution: Dict[str, int]


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
        description="Intelligent grading system with cross-copy analysis and retroactive calibration",
        version="1.0.0"
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Session storage (in-memory for now)
    active_sessions: Dict[str, GradingSessionOrchestrator] = {}

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

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.post("/api/sessions", response_model=SessionResponse)
    async def create_session(
        request: CreateSessionRequest,
        background_tasks: BackgroundTasks
    ):
        """
        Create a new grading session.

        Upload PDFs via /api/sessions/{id}/upload
        """
        session_id = str(uuid.uuid4())[:8]
        orchestrator = GradingSessionOrchestrator(session_id=session_id)

        # Set policy from request
        if request.subject:
            orchestrator.session.policy.subject = request.subject
        if request.topic:
            orchestrator.session.policy.topic = request.topic
        if request.question_weights:
            orchestrator.session.policy.question_weights = request.question_weights

        active_sessions[session_id] = orchestrator

        return SessionResponse(
            session_id=session_id,
            status=orchestrator.session.status,
            created_at=str(orchestrator.session.created_at),
            copies_count=0,
            graded_count=0
        )

    @app.post("/api/sessions/{session_id}/upload")
    async def upload_copies(
        session_id: str,
        files: List[UploadFile] = File(...)
    ):
        """
        Upload PDF copies to a session.
        """
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        orchestrator = active_sessions[session_id]

        # Save uploaded files
        upload_dir = Path(f"temp/{session_id}")
        upload_dir.mkdir(parents=True, exist_ok=True)

        pdf_paths = []
        for file in files:
            if not file.filename.endswith(".pdf"):
                continue

            file_path = upload_dir / file.filename
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            pdf_paths.append(str(file_path))

        return {
            "session_id": session_id,
            "uploaded_count": len(pdf_paths),
            "paths": pdf_paths
        }

    @app.get("/api/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(session_id: str):
        """Get session information."""
        store = SessionStore(session_id)
        session = store.load_session()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        average = None
        if session.graded_copies:
            scores = [g.total_score for g in session.graded_copies]
            average = sum(scores) / len(scores)

        return SessionResponse(
            session_id=session_id,
            status=session.status,
            created_at=str(session.created_at),
            copies_count=len(session.copies),
            graded_count=len(session.graded_copies),
            average_score=average
        )

    @app.post("/api/sessions/{session_id}/grade", response_model=GradeResponse)
    async def start_grading(session_id: str, background_tasks: BackgroundTasks):
        """
        Start the grading process for a session.
        """
        store = SessionStore(session_id)
        session = store.load_session()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Start grading in background
        async def grade_task():
            orchestrator = GradingSessionOrchestrator(session_id=session_id)
            await orchestrator.run()

        background_tasks.add_task(grade_task)

        return GradeResponse(
            success=True,
            session_id=session_id,
            graded_count=len(session.graded_copies),
            total_count=len(session.copies),
            pending_review=0
        )

    @app.post("/api/sessions/{session_id}/decisions")
    async def submit_decision(session_id: str, decision: TeacherDecisionRequest):
        """
        Submit a teacher's grading decision.

        The system will extract a generalizable rule and propose
        applying it to similar copies.
        """
        store = SessionStore(session_id)
        session = store.load_session()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        orchestrator = GradingSessionOrchestrator(session_id=session_id)

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
    async def get_analytics(session_id: str):
        """Get analytics for a session."""
        orchestrator = GradingSessionOrchestrator(session_id=session_id)
        analytics = orchestrator.get_analytics()

        return AnalyticsResponse(**analytics)

    @app.get("/api/sessions")
    async def list_sessions():
        """List all sessions."""
        from storage.file_store import SessionIndex

        index = SessionIndex()
        sessions = index.list_sessions()

        return {"sessions": sessions}

    return app


# Create app instance
app = create_app()
