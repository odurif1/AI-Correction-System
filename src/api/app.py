"""
FastAPI application for the AI correction system.

Provides web API for grading operations with WebSocket support for real-time progress.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import shutil
import uuid
import re
import json
import logging
import os

# Maximum file size for uploads (50 MB)
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

from config.settings import get_settings
from core.session import GradingSessionOrchestrator
from storage.session_store import SessionStore
from api.websocket import manager as ws_manager
from api.schemas import (
    CreateSessionRequest, SessionResponse, SessionDetailResponse,
    GradeResponse, TeacherDecisionRequest,
    DisagreementResponse, ResolveDisagreementRequest,
    AnalyticsResponse, ProviderResponse, ProviderModel,
    SettingsResponse, UpdateSettingsRequest, ExportOptions
)

logger = logging.getLogger(__name__)

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
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    )

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
        background_tasks: BackgroundTasks,
        api_key: str = Depends(get_api_key)
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

        # Save initial session
        orchestrator._save_sync()

        active_sessions[session_id] = orchestrator
        session_progress[session_id] = {
            "status": "created",
            "copies_uploaded": 0,
            "copies_graded": 0,
            "disagreements": []
        }

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
        api_key: str = Depends(get_api_key)
    ):
        """
        Upload PDF copies to a session.
        """
        if session_id not in active_sessions:
            # Try to load existing session
            store = SessionStore(session_id)
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
        for file in files:
            if not file.filename or not file.filename.lower().endswith(".pdf"):
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
                    raise HTTPException(status_code=400, detail="Invalid file path")
            except (OSError, ValueError):
                raise HTTPException(status_code=400, detail="Invalid file path")

            # Check file size
            file.file.seek(0, 2)  # Seek to end
            file_size = file.file.tell()
            file.file.seek(0)  # Reset to beginning

            if file_size > MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)} MB"
                )

            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            pdf_paths.append(str(file_path))

        # Update session progress
        if session_id in session_progress:
            session_progress[session_id]["copies_uploaded"] = len(pdf_paths)
            session_progress[session_id]["status"] = "uploaded"

        return {
            "session_id": session_id,
            "uploaded_count": len(pdf_paths),
            "paths": pdf_paths
        }

    @app.get("/api/sessions/{session_id}", response_model=SessionDetailResponse)
    async def get_session(session_id: str, api_key: str = Depends(get_api_key)):
        """Get detailed session information."""
        store = SessionStore(session_id)
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
    async def delete_session(session_id: str, api_key: str = Depends(get_api_key)):
        """Delete a session and all its data."""
        store = SessionStore(session_id)

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
    async def start_grading(session_id: str, background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)):
        """
        Start the grading process for a session.

        Progress updates are sent via WebSocket at /api/sessions/{session_id}/ws
        """
        store = SessionStore(session_id)
        session = store.load_session()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get or create orchestrator
        if session_id not in active_sessions:
            orchestrator = GradingSessionOrchestrator(session_id=session_id)
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
            try:
                # Run analysis phase
                await orchestrator.analyze_only()

                # Confirm scale (use detected or default)
                orchestrator.confirm_scale(orchestrator.question_scales)

                # Grade with progress callback
                await orchestrator.grade_all(progress_callback=progress_callback)

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
    async def submit_decision(session_id: str, decision: TeacherDecisionRequest, api_key: str = Depends(get_api_key)):
        """
        Submit a teacher's grading decision.

        The system will extract a generalizable rule and propose
        applying it to similar copies.
        """
        store = SessionStore(session_id)
        session = store.load_session()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if session_id not in active_sessions:
            orchestrator = GradingSessionOrchestrator(session_id=session_id)
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
    async def get_analytics(session_id: str, api_key: str = Depends(get_api_key)):
        """Get analytics for a session."""
        orchestrator = GradingSessionOrchestrator(session_id=session_id)
        analytics = orchestrator.get_analytics()

        return AnalyticsResponse(**analytics)

    @app.get("/api/sessions")
    async def list_sessions(api_key: str = Depends(get_api_key)):
        """List all sessions."""
        from storage.file_store import SessionIndex

        index = SessionIndex()
        sessions = index.list_sessions()

        # Build detailed session list
        session_list = []
        for session_id in sessions:
            store = SessionStore(session_id)
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
    async def get_disagreements(session_id: str, api_key: str = Depends(get_api_key)):
        """Get all disagreements for a session."""
        store = SessionStore(session_id)
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
        api_key: str = Depends(get_api_key)
    ):
        """Resolve a disagreement."""
        store = SessionStore(session_id)
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
    async def export_session(session_id: str, format: str, api_key: str = Depends(get_api_key)):
        """Export session results in the specified format."""
        from export.analytics import DataExporter

        store = SessionStore(session_id)
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
    async def get_settings_api():
        """Get application settings."""
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
    async def update_settings_api(request: UpdateSettingsRequest, api_key: str = Depends(get_api_key)):
        """Update application settings.

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
