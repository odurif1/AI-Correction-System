"""Dedicated worker runtime for persistent grading jobs."""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

from config.constants import DATA_DIR
from core.models import DocumentDecision, DocumentType, SessionStatus
from core.services.document_preparation_service import DocumentPreparationService
from core.session import GradingSessionOrchestrator
from db import SessionLocal, User
from services.job_service import (
    JobService,
    PROGRESS_EVENT_SESSION_COMPLETE,
    PROGRESS_EVENT_SESSION_ERROR,
)
from services.token_service import (
    DeductionError,
    InsufficientTokensError,
    TokenDeductionService,
    UserNotFoundError,
)
from storage.session_store import SessionStore

document_preparation_service = DocumentPreparationService()


def build_worker_id() -> str:
    """Generate a stable-enough worker identifier for logs and leases."""
    return f"{socket.gethostname()}-{uuid4().hex[:8]}"


def list_uploaded_pdfs(session_id: str) -> list[Path]:
    """Return uploaded PDFs for a session in deterministic order."""
    upload_dir = Path(DATA_DIR) / "temp" / session_id
    if not upload_dir.exists():
        return []
    return sorted(upload_dir.glob("*.pdf"))


def get_session_copy_paths(session) -> list[str]:
    """Resolve the student-copy PDFs that should feed grading."""
    copy_documents = document_preparation_service.get_copy_documents(session)
    if copy_documents:
        return [doc.storage_path for doc in copy_documents]

    classified = [
        doc for doc in session.documents
        if doc.detected_type == DocumentType.STUDENT_COPIES
        and doc.user_decision != DocumentDecision.EXCLUDE
    ]
    if classified:
        return [doc.storage_path for doc in classified]

    return [str(path) for path in list_uploaded_pdfs(session.session_id)]


def create_persistent_progress_callback(job_id: str):
    """Persist worker progress events into the jobs table."""

    async def callback(event_type: str, data: dict[str, Any]):
        db = SessionLocal()
        try:
            JobService(db).record_event(job_id, event_type, data)
        finally:
            db.close()

    return callback


async def run_grading_job(job_id: str) -> None:
    """Execute a single grading job already claimed by a worker."""
    db = SessionLocal()
    try:
        service = JobService(db)
        job = service.get_job(job_id)
        if not job:
            raise RuntimeError(f"Unknown job {job_id}")

        payload = dict(job.payload or {})
        session_id = job.session_id
        user_id = job.user_id

        store = SessionStore(session_id, user_id=user_id)
        session = store.load_session()
        if not session:
            raise RuntimeError(f"Session {session_id} not found")

        service.touch_job(job_id, stage="analysis")

        force_single_llm = job.requested_llm_mode == "single"
        force_comparison_llm = job.requested_llm_mode == "dual"
        orchestrator = GradingSessionOrchestrator(
            session_id=session_id,
            user_id=user_id,
            workflow_state=payload.get("workflow_state"),
            force_single_llm=force_single_llm,
            force_comparison_llm=force_comparison_llm,
            grading_mode=job.grading_method,
            batch_verify=job.batch_verify,
        )
        orchestrator.pdf_paths = get_session_copy_paths(session)

        if not orchestrator.pdf_paths:
            raise RuntimeError("Aucun PDF de copies disponible pour cette session.")

        await orchestrator.analyze_only()
        service.touch_job(job_id, stage="analysis_complete")

        confirmed_scale = payload.get("grading_scale") or dict(orchestrator.grading_scale)
        orchestrator.confirm_scale(confirmed_scale)
        service.touch_job(job_id, stage="correction")

        progress_callback = create_persistent_progress_callback(job_id)
        await orchestrator.grade_all(progress_callback=progress_callback)

        session = store.load_session()
        if not session:
            raise RuntimeError(f"Session {session_id} not found after grading")

        average_score = 0.0
        if session.graded_copies:
            scores = [graded.total_score for graded in session.graded_copies]
            average_score = sum(scores) / len(scores)

        token_result = {
            "tokens_deducted": 0,
            "remaining_tokens": 0,
            "usage_record_id": None,
            "is_duplicate": False,
        }
        db_user = None

        try:
            deduction_db = SessionLocal()
            deduction_service = TokenDeductionService()
            token_result = deduction_service.deduct_grading_usage(
                user_id=user_id,
                provider=orchestrator.ai,
                session_id=session_id,
                db=deduction_db,
            )
            db_user = deduction_db.query(User).filter(User.id == user_id).first()
        finally:
            deduction_db.close()

        session.status = SessionStatus.COMPLETE
        store.save_session(session)

        completion_payload = {
            "average_score": average_score,
            "total_copies": len(session.graded_copies),
            "tokens_used": token_result["tokens_deducted"],
            "remaining_tokens": db_user.remaining_tokens if db_user else 0,
        }

        event_db = SessionLocal()
        try:
            event_service = JobService(event_db)
            event_service.record_event(job_id, PROGRESS_EVENT_SESSION_COMPLETE, completion_payload)
            event_service.complete_job(
                job_id,
                {
                    "average_score": average_score,
                    "graded_count": len(session.graded_copies),
                    "tokens_used": token_result["tokens_deducted"],
                    "remaining_tokens": db_user.remaining_tokens if db_user else 0,
                },
            )
        finally:
            event_db.close()

    except InsufficientTokensError as exc:
        await _persist_failure(
            job_id,
            exc,
            f"Capacité insuffisante pour finaliser la correction ({exc.tokens_remaining} restants, {exc.tokens_required} requis).",
        )
    except (UserNotFoundError, DeductionError) as exc:
        await _persist_failure(job_id, exc, str(exc))
    except Exception as exc:
        await _persist_failure(job_id, exc, str(exc))
    finally:
        db.close()


async def _persist_failure(job_id: str, exc: Exception, message: str) -> None:
    """Persist a job failure and revert the session status to ERROR."""
    logger.error(f"Job {job_id} failed: {exc}")

    db = SessionLocal()
    try:
        service = JobService(db)
        job = service.get_job(job_id)
        session_id = job.session_id if job else None
        user_id = job.user_id if job else None

        if session_id and user_id:
            try:
                store = SessionStore(session_id, user_id=user_id)
                session = store.load_session()
                if session:
                    session.status = SessionStatus.ERROR
                    store.save_session(session)
            except Exception as session_error:
                logger.warning(f"Failed to persist ERROR status for session {session_id}: {session_error}")

        service.record_event(job_id, PROGRESS_EVENT_SESSION_ERROR, {"error": message})
        service.fail_job(job_id, message, result_payload={"error": message})
    finally:
        db.close()


async def run_worker_loop(
    *,
    worker_id: str | None = None,
    poll_interval: float = 2.0,
    stale_after_seconds: int = 7200,
    once: bool = False,
) -> int:
    """Continuously claim and process queued grading jobs."""
    effective_worker_id = worker_id or build_worker_id()
    logger.info(f"Starting grading worker {effective_worker_id}")

    processed_jobs = 0

    while True:
        db = SessionLocal()
        try:
            job = JobService(db).claim_next_job(
                worker_id=effective_worker_id,
                stale_after_seconds=stale_after_seconds,
            )
        finally:
            db.close()

        if not job:
            if once:
                return processed_jobs
            await asyncio.sleep(poll_interval)
            continue

        await run_grading_job(job.id)
        processed_jobs += 1

        if once:
            return processed_jobs
