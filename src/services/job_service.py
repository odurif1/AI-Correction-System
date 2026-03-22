"""Persistent session job queue and progress tracking."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from loguru import logger
from sqlalchemy.orm import Session

from db.models import (
    SessionJob,
    SessionJobEvent,
    SessionJobStatus,
    SessionJobType,
)

PROGRESS_EVENT_COPY_START = "copy_start"
PROGRESS_EVENT_QUESTION_DONE = "question_done"
PROGRESS_EVENT_COPY_DONE = "copy_done"
PROGRESS_EVENT_COPY_ERROR = "copy_error"
PROGRESS_EVENT_SESSION_COMPLETE = "session_complete"
PROGRESS_EVENT_SESSION_ERROR = "session_error"

ACTIVE_JOB_STATUSES = (
    SessionJobStatus.QUEUED,
    SessionJobStatus.RUNNING,
)


def utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class JobConflictError(RuntimeError):
    """Raised when a session already has an active grading job."""


class JobNotFoundError(RuntimeError):
    """Raised when a requested job cannot be found."""


class JobService:
    """High-level API for queued grading jobs and their progress events."""

    def __init__(self, db: Session):
        self.db = db

    def get_job(self, job_id: str) -> Optional[SessionJob]:
        """Load a single job by identifier."""
        return self.db.query(SessionJob).filter(SessionJob.id == job_id).first()

    def get_latest_session_job(
        self,
        session_id: str,
        user_id: str | None = None,
    ) -> Optional[SessionJob]:
        """Return the latest persisted job for a session."""
        query = self.db.query(SessionJob).filter(SessionJob.session_id == session_id)
        if user_id:
            query = query.filter(SessionJob.user_id == user_id)
        return query.order_by(SessionJob.created_at.desc()).first()

    def get_active_session_job(
        self,
        session_id: str,
        user_id: str,
    ) -> Optional[SessionJob]:
        """Return the queued or running job for a session, if any."""
        return (
            self.db.query(SessionJob)
            .filter(
                SessionJob.session_id == session_id,
                SessionJob.user_id == user_id,
                SessionJob.status.in_(ACTIVE_JOB_STATUSES),
            )
            .order_by(SessionJob.created_at.desc())
            .first()
        )

    def enqueue_grading_job(
        self,
        *,
        session_id: str,
        user_id: str,
        requested_llm_mode: str,
        grading_method: str,
        batch_verify: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> SessionJob:
        """Create a new persistent grading job for a session."""
        active = self.get_active_session_job(session_id, user_id)
        if active:
            raise JobConflictError(
                f"Session {session_id} already has an active job ({active.id})"
            )

        job = SessionJob(
            session_id=session_id,
            user_id=user_id,
            job_type=SessionJobType.GRADE_SESSION,
            status=SessionJobStatus.QUEUED,
            stage="queued",
            requested_llm_mode=requested_llm_mode,
            grading_method=grading_method,
            batch_verify=batch_verify,
            payload=payload or {},
            result_payload={},
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        logger.info(f"Queued grading job {job.id} for session {session_id}")
        return job

    def claim_next_job(
        self,
        *,
        worker_id: str,
        stale_after_seconds: int = 7200,
    ) -> Optional[SessionJob]:
        """Claim the next queued job for a worker."""
        self.fail_stale_running_jobs(stale_after_seconds=stale_after_seconds)

        queued_jobs = (
            self.db.query(SessionJob)
            .filter(SessionJob.status == SessionJobStatus.QUEUED)
            .order_by(SessionJob.created_at.asc())
            .all()
        )

        for queued in queued_jobs:
            updated = (
                self.db.query(SessionJob)
                .filter(
                    SessionJob.id == queued.id,
                    SessionJob.status == SessionJobStatus.QUEUED,
                )
                .update(
                    {
                        SessionJob.status: SessionJobStatus.RUNNING,
                        SessionJob.stage: "starting",
                        SessionJob.worker_id: worker_id,
                        SessionJob.started_at: utcnow(),
                        SessionJob.updated_at: utcnow(),
                    },
                    synchronize_session=False,
                )
            )
            if updated:
                self.db.commit()
                job = self.get_job(queued.id)
                logger.info(f"Worker {worker_id} claimed job {queued.id}")
                return job

        return None

    def fail_stale_running_jobs(self, *, stale_after_seconds: int) -> int:
        """Fail running jobs that have not emitted progress within the timeout."""
        if stale_after_seconds <= 0:
            return 0

        stale_before = utcnow() - timedelta(seconds=stale_after_seconds)
        stale_jobs = (
            self.db.query(SessionJob)
            .filter(
                SessionJob.status == SessionJobStatus.RUNNING,
                SessionJob.updated_at < stale_before,
            )
            .all()
        )

        failed = 0
        for job in stale_jobs:
            job.status = SessionJobStatus.FAILED
            job.stage = "error"
            job.error_message = "Worker heartbeat expired"
            job.finished_at = utcnow()
            job.updated_at = utcnow()
            failed += 1

        if failed:
            self.db.commit()
            logger.warning(f"Marked {failed} stale running jobs as failed")

        return failed

    def touch_job(self, job_id: str, *, stage: str | None = None) -> SessionJob:
        """Refresh a job heartbeat and optionally update its stage."""
        job = self._require_job(job_id)
        if stage:
            job.stage = stage
        job.updated_at = utcnow()
        self.db.commit()
        self.db.refresh(job)
        return job

    def complete_job(self, job_id: str, result_payload: Optional[dict[str, Any]] = None) -> SessionJob:
        """Mark a job as completed."""
        job = self._require_job(job_id)
        job.status = SessionJobStatus.COMPLETED
        job.stage = "complete"
        job.result_payload = result_payload or {}
        job.finished_at = utcnow()
        job.updated_at = utcnow()
        self.db.commit()
        self.db.refresh(job)
        return job

    def fail_job(
        self,
        job_id: str,
        error_message: str,
        *,
        result_payload: Optional[dict[str, Any]] = None,
    ) -> SessionJob:
        """Mark a job as failed."""
        job = self._require_job(job_id)
        job.status = SessionJobStatus.FAILED
        job.stage = "error"
        job.error_message = error_message
        job.result_payload = result_payload or {}
        job.finished_at = utcnow()
        job.updated_at = utcnow()
        self.db.commit()
        self.db.refresh(job)
        return job

    def cancel_session_jobs(self, session_id: str, user_id: str) -> int:
        """Cancel and remove persisted jobs for a deleted session."""
        jobs = (
            self.db.query(SessionJob)
            .filter(
                SessionJob.session_id == session_id,
                SessionJob.user_id == user_id,
            )
            .all()
        )

        if not jobs:
            return 0

        count = len(jobs)
        for job in jobs:
            self.db.delete(job)
        self.db.commit()
        return count

    def list_events(self, job_id: str, *, after_sequence: int = 0) -> list[SessionJobEvent]:
        """Return ordered events for a job after a given sequence number."""
        return (
            self.db.query(SessionJobEvent)
            .filter(
                SessionJobEvent.job_id == job_id,
                SessionJobEvent.sequence > after_sequence,
            )
            .order_by(SessionJobEvent.sequence.asc())
            .all()
        )

    def record_event(self, job_id: str, event_type: str, payload: Optional[dict[str, Any]] = None) -> SessionJobEvent:
        """Persist a job event and update aggregate progress counters."""
        job = self._require_job(job_id)
        event_payload = payload or {}
        sequence = (job.last_event_sequence or 0) + 1

        self._apply_progress_from_event(job, event_type, event_payload)
        job.last_event_sequence = sequence
        job.updated_at = utcnow()

        event = SessionJobEvent(
            job_id=job.id,
            session_id=job.session_id,
            sequence=sequence,
            event_type=event_type,
            payload=event_payload,
        )
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    def _apply_progress_from_event(
        self,
        job: SessionJob,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Update aggregate counters stored on the job row."""
        total_copies = payload.get("total_copies")
        if isinstance(total_copies, int) and total_copies > 0:
            job.total_copies = total_copies

        if event_type == PROGRESS_EVENT_COPY_START:
            job.stage = "correction"
            return

        if event_type == PROGRESS_EVENT_COPY_DONE:
            job.stage = "correction"
            copy_index = payload.get("copy_index")
            if isinstance(copy_index, int) and copy_index > 0:
                job.completed_copies = max(job.completed_copies, copy_index)
            else:
                job.completed_copies += 1
            return

        if event_type == PROGRESS_EVENT_COPY_ERROR:
            job.stage = "correction"
            copy_index = payload.get("copy_index")
            if isinstance(copy_index, int) and copy_index > 0:
                job.completed_copies = max(job.completed_copies, copy_index)
            return

        if event_type == PROGRESS_EVENT_SESSION_COMPLETE:
            job.stage = "complete"
            if job.total_copies and job.completed_copies < job.total_copies:
                job.completed_copies = job.total_copies
            return

        if event_type == PROGRESS_EVENT_SESSION_ERROR:
            job.stage = "error"
            if payload.get("error"):
                job.error_message = str(payload["error"])
            return

        if event_type == "analysis_complete":
            job.stage = "analysis_complete"
            return

        if event_type in {"single_pass_start", "single_pass_complete", "verification_start"}:
            job.stage = "correction"

    def _require_job(self, job_id: str) -> SessionJob:
        job = self.get_job(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")
        return job


def serialize_job_event(event: SessionJobEvent) -> dict[str, Any]:
    """Convert a persisted event row into a WebSocket-friendly payload."""
    payload = dict(event.payload or {})
    payload["type"] = event.event_type
    payload["sequence"] = event.sequence
    payload["timestamp"] = event.created_at.isoformat() if event.created_at else None
    return payload


def build_job_progress_snapshot(
    *,
    session_status: str,
    copies_uploaded: int,
    grading_mode: Optional[str],
    job: Optional[SessionJob],
) -> dict[str, Any]:
    """Build the persisted progress state sent to WebSocket clients."""
    status_value = session_status
    error = None
    job_id = None
    job_status = None
    copies_graded = 0

    if job:
        job_id = job.id
        job_status = job.status.value
        copies_graded = job.completed_copies or 0
        if job.error_message:
            error = job.error_message
        if job.status == SessionJobStatus.COMPLETED:
            status_value = "complete"
        elif job.status == SessionJobStatus.FAILED:
            status_value = "error"
        elif job.status in ACTIVE_JOB_STATUSES:
            status_value = "correction"

    return {
        "status": status_value,
        "job_id": job_id,
        "job_status": job_status,
        "copies_uploaded": copies_uploaded,
        "copies_graded": copies_graded,
        "grading_mode": grading_mode or "dual",
        "error": error,
    }
