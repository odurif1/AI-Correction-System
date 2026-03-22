"""Tests for persistent grading job queue semantics."""

from datetime import timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.database import Base
from db.models import SubscriptionTier, User, SessionJobStatus
from services.job_service import (
    JobConflictError,
    JobService,
    PROGRESS_EVENT_COPY_DONE,
    PROGRESS_EVENT_SESSION_COMPLETE,
    build_job_progress_snapshot,
    utcnow,
)


def make_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()

    user = User(
        id="user-1",
        email="user-1@example.com",
        password_hash="disabled",
        subscription_tier=SubscriptionTier.FREE,
    )
    db.add(user)
    db.commit()
    return db


def test_enqueue_grading_job_creates_queued_job():
    db = make_db()
    service = JobService(db)

    job = service.enqueue_grading_job(
        session_id="session-1",
        user_id="user-1",
        requested_llm_mode="dual",
        grading_method="batch",
        batch_verify="per-copy",
        payload={"grading_scale": {"Q1": 2.0}},
    )

    assert job.status == SessionJobStatus.QUEUED
    assert job.payload == {"grading_scale": {"Q1": 2.0}}
    assert service.get_latest_session_job("session-1", user_id="user-1").id == job.id


def test_enqueue_rejects_second_active_job_for_same_session():
    db = make_db()
    service = JobService(db)

    service.enqueue_grading_job(
        session_id="session-1",
        user_id="user-1",
        requested_llm_mode="dual",
        grading_method="batch",
        batch_verify="per-copy",
    )

    try:
        service.enqueue_grading_job(
            session_id="session-1",
            user_id="user-1",
            requested_llm_mode="single",
            grading_method="individual",
            batch_verify="per-question",
        )
        assert False, "Expected JobConflictError"
    except JobConflictError:
        pass


def test_claim_next_job_marks_job_running():
    db = make_db()
    service = JobService(db)
    created = service.enqueue_grading_job(
        session_id="session-1",
        user_id="user-1",
        requested_llm_mode="dual",
        grading_method="batch",
        batch_verify="per-copy",
    )

    claimed = service.claim_next_job(worker_id="worker-a")

    assert claimed is not None
    assert claimed.id == created.id
    assert claimed.status == SessionJobStatus.RUNNING
    assert claimed.worker_id == "worker-a"


def test_record_event_updates_copy_progress_and_sequences():
    db = make_db()
    service = JobService(db)
    job = service.enqueue_grading_job(
        session_id="session-1",
        user_id="user-1",
        requested_llm_mode="dual",
        grading_method="batch",
        batch_verify="per-copy",
    )
    service.claim_next_job(worker_id="worker-a")

    event = service.record_event(
        job.id,
        PROGRESS_EVENT_COPY_DONE,
        {"copy_index": 2, "total_copies": 5, "copy_id": "copy-2"},
    )

    refreshed = service.get_job(job.id)
    assert event.sequence == 1
    assert refreshed.completed_copies == 2
    assert refreshed.total_copies == 5
    assert service.list_events(job.id, after_sequence=0)[0].id == event.id


def test_complete_job_sets_result_payload_and_progress_snapshot():
    db = make_db()
    service = JobService(db)
    job = service.enqueue_grading_job(
        session_id="session-1",
        user_id="user-1",
        requested_llm_mode="dual",
        grading_method="batch",
        batch_verify="per-copy",
    )
    service.claim_next_job(worker_id="worker-a")
    service.record_event(job.id, PROGRESS_EVENT_SESSION_COMPLETE, {"total_copies": 3})

    completed = service.complete_job(job.id, {"graded_count": 3})

    snapshot = build_job_progress_snapshot(
        session_status="correction",
        copies_uploaded=3,
        grading_mode="dual",
        job=completed,
    )

    assert completed.status == SessionJobStatus.COMPLETED
    assert completed.result_payload["graded_count"] == 3
    assert snapshot["status"] == "complete"
    assert snapshot["copies_graded"] == 3


def test_fail_stale_running_jobs_marks_old_job_failed():
    db = make_db()
    service = JobService(db)
    job = service.enqueue_grading_job(
        session_id="session-1",
        user_id="user-1",
        requested_llm_mode="dual",
        grading_method="batch",
        batch_verify="per-copy",
    )
    running = service.claim_next_job(worker_id="worker-a")
    running.updated_at = utcnow() - timedelta(seconds=4000)
    db.commit()

    failed_count = service.fail_stale_running_jobs(stale_after_seconds=60)

    refreshed = service.get_job(job.id)
    assert failed_count == 1
    assert refreshed.status == SessionJobStatus.FAILED
    assert refreshed.error_message == "Worker heartbeat expired"
