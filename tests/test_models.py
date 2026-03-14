"""
Tests for core models and stable architectural invariants.
"""

import pytest
from datetime import datetime

from src.core.models import (
    CopyDocument, GradedCopy, GradingPolicy, GradingSession,
    TeacherDecision, SessionDocument, DocumentDecision, DocumentType, generate_id
)
from src.core.services.detection_service import DetectionService
from src.ai.base_provider import BaseProvider
from src.api.app import estimate_session_token_budget
from src.config.settings import Settings
from src.services.token_service import TokenDeductionService
from db.models import User, SubscriptionTier


class DummyProvider(BaseProvider):
    def call_vision(self, *args, **kwargs):
        return ""

    def call_text(self, *args, **kwargs):
        return ""

    def get_embedding(self, text: str):
        return []

    def get_embeddings(self, texts):
        return [[] for _ in texts]


def test_generate_id():
    """Test ID generation."""
    id1 = generate_id()
    id2 = generate_id()

    assert id1 != id2
    assert len(id1) == 36


def test_copy_document():
    """Test CopyDocument model."""
    copy = CopyDocument(
        pdf_path="/path/to/test.pdf",
        page_count=3,
        student_name="Test Student",
        content_summary={
            "Q1": "The student answered correctly",
            "Q2": "Partial answer with some errors"
        }
    )

    assert copy.id is not None
    assert copy.pdf_path == "/path/to/test.pdf"
    assert copy.page_count == 3
    assert copy.student_name == "Test Student"
    assert "Q1" in copy.content_summary
    assert not copy.processed


def test_grading_policy():
    """Test GradingPolicy model."""
    policy = GradingPolicy(
        version=1,
        criteria={
            "Q1": "Grade based on correctness",
            "Q2": "Grade based on methodology"
        },
        question_weights={
            "Q1": 5.0,
            "Q2": 10.0
        }
    )

    assert policy.version == 1
    assert len(policy.criteria) == 2
    assert policy.question_weights["Q2"] == 10.0
    assert policy.confidence_thresholds["auto"] == 0.85


def test_graded_copy():
    """Test GradedCopy model."""
    graded = GradedCopy(
        copy_id="test_copy",
        grades={
            "Q1": 5.0,
            "Q2": 8.0
        },
        total_score=13.0,
        max_score=20.0,
        confidence=0.9,
        rationale={
            "Q1": "Correct answer with good explanation",
            "Q2": "Method is correct but minor calculation error"
        }
    )

    assert graded.copy_id == "test_copy"
    assert graded.total_score == 13.0
    assert graded.max_score == 20.0
    assert graded.confidence == 0.9
    assert not graded.teacher_reviewed
    assert len(graded.adjustments) == 0


def test_teacher_decision():
    """Test TeacherDecision model."""
    decision = TeacherDecision(
        question_id="Q1",
        source_copy_id="copy1",
        teacher_guidance="The student's method is correct, give full credit",
        applies_to_all=True
    )

    assert decision.question_id == "Q1"
    assert decision.source_copy_id == "copy1"
    assert decision.applies_to_all is True
    assert decision.extracted_rule is None


def test_grading_session():
    """Test GradingSession model."""
    session = GradingSession(
        session_id="test_session"
    )

    assert session.session_id == "test_session"
    assert session.status == "diagnostic"
    assert session.copies_processed == 0
    assert len(session.copies) == 0
    assert len(session.graded_copies) == 0


def test_provider_usage_reports_billable_tokens_excluding_cached_tokens():
    provider = DummyProvider(mock_mode=True)

    provider._log_call(
        prompt_type="text",
        input_summary="test",
        response_summary="ok",
        duration_ms=1.0,
        prompt_tokens=1000,
        completion_tokens=200,
        cached_tokens=400,
    )

    usage = provider.get_token_usage()

    assert usage["prompt_tokens"] == 1000
    assert usage["completion_tokens"] == 200
    assert usage["cached_tokens"] == 400
    assert usage["total_tokens"] == 1200
    assert usage["billable_prompt_tokens"] == 600
    assert usage["billable_total_tokens"] == 800


def test_estimate_session_token_budget_uses_copy_document_pages():
    session = GradingSession(session_id="budget-test")
    session.documents = [
        SessionDocument(
            filename="copies-a.pdf",
            storage_path="/tmp/a.pdf",
            page_count=3,
            detected_type=DocumentType.STUDENT_COPIES,
            user_decision=DocumentDecision.STUDENT_COPIES,
            usable=True,
        ),
        SessionDocument(
            filename="copies-b.pdf",
            storage_path="/tmp/b.pdf",
            page_count=2,
            detected_type=DocumentType.STUDENT_COPIES,
            user_decision=DocumentDecision.STUDENT_COPIES,
            usable=True,
        ),
    ]

    assert estimate_session_token_budget(session) == 50_000


def test_token_deduction_uses_billable_tokens_when_cache_is_present():
    class ProviderStub:
        def get_token_usage(self):
            return {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "cached_tokens": 400,
                "total_tokens": 1200,
                "billable_total_tokens": 800,
            }

    user = User(
        id="user-1",
        email="prof@example.com",
        password_hash="hash",
        subscription_tier=SubscriptionTier.ESSENTIEL,
        tokens_used_this_month=0,
    )

    class QueryStub:
        def __init__(self, result):
            self.result = result

        def filter(self, *args, **kwargs):
            return self

        def with_for_update(self):
            return self

        def first(self):
            return self.result

    class DBStub:
        def __init__(self, user_obj):
            self.user = user_obj
            self.records = []
            self.committed = False

        def query(self, model):
            if model.__name__ == "UsageRecord":
                return QueryStub(None)
            if model.__name__ == "User":
                return QueryStub(self.user)
            raise AssertionError(f"Unexpected model: {model}")

        def add(self, record):
            self.records.append(record)

        def commit(self):
            self.committed = True

        def refresh(self, _obj):
            return None

        def rollback(self):
            return None

    db = DBStub(user)
    result = TokenDeductionService().deduct_grading_usage(
        user_id="user-1",
        provider=ProviderStub(),
        session_id="session-1",
        db=db,
    )

    assert result["tokens_deducted"] == 800
    assert user.tokens_used_this_month == 800
    assert len(db.records) == 1
    assert db.records[0].total_tokens == 800


@pytest.mark.asyncio
async def test_detection_service_only_exposes_generic_question_labels(tmp_path):
    """
    Detection must not invent or persist a session-wide question text.

    It may expose stable technical labels derived from detected question ids,
    while the detailed question text remains local to a copy/LLM path.
    """
    session = GradingSession(session_id="test_session")
    session.copies.append(
        CopyDocument(
            pdf_path=str(tmp_path / "copy.pdf"),
            page_count=1,
            content_summary={
                "Q1": "Resume local de la reponse eleve",
                "Q2": "Autre resume",
                "free_text": "ignored",
                "Q1_points": "ignored",
            },
        )
    )

    class StoreStub:
        session_dir = tmp_path

        def load_detection(self):
            return None

        def save_copy(self, copy, pdf_bytes):
            return None

        def cleanup_temp_images(self):
            return None

    service = DetectionService(
        session=session,
        store=StoreStub(),
        ai=None,
        pdf_paths=[],
    )

    result = await service.analyze_only()

    assert result["questions"] == {
        "Q1": "Question Q1",
        "Q2": "Question Q2",
    }
    assert "question_text" not in result
