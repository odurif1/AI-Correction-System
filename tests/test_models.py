"""
Tests for core models.
"""

import pytest
from datetime import datetime

from src.core.models import (
    CopyDocument, GradedCopy, GradingPolicy, GradingSession,
    TeacherDecision, generate_id
)


def test_generate_id():
    """Test ID generation."""
    id1 = generate_id()
    id2 = generate_id()

    assert id1 != id2
    assert len(id1) == 8


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
