"""
Tests for consistency checking and calibration.
"""

import pytest
import numpy as np

from src.core.models import (
    GradingSession, GradedCopy, CopyDocument,
    ClassAnswerMap, QuestionAnalysis
)
from src.calibration.consistency import (
    ConsistencyDetector, CalibrationReport
)


@pytest.fixture
def session_with_inconsistency():
    """Create a session with grading inconsistencies."""
    session = GradingSession(
        session_id="test_session",
        status="calibrating"
    )

    # Add copies
    for i in range(5):
        copy = CopyDocument(
            id=f"copy{i}",
            pdf_path=f"/tmp/copy{i}.pdf",
            page_count=1,
            content_summary={
                "Q1": "Similar answer"
            }
        )
        session.copies.append(copy)

    # Add graded copies with inconsistent grades for same answer
    # Copies 0, 1, 2 should have same grade but don't
    graded_data = [
        ("copy0", 5.0),  # High
        ("copy1", 2.0),  # Low - inconsistent!
        ("copy2", 3.0),  # Medium - inconsistent!
        ("copy3", 4.0),
        ("copy4", 4.0)
    ]

    for copy_id, score in graded_data:
        graded = GradedCopy(
            copy_id=copy_id,
            grades={"Q1": score},
            total_score=score,
            max_score=5.0,
            confidence=0.8
        )
        session.graded_copies.append(graded)

    # Add class map with cluster
    session.class_map = ClassAnswerMap()
    session.class_map.clusters = {
        "Q1": [
            {
                "cluster_id": 0,
                "question_id": "Q1",
                "copy_ids": ["copy0", "copy1", "copy2"],
                "representative_description": "Similar answers"
            }
        ]
    }

    return session


def test_detect_inconsistencies(session_with_inconsistency):
    """Test inconsistency detection."""
    detector = ConsistencyDetector(session_with_inconsistency)
    inconsistencies = detector.detect_all()

    # Should detect inconsistency in the cluster
    assert len(inconsistencies) > 0

    inc = inconsistencies[0]
    assert inc.question_id == "Q1"
    assert inc.max_difference > 0


def test_consistency_report(session_with_inconsistency):
    """Test calibration report generation."""
    report_gen = CalibrationReport(session_with_inconsistency)
    report = report_gen.generate()

    assert report["session_id"] == "test_session"
    assert report["total_copies"] == 5
    assert report["inconsistencies_found"] > 0
    assert len(report["inconsistencies"]) > 0
    assert "score_statistics" in report


def test_propose_corrections(session_with_inconsistency):
    """Test correction proposals."""
    report_gen = CalibrationReport(session_with_inconsistency)
    proposals = report_gen.propose_corrections()

    assert len(proposals) > 0

    prop = proposals[0]
    assert "question_id" in prop
    assert "proposed_grade" in prop
    assert "current_grades" in prop


def test_score_statistics(session_with_inconsistency):
    """Test score statistics calculation."""
    report_gen = CalibrationReport(session_with_inconsistency)
    report = report_gen.generate()

    stats = report["score_statistics"]
    assert "mean" in stats
    assert "median" in stats
    assert "std" in stats
    assert "min" in stats
    assert "max" in stats

    # Verify values
    expected_mean = (5.0 + 2.0 + 3.0 + 4.0 + 4.0) / 5
    assert abs(stats["mean"] - expected_mean) < 0.01


def test_empty_session():
    """Test handling of empty session."""
    session = GradingSession(
        session_id="empty_session",
        status="calibrating"
    )

    detector = ConsistencyDetector(session)
    inconsistencies = detector.detect_all()

    assert len(inconsistencies) == 0


def test_recommendations(session_with_inconsistency):
    """Test recommendation generation."""
    report_gen = CalibrationReport(session_with_inconsistency)
    report = report_gen.generate()

    recommendations = report["recommendations"]
    assert len(recommendations) > 0
