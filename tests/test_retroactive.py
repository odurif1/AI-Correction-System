"""
Tests for retroactive application of grading changes.
"""

import pytest
import numpy as np

from src.core.models import (
    CopyDocument, GradedCopy, GradingSession,
    TeacherDecision, SimilarCopies
)
from src.calibration.retroactive import RetroactiveApplier


@pytest.fixture
def sample_session():
    """Create a sample grading session."""
    session = GradingSession(
        session_id="test_session",
        status="calibrating"
    )

    # Add copies with different embeddings
    np.random.seed(42)
    
    for i in range(5):
        copy = CopyDocument(
            id=f"copy{i}",
            pdf_path=f"/tmp/copy{i}.pdf",
            page_count=1,
            content_summary={"Q1": f"Answer {i}"}
        )
        # Each copy gets a unique embedding
        copy.embedding = np.random.rand(1536).tolist()
        session.copies.append(copy)

    # Add graded copies
    for i in range(5):
        graded = GradedCopy(
            copy_id=f"copy{i}",
            grades={"Q1": 4.0},
            total_score=4.0,
            max_score=5.0,
            confidence=0.8
        )
        session.graded_copies.append(graded)

    # Set up clusters (copies 0,1,2 are similar)
    session.clusters = {
        0: ["copy0", "copy1", "copy2"]
    }

    return session


def test_find_similar_by_cluster(sample_session):
    """Test finding similar copies via cluster membership."""
    from src.ai import create_ai_provider
    mock_ai = create_ai_provider()
    applier = RetroactiveApplier(sample_session, ai_provider=mock_ai)

    decision = TeacherDecision(
        question_id="Q1",
        source_copy_id="copy0",
        teacher_guidance="Give full credit for correct approach",
        original_score=4.0,
        new_score=5.0,
        applies_to_all=True
    )

    similar = applier.find_similar(decision)

    # Should find copies in same cluster (cluster 0 has copy0, copy1, copy2)
    # copy1 and copy2 should be in similar results
    assert "copy1" in similar.already_graded or "copy2" in similar.already_graded

    # Results should contain some similar copies
    assert len(similar.already_graded) > 0


def test_build_proposal(sample_session):
    """Test building a proposal message."""
    from src.ai import create_ai_provider
    mock_ai = create_ai_provider()
    applier = RetroactiveApplier(sample_session, ai_provider=mock_ai)

    decision = TeacherDecision(
        question_id="Q1",
        source_copy_id="copy0",
        teacher_guidance="Accept alternative method",
        original_score=3.0,
        new_score=5.0
    )

    similar = SimilarCopies(
        already_graded=["copy1", "copy2"],
        pending=["copy3", "copy4"]
    )

    proposal = applier._build_proposal(decision, similar)

    assert "Q1" in proposal
    assert "copy0" in proposal
    assert "Accept alternative method" in proposal
    assert "2 already graded" in proposal
    assert "2 pending" in proposal


def test_similarity_calculation():
    """Test cosine similarity calculation."""
    from src.analysis.clustering import EmbeddingClustering

    clustering = EmbeddingClustering()

    # Identical vectors
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    assert clustering._cosine_similarity(vec1, vec2) == 1.0

    # Orthogonal vectors
    vec3 = [1.0, 0.0, 0.0]
    vec4 = [0.0, 1.0, 0.0]
    assert clustering._cosine_similarity(vec3, vec4) == 0.0

    # Similar vectors
    vec5 = [1.0, 1.0, 0.0]
    vec6 = [1.0, 0.9, 0.0]
    sim = clustering._cosine_similarity(vec5, vec6)
    assert sim > 0.9
