import json

import pytest

from analysis.detection import Detector
from core.models import ClassAnswerMap, CopyDocument, GradingSession, SessionStatus
from core.services.detection_service import DetectionService


def make_detector(tmp_path):
    return Detector(
        user_id="user-1",
        session_id="session-1",
        cache_dir=tmp_path / "cache",
        provider=object(),
    )


def test_detection_normalizes_numeric_question_ids(tmp_path):
    detector = make_detector(tmp_path)
    response = json.dumps(
        {
            "document_type": "student_copies",
            "confidence_document_type": 0.9,
            "structure": "one_pdf_all_students",
            "subject_integration": "integrated",
            "num_students_detected": 1,
            "students": [],
            "grading_scale": {
                "1": "1 pt",
                "Question 2": 2,
                "Q03": "0.5",
            },
            "confidence_grading_scale": 0.9,
            "questions_detected": ["1", "Question 2", "Q03"],
            "quality_issues": [],
            "blocking_issues": [],
            "warnings": [],
            "detected_language": "fr",
        }
    )

    result = detector._parse_response(response, page_count=3, mode="interactive")

    assert result.grading_scale == {"Q1": 1.0, "Q2": 2.0, "Q3": 0.5}
    assert result.questions_detected == ["Q1", "Q2", "Q3"]


def test_detection_rejects_generic_grading_labels(tmp_path):
    detector = make_detector(tmp_path)
    response = json.dumps(
        {
            "document_type": "student_copies",
            "confidence_document_type": 0.9,
            "structure": "one_pdf_all_students",
            "subject_integration": "integrated",
            "num_students_detected": 1,
            "students": [],
            "grading_scale": {
                "Exercice": 4,
                "Q1": 2,
                "Q2": 2,
            },
            "confidence_grading_scale": 0.9,
            "questions_detected": ["Exercice", "Q1", "Q2"],
            "quality_issues": [],
            "blocking_issues": [],
            "warnings": [],
            "detected_language": "fr",
        }
    )

    result = detector._parse_response(response, page_count=3, mode="interactive")

    assert result.grading_scale == {"Q1": 2.0, "Q2": 2.0}
    assert result.questions_detected == ["Q1", "Q2"]
    assert any("trop vague" in warning.lower() for warning in result.warnings)
    assert result.confidence_grading_scale < 0.9


def test_detection_keeps_richer_labels_for_exam_diversity(tmp_path):
    detector = make_detector(tmp_path)
    response = json.dumps(
        {
            "document_type": "student_copies",
            "confidence_document_type": 0.9,
            "structure": "one_pdf_all_students",
            "subject_integration": "integrated",
            "num_students_detected": 1,
            "students": [],
            "grading_scale": {
                "Exercice 1 - Analyse de document": 5,
                "Exercice 2 - Rédaction": 5,
            },
            "confidence_grading_scale": 0.85,
            "questions_detected": ["Exercice 1 - Analyse de document", "Exercice 2 - Rédaction"],
            "quality_issues": [],
            "blocking_issues": [],
            "warnings": [],
            "detected_language": "fr",
        }
    )

    result = detector._parse_response(response, page_count=3, mode="interactive")

    assert result.grading_scale == {
        "Exercice 1 - Analyse de document": 5.0,
        "Exercice 2 - Rédaction": 5.0,
    }
    assert result.questions_detected == [
        "Exercice 1 - Analyse de document",
        "Exercice 2 - Rédaction",
    ]


def test_detection_applies_progressive_confidence_penalty(tmp_path):
    detector = make_detector(tmp_path)
    response = json.dumps(
        {
            "document_type": "student_copies",
            "confidence_document_type": 0.9,
            "structure": "one_pdf_all_students",
            "subject_integration": "integrated",
            "num_students_detected": 1,
            "students": [],
            "grading_scale": {
                "Exercice": 4,
                "Question 1": 2,
                "Q1": 3,
            },
            "confidence_grading_scale": 0.9,
            "questions_detected": ["Exercice", "Question 1"],
            "quality_issues": [],
            "blocking_issues": [],
            "warnings": [],
            "detected_language": "fr",
        }
    )

    result = detector._parse_response(response, page_count=3, mode="interactive")

    assert result.confidence_grading_scale == 0.7
    assert len(result.warnings) >= 2


def test_detection_translates_quality_issues(tmp_path):
    detector = make_detector(tmp_path)
    response = json.dumps(
        {
            "document_type": "student_copies",
            "confidence_document_type": 0.9,
            "structure": "one_pdf_all_students",
            "subject_integration": "integrated",
            "num_students_detected": 1,
            "students": [],
            "grading_scale": {"Q1": 1},
            "confidence_grading_scale": 0.9,
            "questions_detected": ["Q1"],
            "quality_issues": ["Handwriting difficult to read in some places."],
            "blocking_issues": [],
            "warnings": [],
            "detected_language": "fr",
        }
    )

    result = detector._parse_response(response, page_count=1, mode="interactive")

    assert result.quality_issues == ["Écriture difficile à lire par endroits."]


def test_detection_translates_warnings_and_blocking_issues(tmp_path):
    detector = make_detector(tmp_path)
    response = json.dumps(
        {
            "document_type": "student_copies",
            "confidence_document_type": 0.9,
            "structure": "ambiguous",
            "subject_integration": "not_detected",
            "num_students_detected": 1,
            "students": [],
            "grading_scale": {"Q1": 1},
            "confidence_grading_scale": 0.9,
            "questions_detected": ["Q1"],
            "quality_issues": [],
            "blocking_issues": ["Cannot determine document structure"],
            "warnings": ["The grading scale may be incomplete"],
            "detected_language": "fr",
        }
    )

    result = detector._parse_response(response, page_count=1, mode="interactive")

    assert result.blocking_issues == ["Impossible de déterminer la structure du document"]
    assert result.warnings == ["Le barème détecté semble incomplet"]


@pytest.mark.anyio
async def test_detection_service_does_not_invent_default_scale():
    session = GradingSession()
    session.copies = [
        CopyDocument(
            pdf_path="/tmp/copy.pdf",
            page_count=1,
            content_summary={"Q1": "Réponse"},
            language="fr",
        )
    ]

    class DummyStore:
        def load_detection(self):
            return None

    service = DetectionService(
        session=session,
        store=DummyStore(),
        ai=None,
        pdf_paths=[],
    )

    async def fake_load():
        return None

    service._load_copies_phase = fake_load

    result = await service.analyze_only()

    assert result["questions"] == {"Q1": "Question Q1"}
    assert result["scale"] == {}
    assert result["scale_detected"] is False


@pytest.mark.anyio
async def test_detection_service_analyze_only_is_safe_when_session_already_in_correction():
    session = GradingSession(status=SessionStatus.CORRECTION)
    session.copies = [
        CopyDocument(
            pdf_path="/tmp/copy.pdf",
            page_count=1,
            content_summary={"Q1": "Réponse"},
            language="fr",
        )
    ]

    class DummyStore:
        def load_detection(self):
            return None

    service = DetectionService(
        session=session,
        store=DummyStore(),
        ai=None,
        pdf_paths=[],
    )

    async def fake_load():
        return None

    service._load_copies_phase = fake_load

    result = await service.analyze_only()

    assert session.status == SessionStatus.CORRECTION
    assert isinstance(session.class_map, ClassAnswerMap)
    assert result["questions"] == {"Q1": "Question Q1"}
