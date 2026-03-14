from pathlib import Path

import fitz

from src.core.models import CopyDocument, GradedCopy
from src.export.annotation_service import (
    AnnotationCoordinateDetector,
    AnnotationPlacement,
    CopyAnnotations,
    create_annotation_boxes,
)
from src.export.annotation_pipeline import AnnotationExportService
from src.export.pdf_annotator import PDFAnnotator


def create_pdf(path: Path, pages: list[str]) -> None:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_annotate_copy_prepends_cover_and_keeps_annotations_on_original_pages(tmp_path):
    pdf_path = tmp_path / "copy.pdf"
    output_path = tmp_path / "annotated.pdf"
    create_pdf(pdf_path, ["Original page 1", "Original page 2"])

    copy = CopyDocument(
        id="copy-1",
        pdf_path=str(pdf_path),
        student_name="Alice",
    )
    graded = GradedCopy(
        copy_id="copy-1",
        grades={"Q1": 4.0},
        student_feedback={"Q1": "Bonne methode, attention au resultat final."},
        total_score=4.0,
        max_score=5.0,
    )

    annotator = PDFAnnotator()
    annotator.annotate_copy(
        copy=copy,
        graded=graded,
        output_path=str(output_path),
        smart_placement=False,
    )

    doc = fitz.open(str(output_path))
    try:
        assert len(doc) == 3
        assert "Graded Assessment - Alice" in doc[0].get_text()
        assert "Original page 1" in doc[1].get_text()
        assert "Page 1" in doc[1].get_text()
        assert "Original page 2" in doc[2].get_text()
        assert "Page 2" in doc[2].get_text()
    finally:
        doc.close()


def test_create_annotation_boxes_supports_page_offset(tmp_path):
    pdf_path = tmp_path / "base.pdf"
    create_pdf(pdf_path, ["Cover", "Student page"])

    doc = fitz.open(str(pdf_path))
    try:
        annotations = CopyAnnotations(
            copy_id="copy-1",
            student_name="Alice",
            placements=[
                AnnotationPlacement(
                    question_id="Q1",
                    feedback_text="Feedback",
                    page_number=1,
                    x_percent=10,
                    y_percent=20,
                )
            ],
        )

        boxes = create_annotation_boxes(annotations, doc, page_number_offset=1)

        assert 1 in boxes
        assert 0 not in boxes
        assert boxes[1][0][1] == "Feedback"
    finally:
        doc.close()


def test_heuristic_annotations_spread_feedback_on_same_page(tmp_path):
    pdf_path = tmp_path / "copy.pdf"
    create_pdf(pdf_path, ["Page 1", "Page 2"])

    copy = CopyDocument(
        id="copy-1",
        pdf_path=str(pdf_path),
        student_name="Alice",
    )
    graded = GradedCopy(
        copy_id="copy-1",
        student_feedback={
            "Q1": "f1",
            "Q2": "f2",
            "Q3": "f3",
        },
    )

    placements = PDFAnnotator().prepare_annotations(
        copy=copy,
        graded=graded,
        smart_placement=False,
    ).placements

    assert [placement.page_number for placement in placements] == [1, 1, 2]
    assert placements[0].y_percent < placements[1].y_percent


def test_annotation_export_service_reuses_single_annotation_computation(tmp_path, monkeypatch):
    pdf_path = tmp_path / "copy.pdf"
    create_pdf(pdf_path, ["Page 1"])

    copy = CopyDocument(
        id="copy-1",
        pdf_path=str(pdf_path),
        student_name="Alice",
    )
    graded = GradedCopy(
        copy_id="copy-1",
        student_feedback={"Q1": "f1"},
        total_score=4.0,
        max_score=5.0,
    )

    service = AnnotationExportService()
    computed_annotations = CopyAnnotations(
        copy_id="copy-1",
        student_name="Alice",
        placements=[
            AnnotationPlacement(
                question_id="Q1",
                feedback_text="f1",
                page_number=1,
                x_percent=10,
                y_percent=20,
            )
        ],
    )
    calls = {"prepare": 0, "annotate": 0, "overlay": 0}

    def fake_prepare_annotations(**kwargs):
        calls["prepare"] += 1
        return computed_annotations

    def fake_annotate_copy(**kwargs):
        calls["annotate"] += 1
        assert kwargs["annotations"] is computed_annotations
        output_path = Path(kwargs["output_path"])
        output_path.write_text("annotated", encoding="utf-8")
        return str(output_path)

    def fake_create_annotation_overlay(**kwargs):
        calls["overlay"] += 1
        assert kwargs["annotations"] is computed_annotations
        output_path = Path(kwargs["output_path"])
        output_path.write_text("overlay", encoding="utf-8")
        return str(output_path)

    monkeypatch.setattr(service.annotator, "prepare_annotations", fake_prepare_annotations)
    monkeypatch.setattr(service.annotator, "annotate_copy", fake_annotate_copy)
    monkeypatch.setattr(service.annotator, "create_annotation_overlay", fake_create_annotation_overlay)

    artifact = service.export_copy_artifacts(
        copy=copy,
        graded=graded,
        output_dir=str(tmp_path / "outputs"),
    )

    assert calls == {"prepare": 1, "annotate": 1, "overlay": 1}
    assert Path(artifact.annotated_pdf).exists()
    assert Path(artifact.overlay_pdf).exists()


def test_detector_corrects_llm_page_to_expected_page_from_pdf_text(tmp_path):
    pdf_path = tmp_path / "copy.pdf"
    create_pdf(pdf_path, ["Question 1\nReponse de l'eleve", "Question 2\nAutre reponse"])

    graded = GradedCopy(
        copy_id="copy-1",
        student_feedback={"Q1": "f1"},
        total_score=4.0,
        max_score=5.0,
    )

    class ProviderStub:
        def call_vision(self, prompt, image_path):
            return """
            {
              "annotations": [
                {
                  "question_id": "Q1",
                  "page": 2,
                  "feedback_text": "f1",
                  "x_percent": 15,
                  "y_percent": 20,
                  "confidence": 0.9
                }
              ]
            }
            """

    detector = AnnotationCoordinateDetector(provider=ProviderStub())
    annotations = detector.build_annotations(
        pdf_path=str(pdf_path),
        graded_copy=graded,
        language="fr",
    )

    assert len(annotations.placements) == 1
    placement = annotations.placements[0]
    assert placement.page_number == 1
    assert placement.page_validated is True
    assert placement.page_correction_reason == "corrected_to_expected_page"
    assert placement.x_percent == 70.0
    assert placement.placement == "right_margin"


def test_detector_infers_single_page_as_expected_page_for_all_questions(tmp_path):
    pdf_path = tmp_path / "copy.pdf"
    create_pdf(pdf_path, ["Contenu unique"])

    detector = AnnotationCoordinateDetector(provider=object())
    expected = detector._infer_expected_pages_by_question(str(pdf_path), ["Q1", "Q2"])

    assert expected == {"Q1": [1], "Q2": [1]}


def test_create_annotation_boxes_avoids_overlap_on_same_page(tmp_path):
    pdf_path = tmp_path / "copy.pdf"
    create_pdf(pdf_path, ["Page 1"])

    doc = fitz.open(str(pdf_path))
    try:
        annotations = CopyAnnotations(
            copy_id="copy-1",
            student_name="Alice",
            placements=[
                AnnotationPlacement(
                    question_id="Q1",
                    feedback_text="Feedback 1",
                    page_number=1,
                    x_percent=70,
                    y_percent=20,
                    width_percent=25,
                    height_percent=8,
                ),
                AnnotationPlacement(
                    question_id="Q2",
                    feedback_text="Feedback 2",
                    page_number=1,
                    x_percent=70,
                    y_percent=20,
                    width_percent=25,
                    height_percent=8,
                ),
            ],
        )

        boxes = create_annotation_boxes(annotations, doc)
        rect1 = boxes[0][0][0]
        rect2 = boxes[0][1][0]

        assert not rect1.intersects(rect2)
        assert rect2.y0 > rect1.y0
    finally:
        doc.close()


def test_overlay_pdf_contains_annotations_without_page_label(tmp_path):
    pdf_path = tmp_path / "copy.pdf"
    overlay_path = tmp_path / "overlay.pdf"
    create_pdf(pdf_path, ["Original page 1"])

    copy = CopyDocument(
        id="copy-1",
        pdf_path=str(pdf_path),
        student_name="Alice",
    )
    graded = GradedCopy(
        copy_id="copy-1",
        student_feedback={"Q1": "Feedback visible"},
        total_score=4.0,
        max_score=5.0,
    )

    annotator = PDFAnnotator()
    annotations = CopyAnnotations(
        copy_id="copy-1",
        student_name="Alice",
        placements=[
            AnnotationPlacement(
                question_id="Q1",
                feedback_text="Feedback visible",
                page_number=1,
                x_percent=70,
                y_percent=20,
                width_percent=25,
                height_percent=8,
            )
        ],
    )

    annotator.create_annotation_overlay(
        copy=copy,
        graded=graded,
        output_path=str(overlay_path),
        annotations=annotations,
    )

    doc = fitz.open(str(overlay_path))
    try:
        text = doc[0].get_text()
        assert "Feedback visible" in text
        assert "Annotations" not in text
        assert "Page 1 -" not in text
    finally:
        doc.close()
