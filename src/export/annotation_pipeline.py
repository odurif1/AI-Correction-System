"""
Annotation export pipeline.

Builds annotation placements once per copy, then renders both export targets:
- annotated student PDF
- overlay-only PDF for superimposition
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from core.models import CopyDocument, GradedCopy, GradingSession
from export.pdf_annotator import PDFAnnotator


@dataclass
class AnnotationArtifact:
    """Generated annotation files for a single copy."""
    copy_id: str
    student_name: str | None
    annotated_pdf: str
    overlay_pdf: str


class AnnotationExportService:
    """Generate all annotation-related export files for a grading session."""

    def __init__(self, session: GradingSession | None = None, annotation_provider=None):
        self.session = session
        self.annotator = PDFAnnotator(session=session, annotation_provider=annotation_provider)

    def export_copy_artifacts(
        self,
        copy: CopyDocument,
        graded: GradedCopy,
        output_dir: str,
        smart_placement: bool = True,
        language: str = "fr",
        filename_stem: str | None = None,
    ) -> AnnotationArtifact:
        """
        Export both annotation artifacts for a single student copy.
        """
        base_dir = Path(output_dir)
        annotated_dir = base_dir / "annotated"
        overlay_dir = base_dir / "overlays"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        stem = filename_stem or copy.student_name or copy.id
        safe_stem = stem.replace(" ", "_").replace("/", "-")

        annotations = self.annotator.prepare_annotations(
            copy=copy,
            graded=graded,
            smart_placement=smart_placement,
            language=language,
        )

        annotated_path = annotated_dir / f"{safe_stem}_annotated.pdf"
        overlay_path = overlay_dir / f"{safe_stem}_overlay.pdf"

        self.annotator.annotate_copy(
            copy=copy,
            graded=graded,
            output_path=str(annotated_path),
            smart_placement=smart_placement,
            language=language,
            annotations=annotations,
        )
        self.annotator.create_annotation_overlay(
            copy=copy,
            graded=graded,
            output_path=str(overlay_path),
            smart_placement=smart_placement,
            language=language,
            annotations=annotations,
        )

        return AnnotationArtifact(
            copy_id=copy.id,
            student_name=copy.student_name,
            annotated_pdf=str(annotated_path),
            overlay_pdf=str(overlay_path),
        )

    def export_session_artifacts(
        self,
        copies: List[CopyDocument],
        graded_copies: List[GradedCopy],
        output_dir: str,
        smart_placement: bool = True,
        language: str = "fr",
    ) -> List[AnnotationArtifact]:
        """
        Export both annotation artifacts for each graded copy in the session.
        """
        artifacts: List[AnnotationArtifact] = []

        for index, (copy, graded) in enumerate(zip(copies, graded_copies), start=1):
            student_name = copy.student_name or f"copie_{index}"
            artifacts.append(
                self.export_copy_artifacts(
                    copy=copy,
                    graded=graded,
                    output_dir=output_dir,
                    smart_placement=smart_placement,
                    language=language,
                    filename_stem=student_name,
                )
            )

        return artifacts
