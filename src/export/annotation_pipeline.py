"""
Annotation export pipeline.

Builds annotation placements once per copy, then renders both export targets:
- annotated student PDF
- overlay-only PDF for superimposition
"""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import fitz

from core.models import CopyDocument, GradedCopy, GradingSession
from export.pdf_annotator import PDFAnnotator


@dataclass
class AnnotationArtifact:
    """Generated annotation files for a single copy."""
    copy_id: str
    student_name: str | None
    annotated_pdf: str
    overlay_pdf: str


@dataclass
class SessionAnnotationArtifacts:
    """Generated combined annotation files for a full session."""
    annotated_pdf: str
    overlay_pdf: str
    copy_count: int


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
        annotations: "CopyAnnotations" = None,
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

        if annotations is None:
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
        annotations_by_copy: dict[str, "CopyAnnotations"] | None = None,
    ) -> SessionAnnotationArtifacts:
        """
        Export both annotation artifacts for the full session as two combined PDFs.
        """
        base_dir = Path(output_dir)
        annotated_dir = base_dir / "annotated"
        overlay_dir = base_dir / "overlays"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        annotated_output = annotated_dir / "copies_annotees.pdf"
        overlay_output = overlay_dir / "annotations_overlay.pdf"

        artifacts: List[AnnotationArtifact] = []
        with TemporaryDirectory(dir=base_dir) as temp_dir:
            temp_base = Path(temp_dir)
            for index, (copy, graded) in enumerate(zip(copies, graded_copies), start=1):
                student_name = copy.student_name or f"copie_{index}"
                artifacts.append(
                    self.export_copy_artifacts(
                        copy=copy,
                        graded=graded,
                        output_dir=str(temp_base),
                        smart_placement=smart_placement,
                        language=language,
                        filename_stem=f"{index:03d}_{student_name}",
                        annotations=(annotations_by_copy or {}).get(copy.id),
                    )
                )

            self._merge_pdfs(
                [artifact.annotated_pdf for artifact in artifacts],
                annotated_output,
            )
            self._merge_pdfs(
                [artifact.overlay_pdf for artifact in artifacts],
                overlay_output,
            )

        return SessionAnnotationArtifacts(
            annotated_pdf=str(annotated_output),
            overlay_pdf=str(overlay_output),
            copy_count=len(artifacts),
        )

    @staticmethod
    def _merge_pdfs(pdf_paths: List[str], output_path: Path) -> None:
        """Merge PDFs in order into a single output file."""
        merged_doc = fitz.open()
        try:
            for pdf_path in pdf_paths:
                source_doc = fitz.open(pdf_path)
                try:
                    merged_doc.insert_pdf(source_doc)
                finally:
                    source_doc.close()
            merged_doc.save(str(output_path))
        finally:
            merged_doc.close()
