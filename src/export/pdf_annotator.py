"""
PDF annotation for student feedback.

Adds grades, comments, and feedback annotations to student PDFs.
Supports intelligent annotation placement using LLM vision capabilities.
"""

import fitz  # PyMuPDF
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime

from core.models import CopyDocument, GradedCopy, GradingSession
from config.constants import (
    ANNOTATION_FONT_SIZE, ANNOTATION_COLOR_CORRECT,
    ANNOTATION_COLOR_PARTIAL, ANNOTATION_COLOR_WRONG,
    ANNOTATION_ALPHA
)


class PDFAnnotator:
    """
    Annotates PDFs with grading information.

    Features:
    - Adds grade stamps
    - Adds comment boxes with intelligent placement
    - Highlights areas of interest
    - Preserves original layout
    - Supports LLM-based annotation coordinate detection
    """

    def __init__(self, session: GradingSession = None, annotation_provider=None):
        """
        Initialize annotator.

        Args:
            session: Grading session for context
            annotation_provider: Optional AI provider for coordinate detection
        """
        self.session = session
        self.annotation_provider = annotation_provider
        self._coordinate_detector = None

    @property
    def coordinate_detector(self):
        """Lazy-load the coordinate detector."""
        if self._coordinate_detector is None:
            from export.annotation_service import AnnotationCoordinateDetector
            self._coordinate_detector = AnnotationCoordinateDetector(
                provider=self.annotation_provider
            )
        return self._coordinate_detector

    def annotate_copy(
        self,
        copy: CopyDocument,
        graded: GradedCopy,
        output_path: str = None,
        smart_placement: bool = True,
        language: str = 'fr'
    ) -> str:
        """
        Annotate a student's copy with grading results.

        Args:
            copy: Original copy document
            graded: Graded copy with results
            output_path: Output PDF path (auto-generated if None)
            smart_placement: Use LLM for intelligent annotation placement
            language: Language for annotation prompts

        Returns:
            Path to annotated PDF
        """
        # Determine output path
        if output_path is None:
            output_path = f"{copy.id}_annotated.pdf"

        # Detect annotation coordinates if smart placement enabled
        annotations = None
        if smart_placement and graded.student_feedback:
            try:
                annotations = self.coordinate_detector.detect_annotations(
                    pdf_path=copy.pdf_path,
                    graded_copy=graded,
                    language=language,
                    student_name=copy.student_name
                )
            except Exception as e:
                print(f"Warning: Smart placement failed, using heuristic: {e}")
                annotations = None

        # Open and process with proper resource management
        doc = None
        try:
            doc = fitz.open(copy.pdf_path)

            # Add cover page with summary
            self._add_cover_page(doc, copy, graded)

            # Annotate pages
            if annotations and annotations.placements:
                # Use smart placement
                self._annotate_with_smart_placement(doc, copy, graded, annotations)
            else:
                # Fall back to heuristic placement
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    self._annotate_page(page, page_num, copy, graded)

            # Save
            doc.save(output_path)
        finally:
            if doc is not None:
                doc.close()

        return output_path

    def create_annotation_overlay(
        self,
        copy: CopyDocument,
        graded: GradedCopy,
        output_path: str = None,
        smart_placement: bool = True,
        language: str = 'fr'
    ) -> str:
        """
        Create a transparent overlay PDF with only annotations.

        This overlay can be superimposed on the original copy.
        Useful for:
        - Printing annotations separately
        - Overlaying on scanned copies
        - Non-destructive annotation

        Args:
            copy: Original copy document (for dimensions)
            graded: Graded copy with results
            output_path: Output PDF path (auto-generated if None)
            smart_placement: Use LLM for intelligent annotation placement
            language: Language for annotation prompts

        Returns:
            Path to overlay PDF
        """
        # Determine output path
        if output_path is None:
            output_path = f"{copy.id}_overlay.pdf"

        # Detect annotation coordinates
        annotations = None
        if smart_placement and graded.student_feedback:
            try:
                annotations = self.coordinate_detector.detect_annotations(
                    pdf_path=copy.pdf_path,
                    graded_copy=graded,
                    language=language,
                    student_name=copy.student_name
                )
            except Exception as e:
                print(f"Warning: Smart placement failed, using heuristic: {e}")
                annotations = self._heuristic_annotations(copy, graded)

        # Create overlay document
        overlay_doc = None
        original_doc = None
        try:
            # Open original to get page dimensions
            original_doc = fitz.open(copy.pdf_path)
            overlay_doc = fitz.open()

            # Create each overlay page
            for page_num in range(len(original_doc)):
                orig_page = original_doc[page_num]
                # Create new page with same dimensions
                overlay_page = overlay_doc.new_page(
                    width=orig_page.rect.width,
                    height=orig_page.rect.height
                )

            # Add annotations to overlay pages
            if annotations and annotations.placements:
                from export.annotation_service import create_annotation_boxes
                boxes_by_page = create_annotation_boxes(annotations, overlay_doc)

                for page_num in range(len(overlay_doc)):
                    page = overlay_doc[page_num]

                    # Add question annotations
                    if page_num in boxes_by_page:
                        for rect, feedback_text in boxes_by_page[page_num]:
                            self._add_feedback_annotation(page, rect, feedback_text, graded)

                    # Add page label
                    self._add_text(
                        page,
                        f"Page {page_num + 1} - Annotations",
                        10, 10,
                        size=8,
                        color=(0.5, 0.5, 0.5)
                    )

            # Save overlay
            overlay_doc.save(output_path)
        finally:
            if overlay_doc is not None:
                overlay_doc.close()
            if original_doc is not None:
                original_doc.close()

        return output_path

    def _heuristic_annotations(self, copy: CopyDocument, graded: GradedCopy) -> 'CopyAnnotations':
        """Generate heuristic annotations when LLM fails."""
        from export.annotation_service import CopyAnnotations, AnnotationPlacement

        annotations = CopyAnnotations(
            copy_id=copy.id,
            student_name=copy.student_name
        )

        # Distribute feedback across pages
        num_pages = 1
        try:
            doc = fitz.open(copy.pdf_path)
            try:
                num_pages = len(doc)
            finally:
                doc.close()
        except (FileNotFoundError, PermissionError, OSError) as e:
            # PDF access failed, use default of 1 page
            pass

        questions = list(graded.student_feedback.keys())
        questions_per_page = max(1, len(questions) // num_pages)

        for i, (q_id, feedback) in enumerate(graded.student_feedback.items()):
            page_num = min(i // max(1, questions_per_page), num_pages - 1)
            y_position = 15.0 + (i % max(1, questions_per_page)) * 12.0

            annotations.placements.append(AnnotationPlacement(
                question_id=q_id,
                feedback_text=feedback,
                page_number=page_num + 1,
                x_percent=70.0,
                y_percent=y_position,
                width_percent=25.0,
                height_percent=5.0,
                placement="right_margin",
                confidence=0.5
            ))

        return annotations

    def _annotate_with_smart_placement(
        self,
        doc: fitz.Document,
        copy: CopyDocument,
        graded: GradedCopy,
        annotations: 'CopyAnnotations'
    ):
        """
        Annotate PDF using LLM-determined coordinates.

        Args:
            doc: PDF document
            copy: Original copy document
            graded: Graded copy with results
            annotations: Smart annotation placements
        """
        from export.annotation_service import create_annotation_boxes

        # Group annotations by page
        boxes_by_page = create_annotation_boxes(annotations, doc)

        # Add annotations to each page
        for page_num in range(len(doc)):
            page = doc[page_num]

            # Add page summary in margin
            rect = fitz.Rect(page.rect.width - 80, 50, page.rect.width - 10, 100)
            self._add_annotation_box(
                page,
                rect,
                f"Page {page_num + 1}",
                f"Score: {graded.total_score:.1f}/{graded.max_score:.1f}"
            )

            # Add smart-placed feedback annotations
            if page_num in boxes_by_page:
                for rect, feedback_text in boxes_by_page[page_num]:
                    self._add_feedback_annotation(page, rect, feedback_text, graded)

    def _add_feedback_annotation(
        self,
        page: fitz.Page,
        rect: fitz.Rect,
        feedback_text: str,
        graded: GradedCopy
    ):
        """
        Add a feedback annotation at the specified position.

        Args:
            page: PDF page
            rect: Rectangle for annotation
            feedback_text: Feedback text to display
            graded: Graded copy (for context/color)
        """
        # Draw light background
        page.draw_rect(
            rect,
            color=(0.6, 0.8, 1.0),  # Blue border
            fill=(0.95, 0.98, 1.0),  # Very light blue fill
            width=1.0
        )

        # Calculate text layout
        font_size = 8
        line_height = font_size + 3
        margin = 4
        max_width = rect.width - (margin * 2)

        # Word wrap the feedback
        lines = self._wrap_text(feedback_text, max_width, font_size)

        # Draw each line
        y_pos = rect.y0 + margin + font_size
        for line in lines:
            if y_pos > rect.y1 - margin:
                break  # Stop if we exceed the box height
            self._add_text(
                page,
                line,
                rect.x0 + margin,
                y_pos,
                size=font_size,
                color=(0.1, 0.2, 0.4)
            )
            y_pos += line_height

    def _wrap_text(self, text: str, max_width: float, font_size: int) -> List[str]:
        """
        Wrap text to fit within a given width.

        Args:
            text: Text to wrap
            max_width: Maximum width in points
            font_size: Font size

        Returns:
            List of lines
        """
        # Approximate character width (helvetica)
        char_width = font_size * 0.5
        max_chars = int(max_width / char_width)

        if max_chars <= 0:
            return [text[:20]]  # Fallback

        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            if len(test_line) <= max_chars:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                # Handle very long words
                if len(word) > max_chars:
                    lines.append(word[:max_chars-1] + "…")
                    current_line = ""
                else:
                    current_line = word

        if current_line:
            lines.append(current_line)

        return lines if lines else [text[:max_chars]]

    def _add_cover_page(
        self,
        doc: fitz.Document,
        copy: CopyDocument,
        graded: GradedCopy
    ):
        """Add a cover page with grading summary."""
        # Create new page at beginning
        page = doc.new_page(width=595, height=842)  # A4 size

        # Title
        title = f"Graded Assessment - {copy.student_name or 'Student'}"
        self._add_text_centered(page, title, y=80, size=18, bold=True)

        # Score
        score_text = f"Score: {graded.total_score:.1f} / {graded.max_score:.1f}"
        self._add_text_centered(page, score_text, y=120, size=24, bold=True)

        # Percentage
        percentage = (graded.total_score / graded.max_score * 100) if graded.max_score > 0 else 0
        percentage_text = f"({percentage:.1f}%)"
        self._add_text_centered(page, percentage_text, y=150, size=16)

        # Divider
        self._add_line(page, y=180)

        # Question breakdown
        y_pos = 220
        self._add_text(page, "Question Breakdown:", 50, y_pos, size=14, bold=True)
        y_pos += 30

        for q_id, grade in graded.grades.items():
            # Get student feedback
            feedback = graded.student_feedback.get(q_id, "")
            feedback_display = feedback[:80] if feedback else ""
            text = f"Q{q_id}: {grade}/5 - {feedback_display}"
            self._add_text(page, text, 70, y_pos, size=11)
            y_pos += 20

        # Feedback
        if graded.feedback:
            y_pos += 20
            self._add_text(page, "Feedback:", 50, y_pos, size=14, bold=True)
            y_pos += 30

            # Word wrap feedback
            words = graded.feedback.split()
            line = ""
            for word in words:
                test_line = line + word + " "
                if len(test_line) > 80:
                    self._add_text(page, line, 70, y_pos, size=11)
                    y_pos += 18
                    line = word + " "
                else:
                    line = test_line
            if line:
                self._add_text(page, line, 70, y_pos, size=11)

    def _annotate_page(
        self,
        page: fitz.Page,
        page_num: int,
        copy: CopyDocument,
        graded: GradedCopy
    ):
        """Annotate a single page."""
        # Find blank areas for annotations
        zones = self._find_annotation_zones(page)

        # Add question grades if we have zone info
        # For now, add a summary in the margin

        rect = fitz.Rect(page.rect.width - 80, 50, page.rect.width - 10, 200)

        # Add summary box
        self._add_annotation_box(
            page,
            rect,
            f"Page {page_num + 1}",
            f"Total: {graded.total_score:.1f}/{graded.max_score:.1f}"
        )

    def _find_annotation_zones(self, page: fitz.Page) -> List[fitz.Rect]:
        """
        Find blank zones on a page for annotations.

        Args:
            page: PDF page

        Returns:
            List of blank rectangles
        """
        # Get page dimensions
        width, height = page.rect.width, page.rect.height

        # Define margin zones
        zones = [
            fitz.Rect(width - 100, 50, width - 10, height - 50),  # Right margin
            fitz.Rect(10, height - 100, 150, height - 10),  # Bottom left
        ]

        return zones

    def _add_annotation_box(
        self,
        page: fitz.Page,
        rect: fitz.Rect,
        title: str,
        content: str
    ):
        """Add an annotation box to a page."""
        # Draw border
        page.draw_rect(rect, color=(0.5, 0.5, 0.5), width=1)

        # Add title
        self._add_text(page, title, rect.x0 + 5, rect.y0 + 5, size=10, bold=True)

        # Add content
        self._add_text(page, content, rect.x0 + 5, rect.y0 + 20, size=9)

    def _add_text(
        self,
        page: fitz.Page,
        text: str,
        x: float,
        y: float,
        size: int = ANNOTATION_FONT_SIZE,
        bold: bool = False,
        color: Tuple[float, float, float] = (0, 0, 0)
    ):
        """Add text to a page."""
        # Use helv for normal, helv-bold doesn't exist, so we just use helvetica
        fontname = "helv" if not bold else "hebo"  # hebo is helvetica bold outline
        page.insert_text(
            (x, y),
            text,
            fontname="helv",
            fontsize=size,
            color=color
        )

    def _add_text_centered(
        self,
        page: fitz.Page,
        text: str,
        y: float,
        size: int = ANNOTATION_FONT_SIZE,
        bold: bool = False,
        color: Tuple[float, float, float] = (0, 0, 0)
    ):
        """Add centered text to a page."""
        text_width = len(text) * size * 0.5  # Approximate
        x = (page.rect.width - text_width) / 2

        page.insert_text(
            (x, y),
            text,
            fontname="helv",
            fontsize=size,
            color=color
        )

    def _add_line(self, page: fitz.Page, y: float, margin: float = 50):
        """Add a horizontal line."""
        width = page.rect.width
        page.draw_line(
            fitz.Point(margin, y),
            fitz.Point(width - margin, y),
            color=(0.5, 0.5, 0.5),
            width=1
        )


class BatchAnnotator:
    """
    Batch annotate multiple copies.
    """

    def __init__(self, session: GradingSession, output_dir: str = None):
        """
        Initialize batch annotator.

        Args:
            session: Grading session
            output_dir: Output directory (default: outputs/annotated)
        """
        self.session = session
        self.output_dir = Path(output_dir or "outputs/annotated")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.annotator = PDFAnnotator(session)

    def create_combined_report(self) -> str:
        """
        Create a combined PDF report for the teacher.

        Returns:
            Path to report PDF
        """
        doc = None
        try:
            doc = fitz.open()

            # Add title page
            page = doc.new_page(width=595, height=842)
            self.annotator._add_text_centered(
                page, "Grading Report", y=100, size=24, bold=True
            )
            self.annotator._add_text_centered(
                page, f"Session: {self.session.session_id}", y=140, size=14
            )
            self.annotator._add_text_centered(
                page, f"Date: {datetime.now().strftime('%Y-%m-%d')}", y=170, size=12
            )

            # Add statistics
            if self.session.graded_copies:
                scores = [g.total_score for g in self.session.graded_copies]
                avg = sum(scores) / len(scores)

                y = 250
                self.annotator._add_text(
                    page, f"Total Copies: {len(self.session.graded_copies)}",
                    100, y, size=12
                )
                y += 25
                self.annotator._add_text(
                    page, f"Class Average: {avg:.1f}", 100, y, size=12
                )
                y += 25
                self.annotator._add_text(
                    page, f"Highest Score: {max(scores):.1f}", 100, y, size=12
                )
                y += 25
                self.annotator._add_text(
                    page, f"Lowest Score: {min(scores):.1f}", 100, y, size=12
                )

            # Save
            output_path = self.output_dir / f"{self.session.session_id}_report.pdf"
            doc.save(str(output_path))
        finally:
            if doc is not None:
                doc.close()

        return str(output_path)
