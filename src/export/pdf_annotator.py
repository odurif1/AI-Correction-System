"""
PDF annotation for student feedback.

Adds grades, comments, and feedback annotations to student PDFs.
"""

import fitz  # PyMuPDF
from typing import Dict, List, Optional, Tuple
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
    - Adds comment boxes
    - Highlights areas of interest
    - Preserves original layout
    """

    def __init__(self, session: GradingSession = None):
        """
        Initialize annotator.

        Args:
            session: Grading session for context
        """
        self.session = session

    def annotate_copy(
        self,
        copy: CopyDocument,
        graded: GradedCopy,
        output_path: str = None
    ) -> str:
        """
        Annotate a student's copy with grading results.

        Args:
            copy: Original copy document
            graded: Graded copy with results
            output_path: Output PDF path (auto-generated if None)

        Returns:
            Path to annotated PDF
        """
        # Determine output path
        if output_path is None:
            output_path = f"{copy.id}_annotated.pdf"

        # Open and process with proper resource management
        doc = None
        try:
            doc = fitz.open(copy.pdf_path)

            # Add cover page with summary
            self._add_cover_page(doc, copy, graded)

            # Annotate each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                self._annotate_page(page, page_num, copy, graded)

            # Save
            doc.save(output_path)
        finally:
            if doc is not None:
                doc.close()

        return output_path

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
        flags = fitz.TEXT_FONT_BOLD if bold else 0
        page.insert_text(
            (x, y),
            text,
            fontname="helvetica",
            fontsize=size,
            color=color,
            flags=flags
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
        flags = fitz.TEXT_FONT_BOLD if bold else 0
        text_width = len(text) * size * 0.5  # Approximate
        x = (page.rect.width - text_width) / 2

        page.insert_text(
            (x, y),
            text,
            fontname="helvetica",
            fontsize=size,
            color=color,
            flags=flags
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
