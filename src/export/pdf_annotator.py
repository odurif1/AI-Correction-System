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

PROFESSOR_RED = (0.82, 0.07, 0.12)


def _hex_to_pdf_color(value: str) -> Tuple[float, float, float]:
    color = (value or "").strip().lstrip("#")
    if len(color) != 6:
        return PROFESSOR_RED
    try:
        return tuple(int(color[i:i + 2], 16) / 255 for i in (0, 2, 4))
    except ValueError:
        return PROFESSOR_RED


def _font_name(bold: bool = False, italic: bool = False) -> str:
    if bold and italic:
        return "hebi"
    if bold:
        return "hebo"
    if italic:
        return "heit"
    return "helv"


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
        language: str = 'fr',
        annotations: 'CopyAnnotations' = None,
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

        if annotations is None:
            annotations = self.prepare_annotations(
                copy=copy,
                graded=graded,
                smart_placement=smart_placement,
                language=language,
            )

        # Open and process with proper resource management
        doc = None
        try:
            doc = fitz.open(copy.pdf_path)

            original_page_count = len(doc)

            # Add cover page with summary
            self._add_cover_page(doc, copy, graded)

            # Annotate pages
            if annotations and annotations.placements:
                # Use smart placement
                self._annotate_with_smart_placement(
                    doc, copy, graded, annotations, original_page_count
                )
            else:
                # Fall back to heuristic placement
                for original_page_num in range(original_page_count):
                    page = doc[original_page_num + 1]
                    self._annotate_page(page, original_page_num, copy, graded)

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
        language: str = 'fr',
        annotations: 'CopyAnnotations' = None,
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

        if annotations is None:
            annotations = self.prepare_annotations(
                copy=copy,
                graded=graded,
                smart_placement=smart_placement,
                language=language,
            )

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
                question_annotations, meta_annotations = self._split_annotations(annotations)
                meta_boxes_by_page = self._create_meta_boxes(meta_annotations, overlay_doc)
                reserved_rects_by_page = self._build_reserved_rects_by_page(
                    overlay_doc,
                    graded,
                    first_student_page_index=0,
                    meta_boxes_by_page=meta_boxes_by_page,
                )
                boxes_by_page = create_annotation_boxes(
                    question_annotations,
                    overlay_doc,
                    reserved_rects_by_page=reserved_rects_by_page,
                )

                for page_num in range(len(overlay_doc)):
                    page = overlay_doc[page_num]
                    if page_num in meta_boxes_by_page:
                        for rect, placement in meta_boxes_by_page[page_num]:
                            self._add_meta_annotation(page, rect, placement)
                    elif page_num == 0:
                        self._add_total_score_stamp(page, graded)
                        self._add_overall_feedback_stamp(page, graded)

                    # Add question annotations
                    if page_num in boxes_by_page:
                        for rect, placement in boxes_by_page[page_num]:
                            self._add_feedback_annotation(
                                page,
                                rect,
                                placement.question_id,
                                getattr(placement, "label_text", None),
                                placement.feedback_text,
                                graded,
                                font_size_override=getattr(placement, "font_size", None),
                                text_color=getattr(placement, "text_color", None),
                                bold_override=getattr(placement, "bold", False),
                                italic_override=getattr(placement, "italic", False),
                                boxed=getattr(placement, "boxed", False),
                            )
            elif len(overlay_doc) > 0:
                self._add_total_score_stamp(overlay_doc[0], graded)
                self._add_overall_feedback_stamp(overlay_doc[0], graded)

            # Save overlay
            overlay_doc.save(output_path)
        finally:
            if overlay_doc is not None:
                overlay_doc.close()
            if original_doc is not None:
                original_doc.close()

        return output_path

    def prepare_annotations(
        self,
        copy: CopyDocument,
        graded: GradedCopy,
        smart_placement: bool = True,
        language: str = 'fr',
    ) -> 'CopyAnnotations':
        """
        Compute annotation placements once so multiple renderers can reuse them.
        """
        try:
            return self.coordinate_detector.build_annotations(
                pdf_path=copy.pdf_path,
                graded_copy=graded,
                language=language,
                student_name=copy.student_name,
                use_llm=smart_placement,
            )
        except Exception as e:
            print(f"Warning: Annotation placement failed, retrying with heuristic placement: {e}")
            return self.coordinate_detector.build_annotations(
                pdf_path=copy.pdf_path,
                graded_copy=graded,
                language=language,
                student_name=copy.student_name,
                use_llm=False,
            )

    def _annotate_with_smart_placement(
        self,
        doc: fitz.Document,
        copy: CopyDocument,
        graded: GradedCopy,
        annotations: 'CopyAnnotations',
        original_page_count: int,
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

        question_annotations, meta_annotations = self._split_annotations(annotations)
        meta_boxes_by_page = self._create_meta_boxes(meta_annotations, doc, page_number_offset=1)
        reserved_rects_by_page = self._build_reserved_rects_by_page(
            doc,
            graded,
            first_student_page_index=1,
            meta_boxes_by_page=meta_boxes_by_page,
        )

        # Group annotations by page
        boxes_by_page = create_annotation_boxes(
            question_annotations,
            doc,
            page_number_offset=1,
            reserved_rects_by_page=reserved_rects_by_page,
        )

        # Add annotations to each page
        for page_num in range(1, original_page_count + 1):
            page = doc[page_num]
            if page_num in meta_boxes_by_page:
                for rect, placement in meta_boxes_by_page[page_num]:
                    self._add_meta_annotation(page, rect, placement)
            elif page_num == 1:
                self._add_total_score_stamp(page, graded)
                self._add_overall_feedback_stamp(page, graded)

            # Add smart-placed feedback annotations
            if page_num in boxes_by_page:
                for rect, placement in boxes_by_page[page_num]:
                            self._add_feedback_annotation(
                                page,
                                rect,
                                placement.question_id,
                                getattr(placement, "label_text", None),
                                placement.feedback_text,
                                graded,
                                font_size_override=getattr(placement, "font_size", None),
                                text_color=getattr(placement, "text_color", None),
                                bold_override=getattr(placement, "bold", False),
                                italic_override=getattr(placement, "italic", False),
                                boxed=getattr(placement, "boxed", False),
                            )

    def _add_feedback_annotation(
        self,
        page: fitz.Page,
        rect: fitz.Rect,
        question_id: str,
        label_text: Optional[str],
        feedback_text: str,
        graded: GradedCopy,
        font_size_override: Optional[int] = None,
        text_color: Optional[str] = None,
        bold_override: Optional[bool] = None,
        italic_override: Optional[bool] = None,
        boxed: bool = False,
    ):
        """
        Add a feedback annotation at the specified position.

        Args:
            page: PDF page
            rect: Rectangle for annotation
            question_id: Question identifier
            feedback_text: Feedback text to display
            graded: Graded copy (for context/color)
        """
        header = self._format_annotation_header(question_id, graded, label_text=label_text)
        full_text = f"{header} {feedback_text}".strip() if header else feedback_text
        preferred_font_size = max(7, min(18, int(font_size_override or 10)))
        font_size, lines = self._fit_annotation_text_to_rect(full_text, rect, preferred_font_size=preferred_font_size)
        line_height = font_size + 3
        color = _hex_to_pdf_color(text_color or "#B0121F")
        bold = bool(bold_override)
        italic = bool(italic_override)

        if boxed:
            page.draw_rect(rect, color=color, width=1)

        # Draw each line
        y_pos = rect.y0 + font_size
        for line in lines:
            if y_pos > rect.y1:
                break  # Stop if we exceed the box height
            self._add_text(
                page,
                line,
                rect.x0,
                y_pos,
                size=font_size,
                bold=bold,
                italic=italic,
                color=color
            )
            y_pos += line_height

    def _format_annotation_header(self, question_id: str, graded: GradedCopy, label_text: Optional[str] = None) -> str:
        """Format the question label with earned points for inline annotations."""
        if question_id.startswith("__"):
            return ""
        base_label = (label_text or "").strip()
        grade = graded.grades.get(question_id)
        max_points = graded.max_points_by_question.get(question_id)

        if grade is None:
            return base_label
        if max_points is None:
            return f"{base_label} ({grade:.1f})".strip()

        max_display = int(max_points) if max_points == int(max_points) else max_points
        return f"{base_label} ({grade:.1f}/{max_display})".strip()

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

    def _fit_annotation_text_to_rect(
        self,
        text: str,
        rect: fitz.Rect,
        preferred_font_size: int = 8,
        min_font_size: int = 6,
    ) -> Tuple[int, List[str]]:
        """Fit annotation text into its box by shrinking modestly, then truncating."""
        available_height = max(rect.height - 2.0, 1.0)

        for font_size in range(preferred_font_size, min_font_size - 1, -1):
            line_height = font_size + 3
            max_lines = max(1, int(available_height // line_height))
            lines = self._wrap_text(text, rect.width, font_size)
            if len(lines) <= max_lines:
                return font_size, lines

        font_size = min_font_size
        line_height = font_size + 3
        max_lines = max(1, int(available_height // line_height))
        lines = self._wrap_text(text, rect.width, font_size)
        return font_size, self._truncate_annotation_lines(lines, max_lines, rect.width, font_size)

    def _truncate_annotation_lines(
        self,
        lines: List[str],
        max_lines: int,
        max_width: float,
        font_size: int,
    ) -> List[str]:
        """Truncate wrapped lines cleanly without adding visible markers."""
        if len(lines) <= max_lines:
            return lines
        if max_lines <= 0:
            return []

        visible = list(lines[:max_lines])
        visible[-1] = self._truncate_line_to_width(visible[-1], max_width, font_size)
        return visible

    def _truncate_line_to_width(self, text: str, max_width: float, font_size: int) -> str:
        """Trim a single line until it fits the available width."""
        char_width = max(font_size * 0.5, 1.0)
        max_chars = max(1, int(max_width / char_width))
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip()

    def _add_cover_page(
        self,
        doc: fitz.Document,
        copy: CopyDocument,
        graded: GradedCopy
    ):
        """Add a cover page with grading summary."""
        # Create new page at beginning
        page = doc.new_page(pno=0, width=595, height=842)  # A4 size

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
        if page_num == 0:
            self._add_total_score_stamp(page, graded)

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
        italic: bool = False,
        color: Tuple[float, float, float] = (0, 0, 0)
    ):
        """Add text to a page."""
        fontname = _font_name(bold=bold, italic=italic)
        page.insert_text(
            (x, y),
            text,
            fontname=fontname,
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
        italic: bool = False,
        color: Tuple[float, float, float] = (0, 0, 0)
    ):
        """Add centered text to a page."""
        text_width = len(text) * size * 0.5  # Approximate
        x = (page.rect.width - text_width) / 2

        page.insert_text(
            (x, y),
            text,
            fontname=_font_name(bold=bold, italic=italic),
            fontsize=size,
            color=color
        )

    def _add_total_score_stamp(self, page: fitz.Page, graded: GradedCopy) -> None:
        """Add the total score prominently at the top of the first student page."""
        max_display = int(graded.max_score) if graded.max_score == int(graded.max_score) else graded.max_score
        score_text = f"Note : {graded.total_score:.1f}/{max_display}"
        x = max(24, page.rect.width - 180)
        y = 28
        self._add_text(page, score_text, x, y, size=16, bold=True, color=PROFESSOR_RED)

    def _add_overall_feedback_stamp(self, page: fitz.Page, graded: GradedCopy) -> None:
        """Add the overall student feedback near the top of the first page."""
        if not graded.feedback:
            return

        summary = " ".join(graded.feedback.split())
        lines = self._wrap_text(summary, max_width=page.rect.width - 48, font_size=10)[:3]
        if not lines:
            return

        y = 52
        for line in lines:
            self._add_text(page, line, 24, y, size=10, color=PROFESSOR_RED)
            y += 12

    def _build_reserved_rects_by_page(
        self,
        doc: fitz.Document,
        graded: GradedCopy,
        first_student_page_index: int,
        meta_boxes_by_page: Dict[int, List[Tuple[fitz.Rect, Any]]] | None = None,
    ) -> Dict[int, List[fitz.Rect]]:
        """Reserve the header band used by the score and overall feedback."""
        if meta_boxes_by_page:
            return {
                page_index: [fitz.Rect(rect) for rect, _ in boxes]
                for page_index, boxes in meta_boxes_by_page.items()
            }
        if first_student_page_index < 0 or first_student_page_index >= len(doc):
            return {}

        page = doc[first_student_page_index]
        header_bottom = 52.0
        if graded.feedback:
            summary = " ".join(graded.feedback.split())
            lines = self._wrap_text(summary, max_width=page.rect.width - 48, font_size=10)[:3]
            header_bottom += (len(lines) * 12.0) + 8.0
        else:
            header_bottom += 8.0

        return {
            first_student_page_index: [
                fitz.Rect(
                    12.0,
                    12.0,
                    page.rect.width - 12.0,
                    min(page.rect.height - 12.0, header_bottom),
                )
            ]
        }

    def _split_annotations(self, annotations: "CopyAnnotations") -> Tuple["CopyAnnotations", "CopyAnnotations"]:
        from export.annotation_service import CopyAnnotations

        meta = [placement for placement in annotations.placements if placement.question_id.startswith("__")]
        questions = [placement for placement in annotations.placements if not placement.question_id.startswith("__")]
        return (
            CopyAnnotations(copy_id=annotations.copy_id, student_name=annotations.student_name, placements=questions),
            CopyAnnotations(copy_id=annotations.copy_id, student_name=annotations.student_name, placements=meta),
        )

    def _create_meta_boxes(
        self,
        annotations: "CopyAnnotations",
        doc: fitz.Document,
        page_number_offset: int = 0,
    ) -> Dict[int, List[Tuple[fitz.Rect, Any]]]:
        from export.annotation_service import _build_annotation_rect

        boxes: Dict[int, List[Tuple[fitz.Rect, Any]]] = {}
        for placement in annotations.placements:
            page_index = max(0, min(placement.page_number - 1 + page_number_offset, len(doc) - 1))
            page = doc[page_index]
            rect = _build_annotation_rect(placement, page.rect.width, page.rect.height)
            boxes.setdefault(page_index, []).append((rect, placement))
        return boxes

    def _add_meta_annotation(self, page: fitz.Page, rect: fitz.Rect, placement: Any) -> None:
        font_size = max(7, min(22, int(getattr(placement, "font_size", 16 if placement.question_id == "__total_score__" else 10))))
        bold = bool(getattr(placement, "bold", placement.question_id == "__total_score__"))
        italic = bool(getattr(placement, "italic", False))
        line_height = font_size + (3 if bold else 2)
        lines = self._wrap_text(placement.feedback_text, rect.width, font_size)
        y_pos = rect.y0 + font_size
        color = _hex_to_pdf_color(getattr(placement, "text_color", "#B0121F"))
        if getattr(placement, "boxed", False):
            page.draw_rect(rect, color=color, width=1)
        for line in lines:
            if y_pos > rect.y1:
                break
            self._add_text(page, line, rect.x0, y_pos, size=font_size, bold=bold, italic=italic, color=color)
            y_pos += line_height

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
