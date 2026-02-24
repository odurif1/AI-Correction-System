"""
Annotation coordinate detection service.

This service runs in a post-processing step after grading to determine
optimal coordinates for placing student feedback on PDFs.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import fitz  # PyMuPDF

from core.models import GradedCopy
from config.settings import get_settings
from prompts.annotation import (
    build_direct_annotation_prompt,
    parse_annotation_response,
)


@dataclass
class AnnotationPlacement:
    """
    Represents a single annotation placement on a PDF.

    Coordinates are in percentages (0-100) relative to page dimensions:
    - x_percent: 0% = left edge, 100% = right edge
    - y_percent: 0% = top edge, 100% = bottom edge
    - width_percent: annotation width as % of page width
    - height_percent: annotation height as % of page height

    Page numbers are 1-based (page 1 = first page).
    """
    question_id: str           # Question identifier (e.g., "Q1", "Q2")
    feedback_text: str         # Feedback text to display
    page_number: int           # 1-based page number (1 = first page)
    x_percent: float           # Horizontal position (0-100%)
    y_percent: float           # Vertical position (0-100%)
    width_percent: float = 30.0  # Width as % of page width
    height_percent: float = 5.0  # Height as % of page height
    placement: str = "below_answer"  # Placement hint from LLM
    confidence: float = 0.0    # LLM confidence (0.0-1.0)


@dataclass
class CopyAnnotations:
    """
    All annotations for a single student copy.

    Contains a list of AnnotationPlacement objects, one per question
    that received feedback.
    """
    copy_id: str
    student_name: Optional[str]
    placements: List[AnnotationPlacement] = field(default_factory=list)


class AnnotationCoordinateDetector:
    """
    Detects optimal coordinates for placing feedback annotations.

    Uses LLM vision capabilities to analyze the PDF and determine
    where to place feedback without overlapping student text.

    The annotation LLM can be configured separately from grading LLMs
    via AI_CORRECTION_ANNOTATION_PROVIDER and AI_CORRECTION_ANNOTATION_MODEL.
    """

    def __init__(self, provider=None):
        """
        Initialize detector.

        Args:
            provider: AI provider with vision capabilities.
                      If None, will be created from settings.
        """
        self.settings = get_settings()

        # Use provided provider or create from settings
        if provider:
            self.provider = provider
        else:
            self.provider = self._create_annotation_provider()

    def _create_annotation_provider(self):
        """Create annotation provider from settings."""
        from ai.provider_factory import create_ai_provider

        provider_name = self.settings.annotation_provider
        model = self.settings.annotation_model

        # Skip annotation if no model configured
        if not model:
            print("Info: No annotation model configured. Set AI_CORRECTION_ANNOTATION_MODEL to enable smart placement.")
            return None

        # Use annotation provider if specified, otherwise use main provider
        if not provider_name:
            provider_name = self.settings.ai_provider

        try:
            return create_ai_provider(provider_name, model=model)
        except Exception as e:
            print(f"Warning: Could not create annotation provider: {e}")
            return None

    def detect_annotations(
        self,
        pdf_path: str,
        graded_copy: GradedCopy,
        language: str = 'fr',
        student_name: str = None
    ) -> CopyAnnotations:
        """
        Detect annotation placements for a graded copy.

        This is the main entry point - uses the direct single-pass approach.

        Args:
            pdf_path: Path to the student PDF
            graded_copy: The graded copy with feedback
            language: Language for prompts
            student_name: Optional student name (from CopyDocument)

        Returns:
            CopyAnnotations with all placements
        """
        # Build feedback dict
        feedback_by_question = graded_copy.student_feedback or {}

        if not feedback_by_question:
            return CopyAnnotations(
                copy_id=graded_copy.copy_id,
                student_name=student_name
            )

        # If no provider, fall back to heuristic placement
        if not self.provider:
            return self._heuristic_placement(pdf_path, graded_copy, student_name)

        # Use LLM for intelligent placement
        return self._llm_placement(pdf_path, graded_copy, feedback_by_question, language, student_name)

    def _llm_placement(
        self,
        pdf_path: str,
        graded_copy: GradedCopy,
        feedback_by_question: Dict[str, str],
        language: str,
        student_name: str = None
    ) -> CopyAnnotations:
        """Use LLM to determine annotation placements."""
        import os

        # Build prompt
        prompt = build_direct_annotation_prompt(feedback_by_question, language)

        # Convert PDF to temp image files for vision model
        image_paths = self._pdf_to_images(pdf_path)

        if not image_paths:
            print("Warning: No images extracted from PDF, using heuristic placement")
            return self._heuristic_placement(pdf_path, graded_copy, student_name)

        # Call vision LLM
        try:
            # Use call_vision with image_path (list of temp file paths)
            response = self.provider.call_vision(
                prompt=prompt,
                image_path=image_paths
            )

            # Parse response
            parsed = parse_annotation_response(response)

            return self._build_annotations_from_response(
                graded_copy, parsed, pdf_path, student_name
            )

        except Exception as e:
            # Fall back to heuristic on error
            print(f"LLM placement failed: {e}. Falling back to heuristic.")
            return self._heuristic_placement(pdf_path, graded_copy, student_name)

        finally:
            # Clean up temp files
            for temp_path in image_paths:
                try:
                    os.unlink(temp_path)
                except (FileNotFoundError, PermissionError, OSError):
                    pass  # Cleanup failure is non-critical

    def _heuristic_placement(
        self,
        pdf_path: str,
        graded_copy: GradedCopy,
        student_name: str = None
    ) -> CopyAnnotations:
        """
        Heuristic-based placement when LLM is not available.

        Places feedback in the right margin, one per question.
        """
        annotations = CopyAnnotations(
            copy_id=graded_copy.copy_id,
            student_name=student_name
        )

        # Get page count with proper resource management
        num_pages = 1
        try:
            doc = fitz.open(pdf_path)
            try:
                num_pages = len(doc)
            finally:
                doc.close()
        except (FileNotFoundError, PermissionError, OSError):
            # PDF access failed, use default of 1 page
            pass

        # Distribute questions across pages
        questions = list(graded_copy.student_feedback.keys())
        questions_per_page = max(1, len(questions) // num_pages)

        for i, (q_id, feedback) in enumerate(graded_copy.student_feedback.items()):
            page_num = min(i // max(1, questions_per_page), num_pages - 1)

            # Place in right margin
            y_position = 15.0 + (i % questions_per_page) * 12.0

            annotations.placements.append(AnnotationPlacement(
                question_id=q_id,
                feedback_text=feedback,
                page_number=page_num + 1,  # 1-based page number
                x_percent=70.0,  # Right side
                y_percent=y_position,
                width_percent=25.0,
                height_percent=5.0,
                placement="right_margin",
                confidence=0.5  # Lower confidence for heuristic
            ))

        return annotations

    def _build_annotations_from_response(
        self,
        graded_copy: GradedCopy,
        parsed_response: Dict[str, Any],
        pdf_path: str,
        student_name: str = None
    ) -> CopyAnnotations:
        """Build CopyAnnotations from parsed LLM response."""
        annotations = CopyAnnotations(
            copy_id=graded_copy.copy_id,
            student_name=student_name
        )

        if "error" in parsed_response:
            # Fall back to heuristic
            return self._heuristic_placement(pdf_path, graded_copy, student_name)

        annotation_list = parsed_response.get("annotations", [])

        # Also try annotations by question_id (from zone-based format)
        if not annotation_list:
            ann_by_q = parsed_response.get("annotations", {})
            if isinstance(ann_by_q, dict):
                for q_id, data in ann_by_q.items():
                    annotation_list.append({
                        "question_id": q_id,
                        "feedback_text": data.get("feedback_text", ""),
                        "x_percent": 0,  # Will be derived from zone
                        "y_percent": 0,
                        "placement": data.get("zone_id", ""),
                        "confidence": data.get("confidence", 0.5)
                    })

        # Get page count with proper resource management
        num_pages = 1
        try:
            doc = fitz.open(pdf_path)
            try:
                num_pages = len(doc)
            finally:
                doc.close()
        except (FileNotFoundError, PermissionError, OSError):
            pass

        for ann_data in annotation_list:
            q_id = ann_data.get("question_id", "")

            # Get page number (1-based), default to page 1
            page_num = ann_data.get("page", 1)
            # Ensure page is within valid range
            page_num = max(1, min(page_num, num_pages))

            annotations.placements.append(AnnotationPlacement(
                question_id=q_id,
                feedback_text=ann_data.get("feedback_text", graded_copy.student_feedback.get(q_id, "")),
                page_number=page_num,  # 1-based
                x_percent=float(ann_data.get("x_percent", 70.0)),
                y_percent=float(ann_data.get("y_percent", 20.0)),
                width_percent=float(ann_data.get("width_percent", 25.0)),
                height_percent=float(ann_data.get("height_percent", 5.0)),
                placement=ann_data.get("placement", "right_margin"),
                confidence=float(ann_data.get("confidence", 0.5))
            ))

        return annotations

    def _pdf_to_images(self, pdf_path: str) -> List[str]:
        """
        Convert PDF pages to temp PNG files for vision models.

        Returns list of temp file paths. Caller is responsible for cleanup.
        """
        import tempfile
        import os

        temp_files = []

        try:
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Render page to image (2x zoom for better quality)
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)

                # Save to temp file
                temp_fd, temp_path = tempfile.mkstemp(suffix=f"_page_{page_num}.png")
                os.close(temp_fd)
                pix.save(temp_path)
                temp_files.append(temp_path)

            doc.close()
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            # Clean up any temp files created so far
            for temp_path in temp_files:
                try:
                    os.unlink(temp_path)
                except (FileNotFoundError, PermissionError, OSError):
                    pass  # Cleanup failure is non-critical
            return []

        return temp_files


def percent_to_points(
    percent: float,
    dimension: float,
    margin: float = 10.0
) -> float:
    """Convert percentage to points with margin safety."""
    value = (percent / 100.0) * dimension
    # Ensure within bounds
    return max(margin, min(value, dimension - margin))


def create_annotation_boxes(
    annotations: CopyAnnotations,
    pdf_doc: fitz.Document
) -> Dict[int, List[Tuple[fitz.Rect, str]]]:
    """
    Create annotation boxes for each page.

    Converts 1-based page numbers from annotations to 0-based for PyMuPDF.

    Args:
        annotations: The annotation placements (page_number is 1-based)
        pdf_doc: The PDF document (0-based indexing)

    Returns:
        Dict mapping 0-based page_index -> list of (rect, feedback_text)
    """
    boxes = {}

    for placement in annotations.placements:
        # Convert 1-based page_number to 0-based page_index
        page_index = placement.page_number - 1
        # Ensure within bounds
        if page_index >= len(pdf_doc):
            page_index = len(pdf_doc) - 1
        if page_index < 0:
            page_index = 0

        page = pdf_doc[page_index]
        page_width = page.rect.width
        page_height = page.rect.height

        # Convert percentages to points
        x = percent_to_points(placement.x_percent, page_width)
        y = percent_to_points(placement.y_percent, page_height)
        width = percent_to_points(placement.width_percent, page_width, margin=5)
        height = percent_to_points(placement.height_percent, page_height, margin=3)

        rect = fitz.Rect(x, y, x + width, y + height)

        if page_index not in boxes:
            boxes[page_index] = []
        boxes[page_index].append((rect, placement.feedback_text))

    return boxes
