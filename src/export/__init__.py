"""
Export module for generating graded PDFs and reports.

Post-grading tasks:
- PDF annotation with grades and feedback
- Multi-copy report generation
- Analytics and data export
- Intelligent annotation placement detection
"""

from export.pdf_annotator import PDFAnnotator, BatchAnnotator
from export.analytics import AnalyticsGenerator, DataExporter
from export.annotation_service import (
    AnnotationCoordinateDetector,
    AnnotationPlacement,
    CopyAnnotations,
    create_annotation_boxes,
)

__all__ = [
    'PDFAnnotator',
    'BatchAnnotator',
    'AnalyticsGenerator',
    'DataExporter',
    'AnnotationCoordinateDetector',
    'AnnotationPlacement',
    'CopyAnnotations',
    'create_annotation_boxes',
]
