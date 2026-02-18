"""
Vision module for PDF and image processing.

Provides PDF reading, image conversion, and layout analysis.
"""

from vision.pdf_reader import (
    PDFReader,
    extract_student_answers,
    split_pdf_by_ranges,
)

__all__ = [
    'PDFReader',
    'extract_student_answers',
    'split_pdf_by_ranges',
]
