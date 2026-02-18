"""
PDF reading and processing utilities.

Uses PyMuPDF (fitz) for PDF manipulation and pdfplumber for layout analysis.
"""

import io
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from config.constants import PDF_DPI
from core.exceptions import PDFReadError, InvalidPDFError


class PDFReader:
    """
    PDF reader with support for text extraction and image conversion.

    Handles:
    - Loading PDF files
    - Extracting pages as images
    - Text extraction (for reference)
    - Page counting
    """

    def __init__(self, pdf_path: str):
        """
        Initialize PDF reader.

        Args:
            pdf_path: Path to PDF file

        Raises:
            InvalidPDFError: If file doesn't exist or isn't a PDF
            PDFReadError: If PDF cannot be opened
        """
        self.pdf_path = Path(pdf_path)
        self.doc = None
        self.page_count = 0

        # Validate file exists
        if not self.pdf_path.exists():
            raise InvalidPDFError(f"PDF file not found: {pdf_path}")

        # Validate extension
        if self.pdf_path.suffix.lower() != ".pdf":
            raise InvalidPDFError(f"Not a PDF file: {pdf_path}")

        # Open PDF with error handling
        try:
            self.doc = fitz.open(str(self.pdf_path))
            self.page_count = len(self.doc)
        except Exception as e:
            raise PDFReadError(f"Failed to open PDF: {e}") from e

    def get_page_count(self) -> int:
        """Get the number of pages in the PDF."""
        return self.page_count

    def get_page_image(self, page_num: int, dpi: int = PDF_DPI) -> Image.Image:
        """
        Convert a PDF page to a PIL Image.

        Args:
            page_num: Page number (0-indexed)
            dpi: Resolution for rendering

        Returns:
            PIL Image of the page
        """
        if page_num >= self.page_count:
            raise IndexError(f"Page {page_num} out of range (0-{self.page_count-1})")

        page = self.doc[page_num]

        # Set zoom for DPI
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat)

        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))

        return img

    def get_all_page_images(self, dpi: int = PDF_DPI) -> List[Image.Image]:
        """
        Get all pages as PIL Images.

        Args:
            dpi: Resolution for rendering

        Returns:
            List of PIL Images
        """
        return [self.get_page_image(i, dpi) for i in range(self.page_count)]

    def get_page_image_bytes(self, page_num: int, dpi: int = PDF_DPI) -> bytes:
        """
        Get a page as PNG bytes.

        Args:
            page_num: Page number (0-indexed)
            dpi: Resolution for rendering

        Returns:
            PNG bytes
        """
        img = self.get_page_image(page_num, dpi)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    def extract_text(self, page_num: int = None) -> str:
        """
        Extract text from a page or entire document.

        Args:
            page_num: Page number (None = all pages)

        Returns:
            Extracted text
        """
        if page_num is not None:
            return self.doc[page_num].get_text()

        return self.doc.get_text()

    def get_page_text_blocks(self, page_num: int) -> List[Dict]:
        """
        Get text blocks from a page with position information.

        Args:
            page_num: Page number

        Returns:
            List of text blocks with bounding boxes
        """
        page = self.doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        result = []
        for block in blocks:
            if "lines" in block:  # Text block
                result.append({
                    "bbox": block["bbox"],  # (x0, y0, x1, y1)
                    "text": block.get_text(),
                    "type": "text"
                })
            elif "image" in block:  # Image block
                result.append({
                    "bbox": block["bbox"],
                    "type": "image"
                })

        return result

    def get_page_dimensions(self, page_num: int) -> Tuple[float, float]:
        """
        Get page dimensions in points.

        Args:
            page_num: Page number

        Returns:
            (width, height) in points
        """
        page = self.doc[page_num]
        return (page.rect.width, page.rect.height)

    def find_text(self, text: str, page_num: int = None) -> List[fitz.Rect]:
        """
        Find text locations in the PDF.

        Args:
            text: Text to search for
            page_num: Page number (None = all pages)

        Returns:
            List of bounding rectangles where text was found
        """
        results = []

        pages = [page_num] if page_num is not None else range(self.page_count)

        for pn in pages:
            page = self.doc[pn]
            areas = page.search_for(text)
            results.extend(areas)

        return results

    def get_blank_areas(
        self,
        page_num: int,
        margin: float = 50
    ) -> List[fitz.Rect]:
        """
        Find blank areas on a page suitable for annotation.

        Args:
            page_num: Page number
            margin: Margin from edges in points

        Returns:
            List of blank rectangles
        """
        page = self.doc[page_num]
        width, height = self.get_page_dimensions(page_num)

        # Get all content areas
        content_areas = []
        for block in page.get_text("dict")["blocks"]:
            content_areas.append(fitz.Rect(block["bbox"]))

        # Find margins
        page_rect = page.rect

        # Simple approach: return margins
        blanks = [
            fitz.Rect(0, 0, margin, height),  # Left margin
            fitz.Rect(width - margin, 0, width, height),  # Right margin
        ]

        return blanks

    def close(self):
        """Close the PDF document."""
        if hasattr(self, 'doc'):
            self.doc.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @classmethod
    def validate_pdf(cls, pdf_path: str) -> bool:
        """
        Check if a file is a valid PDF.

        Args:
            pdf_path: Path to file

        Returns:
            True if valid PDF
        """
        path = Path(pdf_path)

        if not path.exists():
            return False

        if path.suffix.lower() != ".pdf":
            return False

        try:
            with fitz.open(str(path)) as doc:
                return len(doc) > 0
        except Exception:
            return False

    @classmethod
    def get_page_count_from_path(cls, pdf_path: str) -> int:
        """Get page count without opening a reader."""
        try:
            with fitz.open(str(pdf_path)) as doc:
                return len(doc)
        except Exception:
            return 0


def extract_student_answers(pdf_path: str) -> Dict[str, str]:
    """
    Extract student answers from a PDF exam copy.

    This is a placeholder for more sophisticated answer extraction.
    In production, this would use vision AI to locate and extract answers.

    Args:
        pdf_path: Path to student's PDF

    Returns:
        Dict mapping question IDs to answer summaries
    """
    # For now, return empty dict - will be filled by AI vision
    return {}


def split_pdf_by_ranges(pdf_path: str, output_dir: str, ranges: List[Tuple[int, int]]) -> List[str]:
    """
    Split a PDF into multiple files based on page ranges.

    Args:
        pdf_path: Input PDF path
        output_dir: Output directory
        ranges: List of (start_page, end_page) tuples (0-indexed, inclusive)

    Returns:
        List of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []

    with fitz.open(pdf_path) as doc:
        for i, (start, end) in enumerate(ranges):
            new_doc = fitz.open()
            try:
                for page_num in range(start, min(end + 1, len(doc))):
                    new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

                output_path = output_dir / f"split_{i}.pdf"
                new_doc.save(str(output_path))
                output_paths.append(str(output_path))
            finally:
                new_doc.close()

    return output_paths
