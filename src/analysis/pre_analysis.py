"""
PDF Pre-Analysis module.

This module provides functionality to analyze PDFs before grading to:
- Validate the PDF contains student copies
- Detect document structure (one student or multiple)
- Detect grading scale / barème
- Identify blocking issues
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.models import (
    PreAnalysisResult,
    StudentInfo,
    DocumentType,
    PDFStructure,
    SubjectIntegration,
)
from core.exceptions import (
    PDFReadError,
    InvalidPDFError,
    AnalysisError,
)
from vision.pdf_reader import PDFReader
from ai.provider_factory import create_ai_provider
from analysis.pre_analysis_prompts import build_pre_analysis_prompt
from analysis.pre_analysis_translations import get_translations

logger = logging.getLogger(__name__)


class PreAnalyzer:
    """
    Analyzes PDF documents before grading.

    This class performs a quick analysis of the PDF to:
    1. Validate it contains student copies
    2. Detect document structure
    3. Detect grading scale
    4. Identify blocking issues

    Results can be cached to avoid re-analysis.
    """

    def __init__(
        self,
        user_id: str,
        session_id: str,
        language: str = "fr",
        cache_dir: Path = None,
        provider = None  # Optional: pass existing provider for token tracking
    ):
        """
        Initialize the pre-analyzer.

        Args:
            user_id: User ID for storage
            session_id: Session ID for caching
            language: Language for prompts (fr, en)
            cache_dir: Directory for caching results
            provider: Optional AI provider (if None, creates new one)
        """
        self.user_id = user_id
        self.session_id = session_id
        self.language = language
        self.cache_dir = cache_dir or Path(f"data/{user_id}/{session_id}/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Use provided provider or create new one
        self.provider = provider if provider else create_ai_provider()

    def analyze(
        self,
        pdf_path: str,
        force_refresh: bool = False
    ) -> PreAnalysisResult:
        """
        Analyze a PDF document.

        Args:
            pdf_path: Path to the PDF file
            force_refresh: Force re-analysis even if cached

        Returns:
            PreAnalysisResult with detected information

        Raises:
            InvalidPDFError: If PDF is invalid
            AnalysisError: If analysis fails
        """
        start_time = time.time()

        # Check cache
        cache_key = self._get_cache_key(pdf_path)
        if not force_refresh:
            cached = self._load_from_cache(cache_key)
            if cached:
                cached.cached = True
                logger.info(f"Loaded pre-analysis from cache: {cache_key}")
                return cached

        # Validate and read PDF
        try:
            pdf_reader = PDFReader(pdf_path)
            page_count = pdf_reader.get_page_count()
        except (InvalidPDFError, PDFReadError) as e:
            raise e
        except Exception as e:
            raise InvalidPDFError(f"Failed to open PDF: {e}") from e

        # Sample pages for analysis
        # For efficiency, we don't analyze all pages if PDF is large
        sample_pages = self._get_sample_pages(page_count)
        sample_images = []

        for page_num in sample_pages:
            try:
                img = pdf_reader.get_page_image(page_num)
                sample_images.append((page_num, img))
            except Exception as e:
                logger.warning(f"Failed to get page {page_num}: {e}")

        if not sample_images:
            raise AnalysisError("Could not extract any pages from PDF")

        # Build prompt
        prompt = build_pre_analysis_prompt(self.language)

        # Call AI with images
        try:
            # For the first pass, use the first few sample images
            analysis_images = sample_images[:min(5, len(sample_images))]

            # Convert PIL images to format expected by provider
            image_paths = []
            for page_num, img in analysis_images:
                # Save temporarily and pass path
                temp_path = self.cache_dir / f"temp_page_{page_num}.png"
                img.save(str(temp_path))
                image_paths.append(str(temp_path))

            # Call vision API with multiple images
            response = self._call_vision_with_multiple_images(prompt, image_paths)

            # Parse response
            result = self._parse_response(response, page_count)

            # Calculate duration
            result.analysis_duration_ms = (time.time() - start_time) * 1000

            # Save to cache
            self._save_to_cache(cache_key, result)

            # Cleanup temp files
            for path in image_paths:
                try:
                    Path(path).unlink()
                except Exception:
                    pass

            return result

        except Exception as e:
            logger.error(f"Pre-analysis failed: {e}")
            raise AnalysisError(f"Pre-analysis failed: {e}") from e

        finally:
            pdf_reader.close()

    def _call_vision_with_multiple_images(
        self,
        prompt: str,
        image_paths: List[str]
    ) -> str:
        """
        Call vision API with multiple images.

        Args:
            prompt: The prompt to send
            image_paths: List of image paths

        Returns:
            API response text
        """
        if not hasattr(self.provider, 'call_vision'):
            raise AnalysisError("Provider does not support vision calls")

        # Gemini supports multiple images in a single call
        # Pass all images as a list
        if len(image_paths) == 1:
            return self.provider.call_vision(prompt, image_path=image_paths[0])

        # For multiple images, pass the list to the provider
        # The Gemini provider handles lists of images
        combined_prompt = f"{prompt}\n\nVoici les {len(image_paths)} pages du document à analyser:"
        return self.provider.call_vision(combined_prompt, image_path=image_paths)

    def _get_sample_pages(self, page_count: int) -> List[int]:
        """
        Get sample page indices for analysis.

        For large PDFs, we sample strategically:
        - First pages (usually subject)
        - Middle pages (sample of students)
        - Last pages

        Args:
            page_count: Total number of pages

        Returns:
            List of 0-indexed page numbers to sample
        """
        if page_count <= 10:
            # Small PDF: analyze all pages
            return list(range(page_count))

        if page_count <= 20:
            # Medium PDF: first 3, middle 3, last 3
            return [
                0, 1, 2,  # First pages
                page_count // 2 - 1, page_count // 2, page_count // 2 + 1,  # Middle
                page_count - 3, page_count - 2, page_count - 1  # Last pages
            ]

        # Large PDF: sample more strategically
        samples = []

        # First 3 pages (subject)
        samples.extend([0, 1, 2])

        # Sample every ~5 pages for the rest
        step = max(5, page_count // 10)
        for i in range(3, page_count - 3, step):
            samples.append(i)

        # Last 2 pages
        samples.extend([page_count - 2, page_count - 1])

        return list(set(samples))  # Remove duplicates

    def _parse_response(
        self,
        response: str,
        page_count: int
    ) -> PreAnalysisResult:
        """
        Parse AI response into PreAnalysisResult.

        Args:
            response: Raw AI response
            page_count: Number of pages in PDF

        Returns:
            PreAnalysisResult
        """
        # Extract JSON from response
        try:
            # Try to find JSON in the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            # Return a default result with low confidence
            return PreAnalysisResult(
                is_valid_pdf=True,
                page_count=page_count,
                document_type=DocumentType.UNCLEAR,
                confidence_document_type=0.0,
                structure=PDFStructure.AMBIGUOUS,
                blocking_issues=["Impossible d'analyser la réponse de l'IA"],
                has_blocking_issues=True,
            )

        # Parse students
        students = []
        for s in data.get("students", []):
            students.append(StudentInfo(
                index=s.get("index", 0),
                name=s.get("name"),
                start_page=s.get("start_page", 1),
                end_page=s.get("end_page", 1),
                confidence=s.get("confidence", 0.5),
            ))

        # Map document type
        doc_type_str = data.get("document_type", "unclear")
        doc_type_map = {
            "student_copies": DocumentType.STUDENT_COPIES,
            "subject_only": DocumentType.SUBJECT_ONLY,
            "random_document": DocumentType.RANDOM_DOCUMENT,
            "unclear": DocumentType.UNCLEAR,
        }
        document_type = doc_type_map.get(doc_type_str, DocumentType.UNCLEAR)

        # Map structure
        structure_str = data.get("structure", "ambiguous")
        structure_map = {
            "one_pdf_one_student": PDFStructure.ONE_PDF_ONE_STUDENT,
            "one_pdf_all_students": PDFStructure.ONE_PDF_ALL_STUDENTS,
            "ambiguous": PDFStructure.AMBIGUOUS,
        }
        structure = structure_map.get(structure_str, PDFStructure.AMBIGUOUS)

        # Map subject integration
        subject_str = data.get("subject_integration", "not_detected")
        subject_map = {
            "integrated": SubjectIntegration.INTEGRATED,
            "separate": SubjectIntegration.SEPARATE,
            "not_detected": SubjectIntegration.NOT_DETECTED,
        }
        subject_integration = subject_map.get(subject_str, SubjectIntegration.NOT_DETECTED)

        # Parse candidate scales (new feature for multi-scale detection)
        candidate_scales = []
        raw_candidate_scales = data.get("candidate_scales", [])

        if raw_candidate_scales:
            # AI returned multiple candidate scales
            for candidate in raw_candidate_scales:
                candidate_scales.append({
                    "scale": candidate.get("scale", {}),
                    "confidence": candidate.get("confidence", 0.0)
                })
        else:
            # Backward compatibility: if single scale returned, store as first candidate
            single_scale = data.get("grading_scale", {})
            single_confidence = data.get("confidence_grading_scale", 0.5)
            if single_scale:
                candidate_scales.append({
                    "scale": single_scale,
                    "confidence": single_confidence
                })

        # Determine primary grading scale and confidence
        grading_scale = data.get("grading_scale", {})
        confidence_grading_scale = data.get("confidence_grading_scale", 0.5)

        # If candidate_scales is populated but grading_scale is empty, use best candidate
        if not grading_scale and candidate_scales:
            # Sort by confidence and use the best one
            candidate_scales.sort(key=lambda x: x["confidence"], reverse=True)
            best_candidate = candidate_scales[0]
            grading_scale = best_candidate["scale"]
            confidence_grading_scale = best_candidate["confidence"]

        # Determine blocking issues
        blocking_issues = data.get("blocking_issues", [])
        has_blocking_issues = bool(blocking_issues) or document_type in [
            DocumentType.RANDOM_DOCUMENT,
            DocumentType.SUBJECT_ONLY,
        ]

        # Add automatic blocking issues based on document type
        if document_type == DocumentType.RANDOM_DOCUMENT:
            t = get_translations(self.language)
            blocking_issues.append(t["ui"]["blocking_issue_detected"])
        elif document_type == DocumentType.SUBJECT_ONLY:
            blocking_issues.append("Le document contient uniquement le sujet, pas de copies d'élèves")

        # Build result with candidate_scales
        result = PreAnalysisResult(
            is_valid_pdf=True,
            page_count=page_count,
            document_type=document_type,
            confidence_document_type=data.get("confidence_document_type", 0.5),
            structure=structure,
            subject_integration=subject_integration,
            num_students_detected=data.get("num_students_detected", len(students)),
            students=students,
            grading_scale=grading_scale,
            confidence_grading_scale=confidence_grading_scale,
            questions_detected=data.get("questions_detected", []),
            quality_issues=data.get("quality_issues", []),
            overall_quality_score=data.get("overall_quality_score", 1.0),
            blocking_issues=blocking_issues,
            has_blocking_issues=has_blocking_issues,
            warnings=data.get("warnings", []),
            detected_language=data.get("detected_language", self.language),
        )

        # Add candidate_scales to the result (stored in extra dict or as attribute)
        # Since PreAnalysisResult is a Pydantic model, we can add it to model_dump
        # For now, we'll monkey-patch it onto the instance
        result.candidate_scales = candidate_scales

        return result

    def _get_cache_key(self, pdf_path: str) -> str:
        """Generate cache key based on PDF hash."""
        # Use file size and modification time for quick hash
        path = Path(pdf_path)
        stat = path.stat()
        hash_input = f"{pdf_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[PreAnalysisResult]:
        """Load result from cache if exists."""
        cache_file = self.cache_dir / f"pre_analysis_{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            # Reconstruct PreAnalysisResult
            students = [StudentInfo(**s) for s in data.get("students", [])]
            candidate_scales = data.get("candidate_scales", [])

            result = PreAnalysisResult(
                is_valid_pdf=data.get("is_valid_pdf", True),
                page_count=data.get("page_count", 0),
                document_type=DocumentType(data.get("document_type", "unclear")),
                confidence_document_type=data.get("confidence_document_type", 0.0),
                structure=PDFStructure(data.get("structure", "ambiguous")),
                subject_integration=SubjectIntegration(data.get("subject_integration", "not_detected")),
                num_students_detected=data.get("num_students_detected", 0),
                students=students,
                grading_scale=data.get("grading_scale", {}),
                confidence_grading_scale=data.get("confidence_grading_scale", 0.0),
                candidate_scales=candidate_scales,
                questions_detected=data.get("questions_detected", []),
                quality_issues=data.get("quality_issues", []),
                overall_quality_score=data.get("overall_quality_score", 1.0),
                blocking_issues=data.get("blocking_issues", []),
                has_blocking_issues=data.get("has_blocking_issues", False),
                warnings=data.get("warnings", []),
                detected_language=data.get("detected_language", "fr"),
                analysis_id=data.get("analysis_id", ""),
                cached=True,
                analysis_duration_ms=data.get("analysis_duration_ms", 0.0),
            )
            return result

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_to_cache(self, cache_key: str, result: PreAnalysisResult):
        """Save result to cache."""
        cache_file = self.cache_dir / f"pre_analysis_{cache_key}.json"

        try:
            data = result.model_dump()
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")


def quick_analyze_pdf(
    pdf_path: str,
    language: str = "fr"
) -> Dict[str, Any]:
    """
    Perform a quick analysis of a PDF.

    This is a convenience function for one-off analysis.

    Args:
        pdf_path: Path to PDF file
        language: Language for prompts

    Returns:
        Dict with basic analysis results
    """
    try:
        pdf_reader = PDFReader(pdf_path)
        page_count = pdf_reader.get_page_count()
        pdf_reader.close()

        return {
            "is_valid": True,
            "page_count": page_count,
            "estimated_students": max(1, page_count // 2),  # Rough estimate
        }

    except Exception as e:
        return {
            "is_valid": False,
            "error": str(e),
        }
