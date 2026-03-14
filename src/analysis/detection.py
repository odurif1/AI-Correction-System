"""
PDF Detection module.

This module provides functionality to detect PDF structure before grading:
- Validate the PDF contains student copies
- Detect document structure (one student or multiple)
- Detect grading scale / barème
- Identify blocking issues
"""

import hashlib
import json
import logging
import time
import re
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.models import (
    DetectionResult,
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
from analysis.detection_prompts import build_detection_prompt
from analysis.detection_translations import (
    get_translations,
    translate_detection_message,
    translate_quality_issue,
)

logger = logging.getLogger(__name__)


GENERIC_SECTION_LABELS = {
    "exercice",
    "exercise",
    "question",
    "partie",
    "part",
    "section",
}


def _normalize_text_label(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _ascii_normalized_lower(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_only).strip().lower()


def _parse_points_value(value: Any) -> float:
    try:
        if isinstance(value, str):
            match = re.search(r"[\d.]+", value)
            if match:
                return float(match.group())
            return 1.0
        return float(value)
    except (ValueError, TypeError):
        return 1.0


def _normalize_question_key(raw_key: Any) -> tuple[Optional[str], Optional[str]]:
    """Return a stable internal key and an optional warning message."""
    if raw_key is None:
        return None, "Une entrée du barème sans identifiant a été ignorée."

    label = _normalize_text_label(str(raw_key))
    if not label:
        return None, "Une entrée vide du barème a été ignorée."

    lowered = _ascii_normalized_lower(label)
    if lowered in GENERIC_SECTION_LABELS:
        return None, f"L'entrée de barème '{label}' est trop vague et a été ignorée."

    question_match = re.fullmatch(r"(?:q|question)\s*[:#\- ]*\s*0*(\d+)", lowered)
    if question_match:
        return f"Q{int(question_match.group(1))}", None

    numeric_match = re.fullmatch(r"0*(\d+)", lowered)
    if numeric_match:
        return f"Q{int(numeric_match.group(1))}", None

    # Keep richer labels as-is after light normalization to preserve exam diversity.
    return label, None


def _normalize_grading_scale(
    grading_scale: Dict[str, Any],
    questions_detected: List[Any],
) -> tuple[Dict[str, float], List[str], List[str]]:
    """Normalize grading keys conservatively while preserving richer labels."""
    normalized_scale: Dict[str, float] = {}
    normalized_questions: List[str] = []
    warnings: List[str] = []

    def register_question(raw_label: Any) -> None:
        normalized_key, warning = _normalize_question_key(raw_label)
        if warning and warning not in warnings:
            warnings.append(warning)
        if normalized_key and normalized_key not in normalized_questions:
            normalized_questions.append(normalized_key)

    for raw_key, raw_value in grading_scale.items():
        normalized_key, warning = _normalize_question_key(raw_key)
        if warning and warning not in warnings:
            warnings.append(warning)
        if not normalized_key:
            continue

        points_value = _parse_points_value(raw_value)
        if normalized_key in normalized_scale and normalized_scale[normalized_key] != points_value:
            warnings.append(
                f"Plusieurs valeurs ont été détectées pour '{normalized_key}'. La première a été conservée."
            )
            continue

        normalized_scale.setdefault(normalized_key, points_value)
        if normalized_key not in normalized_questions:
            normalized_questions.append(normalized_key)

    for raw_question in questions_detected:
        register_question(raw_question)

    if not normalized_questions:
        normalized_questions = list(normalized_scale.keys())

    return normalized_scale, normalized_questions, warnings


class Detector:
    """
    Detects PDF document structure before grading.

    This class performs detection of the PDF to:
    1. Validate it contains student copies
    2. Detect document structure
    3. Detect grading scale
    4. Identify blocking issues
    """

    def __init__(
        self,
        user_id: str,
        session_id: str,
        language: str = "fr",
        cache_dir: Path = None,
        provider = None
    ):
        """
        Initialize the detector.

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

    def detect(
        self,
        pdf_path: str,
        mode: str = "interactive",
        force_refresh: bool = False
    ) -> DetectionResult:
        """
        Detect PDF document structure.

        Args:
            pdf_path: Path to the PDF file
            mode: Detection mode ("interactive" or "auto")
            force_refresh: Force re-detection even if cached

        Returns:
            DetectionResult with detected information

        Raises:
            InvalidPDFError: If PDF is invalid
            AnalysisError: If detection fails
        """
        start_time = time.time()

        # Check cache
        cache_key = self._get_cache_key(pdf_path)
        if not force_refresh:
            cached = self._load_from_cache(cache_key)
            if cached:
                logger.info(f"Loaded detection from cache: {cache_key}")
                return cached

        # Validate and read PDF
        try:
            pdf_reader = PDFReader(pdf_path)
            page_count = pdf_reader.get_page_count()
        except (InvalidPDFError, PDFReadError) as e:
            raise e
        except Exception as e:
            raise InvalidPDFError(f"Failed to open PDF: {e}") from e

        # Sample pages for detection
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
        prompt = build_detection_prompt(self.language)

        # Call AI with images
        try:
            # Use the full document for structure and scale detection.
            detection_images = sample_images

            # Convert PIL images to format expected by provider
            image_paths = []
            for page_num, img in detection_images:
                # Save temporarily and pass path
                temp_path = self.cache_dir / f"temp_page_{page_num}.png"
                img.save(str(temp_path))
                image_paths.append(str(temp_path))

            # Call vision API with multiple images
            response = self._call_vision_with_multiple_images(prompt, image_paths)

            # Parse response
            result = self._parse_response(response, page_count, mode)

            # Calculate duration
            result.detection_duration_ms = (time.time() - start_time) * 1000

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
            logger.error(f"Detection failed: {e}")
            raise AnalysisError(f"Detection failed: {e}") from e

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
        if len(image_paths) == 1:
            return self.provider.call_vision(prompt, image_path=image_paths[0])

        # For multiple images, pass the list to the provider
        combined_prompt = f"{prompt}\n\nVoici les {len(image_paths)} pages du document à analyser:"
        return self.provider.call_vision(combined_prompt, image_path=image_paths)

    def _get_sample_pages(self, page_count: int) -> List[int]:
        """
        Get page indices for detection.

        Returns all pages — the cost is marginal since grading will
        process them all anyway, and full coverage gives accurate
        student detection (names + boundaries).
        """
        return list(range(page_count))

    def _parse_response(
        self,
        response: str,
        page_count: int,
        mode: str = "interactive"
    ) -> DetectionResult:
        """
        Parse AI response into DetectionResult.

        Args:
            response: Raw AI response
            page_count: Number of pages in PDF
            mode: Detection mode

        Returns:
            DetectionResult
        """
        # Extract JSON from response
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return DetectionResult(
                mode=mode,
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

        normalized_questions_detected, question_warnings = [], []

        # Parse candidate scales
        candidate_scales = []
        raw_candidate_scales = data.get("candidate_scales", [])

        if raw_candidate_scales:
            for candidate in raw_candidate_scales:
                normalized_candidate_scale, _, candidate_warnings = _normalize_grading_scale(
                    candidate.get("scale", {}),
                    [],
                )
                question_warnings.extend(
                    warning for warning in candidate_warnings if warning not in question_warnings
                )
                candidate_scales.append({
                    "scale": normalized_candidate_scale,
                    "confidence": candidate.get("confidence", 0.0)
                })
        else:
            single_scale = data.get("grading_scale", {})
            single_confidence = data.get("confidence_grading_scale", 0.5)
            if single_scale:
                normalized_single_scale, _, single_warnings = _normalize_grading_scale(single_scale, [])
                question_warnings.extend(
                    warning for warning in single_warnings if warning not in question_warnings
                )
                candidate_scales.append({
                    "scale": normalized_single_scale,
                    "confidence": single_confidence
                })

        # Determine primary grading scale and confidence
        grading_scale, normalized_questions_detected, primary_warnings = _normalize_grading_scale(
            data.get("grading_scale", {}),
            data.get("questions_detected", []),
        )
        question_warnings.extend(
            warning for warning in primary_warnings if warning not in question_warnings
        )
        confidence_grading_scale = data.get("confidence_grading_scale", 0.5)

        if not grading_scale and candidate_scales:
            candidate_scales.sort(key=lambda x: x["confidence"], reverse=True)
            best_candidate = candidate_scales[0]
            grading_scale = best_candidate["scale"]
            confidence_grading_scale = best_candidate["confidence"]
            if not normalized_questions_detected:
                normalized_questions_detected = list(grading_scale.keys())

        if question_warnings:
            penalty = min(0.35, 0.1 + (0.05 * len(question_warnings)))
            confidence_grading_scale = max(0.2, confidence_grading_scale - penalty)

        # Determine blocking issues
        blocking_issues = [
            translate_detection_message(issue, self.language)
            for issue in data.get("blocking_issues", [])
        ]
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

        # Build result
        translated_quality_issues = [
            translate_quality_issue(issue, self.language)
            for issue in data.get("quality_issues", [])
        ]
        translated_warnings = [
            translate_detection_message(warning, self.language)
            for warning in data.get("warnings", [])
        ]

        result = DetectionResult(
            mode=mode,
            is_valid_pdf=True,
            page_count=page_count,
            document_type=document_type,
            confidence_document_type=data.get("confidence_document_type", 0.5),
            structure=structure,
            subject_integration=subject_integration,
            num_students_detected=data.get("num_students_detected", len(students)),
            students=students,
            pages_per_student=data.get("pages_per_student"),
            consistent_pages_per_student=data.get("consistent_pages_per_student", False),
            subject_page_count=data.get("subject_page_count", 0),
            grading_scale=grading_scale,
            confidence_grading_scale=confidence_grading_scale,
            questions_detected=normalized_questions_detected,
            quality_issues=translated_quality_issues,
            overall_quality_score=data.get("overall_quality_score", 1.0),
            blocking_issues=blocking_issues,
            has_blocking_issues=has_blocking_issues,
            warnings=translated_warnings + question_warnings,
            detected_language=data.get("detected_language", self.language),
            exam_name=data.get("exam_name"),
        )

        # Add candidate_scales
        result.candidate_scales = candidate_scales

        return result

    def _get_cache_key(self, pdf_path: str) -> str:
        """Generate cache key based on PDF hash."""
        path = Path(pdf_path)
        stat = path.stat()
        hash_input = f"{pdf_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[DetectionResult]:
        """Load result from cache if exists."""
        cache_file = self.cache_dir / f"detection_{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            students = [StudentInfo(**s) for s in data.get("students", [])]
            candidate_scales = data.get("candidate_scales", [])

            result = DetectionResult(
                mode=data.get("mode", "interactive"),
                is_valid_pdf=data.get("is_valid_pdf", True),
                page_count=data.get("page_count", 0),
                document_type=DocumentType(data.get("document_type", "unclear")),
                confidence_document_type=data.get("confidence_document_type", 0.0),
                structure=PDFStructure(data.get("structure", "ambiguous")),
                subject_integration=SubjectIntegration(data.get("subject_integration", "not_detected")),
                num_students_detected=data.get("num_students_detected", 0),
                students=students,
                pages_per_student=data.get("pages_per_student"),
                consistent_pages_per_student=data.get("consistent_pages_per_student", False),
                subject_page_count=data.get("subject_page_count", 0),
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
                detection_id=data.get("detection_id", ""),
                detection_duration_ms=data.get("detection_duration_ms", 0.0),
                exam_name=data.get("exam_name"),
            )
            return result

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_to_cache(self, cache_key: str, result: DetectionResult):
        """Save result to cache."""
        cache_file = self.cache_dir / f"detection_{cache_key}.json"

        try:
            data = result.model_dump()
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")


def quick_detect_pdf(
    pdf_path: str,
    language: str = "fr"
) -> Dict[str, Any]:
    """
    Perform a quick detection of a PDF.

    Args:
        pdf_path: Path to PDF file
        language: Language for prompts

    Returns:
        Dict with basic detection results
    """
    try:
        pdf_reader = PDFReader(pdf_path)
        page_count = pdf_reader.get_page_count()
        pdf_reader.close()

        return {
            "is_valid": True,
            "page_count": page_count,
            "estimated_students": max(1, page_count // 2),
        }

    except Exception as e:
        return {
            "is_valid": False,
            "error": str(e),
        }
