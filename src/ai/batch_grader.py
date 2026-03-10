"""
Batch Grader - Grades all copies in a single API call.

Instead of grading each copy individually, this module sends all copies
to the LLM at once, allowing:
- Consistency across all copies (same answer = same grade)
- Pattern detection (clustering, outliers)
- 90%+ reduction in API calls

Architecture:
- Batches of N copies (configurable via --batch-size)
- Each batch is one API call
- Dual LLM mode: 2 parallel batch calls
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union

from config.settings import get_settings
from config.constants import MAX_RETRIES, RETRY_BASE_DELAY, GRADE_AGREEMENT_THRESHOLD
from core.exceptions import OutputTruncatedError
from utils.json_extractor import extract_json_from_response
from prompts.batch import (
    build_batch_grading_prompt,
    build_dual_llm_verification_prompt,
    build_ultimatum_prompt,
    build_common_prefix,
)

logger = logging.getLogger(__name__)

# Status codes that are retryable
RETRYABLE_STATUS_CODES = {503, 429, 500, 502, 504}


def _is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable (503, 429, etc.)."""
    error_str = str(error).lower()
    # Check for status codes in error message
    for code in RETRYABLE_STATUS_CODES:
        if str(code) in error_str:
            return True
    # Check for common retryable error messages
    retryable_messages = ['unavailable', 'overloaded', 'rate limit', 'timeout', 'temporary']
    return any(msg in error_str for msg in retryable_messages)


async def _call_provider_vision(
    provider,
    prompt: str,
    images: Optional[List[str]] = None,
    max_retries: int = MAX_RETRIES # Kept for signature compatibility but unused as tenacity handles retries
) -> Optional[str]:
    """
    Call a provider's vision API.
    Retries are now handled dynamically by the provider using tenacity.

    Args:
        provider: LLM provider instance
        prompt: The prompt to send
        images: Optional list of image paths
        max_retries: Ignored

    Returns:
        Raw response as string, or None if failed
    """
    try:
        # Run synchronous call in a separate thread to avoid blocking the event loop
        raw_response = await asyncio.to_thread(
            provider.call_vision,
            prompt, 
            image_path=images or [], 
            response_format="json"
        )
        if not isinstance(raw_response, str):
            raw_response = str(raw_response)
        return raw_response
    except OutputTruncatedError:
        raise  # Let truncation errors propagate
    except Exception as e:
        logger.error(f"Provider vision call failed after internal retries: {e}")
        return None


def get_agreement_threshold(max_points: float) -> float:
    """
    Calculate the relative agreement threshold based on max_points.

    Args:
        max_points: Maximum points for the question

    Returns:
        Absolute threshold value (e.g., 0.1 for 10% of 1 point)
    """
    return max_points * get_settings().grade_agreement_threshold


def _parse_grade_value(value) -> float:
    """
    Parse a grade value that may be in various formats.

    Handles:
    - float/int: 1.0, 1
    - string with fraction: "1/1", "0.5/2"
    - string with just number: "1.0", "1"

    Args:
        value: The grade value to parse

    Returns:
        Float value of the grade (numerator only, not the fraction result)
    """
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        value = value.strip()
        # Handle fraction format like "1/1" or "0.5/2"
        if '/' in value:
            try:
                numerator = value.split('/')[0].strip()
                return float(numerator)
            except (ValueError, IndexError):
                pass
        # Try direct float conversion
        try:
            return float(value)
        except ValueError:
            pass

    return 0.0


def get_flip_flop_threshold(max_points: float) -> float:
    """
    Calculate the relative flip-flop detection threshold based on max_points.

    Args:
        max_points: Maximum points for the question

    Returns:
        Absolute threshold value for detecting significant position swaps
    """
    return max_points * get_settings().flip_flop_threshold


@dataclass
class BatchCopyResult:
    """Result for a single copy within a batch."""
    copy_index: int
    student_name: Optional[str]
    questions: Dict[str, Dict[str, Any]]  # {Q1: {grade, reading, feedback, ...}}
    overall_feedback: str = ""
    image_paths: List[str] = field(default_factory=list)  # Paths to copy images

    # Position in original PDF (1-based page numbers) for page-based matching
    pdf_start_page: Optional[int] = None
    pdf_end_page: Optional[int] = None

    @property
    def page_range(self) -> Optional[Tuple[int, int]]:
        """Return page range as tuple, or None if not available."""
        if self.pdf_start_page is not None and self.pdf_end_page is not None:
            return (self.pdf_start_page, self.pdf_end_page)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "copy_index": self.copy_index,
            "student_name": self.student_name,
            "questions": self.questions,
            "overall_feedback": self.overall_feedback,
            "image_paths": self.image_paths,
            "pdf_start_page": self.pdf_start_page,
            "pdf_end_page": self.pdf_end_page
        }


@dataclass
class BatchResult:
    """Complete result from batch grading."""
    copies: List[BatchCopyResult]
    patterns: Dict[str, Any]  # Detected patterns across copies
    raw_response: str
    parse_success: bool
    parse_errors: List[str]
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "copies": [c.to_dict() for c in self.copies],
            "patterns": self.patterns,
            "parse_success": self.parse_success,
            "parse_errors": self.parse_errors,
            "duration_ms": self.duration_ms
        }


class BatchGrader:
    """
    Grade all copies in batches using a single API call per batch.

    Usage:
        grader = BatchGrader(provider)
        result = await grader.grade_batch(copies, questions, language="fr")

    For verification/ultimatum phases with caching, use CacheManager instead.
    """

    def __init__(self, provider):
        """
        Initialize batch grader.

        Args:
            provider: LLM provider (must support multi-image calls)
        """
        self.provider = provider

    async def grade_batch(
        self,
        copies: List[Dict[str, Any]],
        questions: Dict[str, Dict[str, Any]],
        language: str = "fr",
        detect_students: bool = False,
        common_prefix: str = None,
        detection_hints: Dict[str, Any] = None
    ) -> BatchResult:
        """
        Grade a batch of copies in a single API call.

        Args:
            copies: List of copy data dicts with:
                - copy_index: 1-based index
                - image_paths: List of image paths for this copy
                - (optional) student_name: Pre-detected name
            questions: Dict of {question_id: {text, criteria, max_points}}
            language: Language for prompts
            detect_students: If True, ask LLM to detect multiple students in single PDF
            common_prefix: Optional pre-built common prefix for implicit caching.
                          When provided, enables Gemini caching across all phases.

        Returns:
            BatchResult with all copy grades and detected patterns
        """
        start_time = time.time()

        # Build prompt with common prefix for caching
        prompt = build_batch_grading_prompt(
            copies, questions, language, detect_students,
            common_prefix=common_prefix,
            detection_hints=detection_hints
        )

        # Collect all images from all copies
        all_images = []
        for copy in copies:
            all_images.extend(copy.get('image_paths', []))

        # Call LLM with all images (with retry logic)
        raw_response = await _call_provider_vision(
            self.provider, prompt, all_images, max_retries=MAX_RETRIES
        )
        if raw_response is None:
            logger.error("Batch grading API call failed after all retries")
            return BatchResult(
                copies=[],
                patterns={},
                raw_response="",
                parse_success=False,
                parse_errors=["API call failed after all retries"],
                duration_ms=(time.time() - start_time) * 1000
            )

        # Parse response
        result = self._parse_batch_response(raw_response, copies, start_time)

        return result

    def _parse_batch_response(
        self,
        raw_response: str,
        copies: List[Dict[str, Any]],
        start_time: float
    ) -> BatchResult:
        """Parse the LLM response into structured BatchResult."""

        parse_errors = []
        copies_results = []
        patterns = {}

        # Collect ALL images from all input copies
        # In batch mode with student detection, all images belong to all detected copies
        all_input_images = []
        for c in copies:
            all_input_images.extend(c.get('image_paths', []))

        # Build lookup for input copies by copy_index (for page info)
        copies_by_index = {c.get('copy_index'): c for c in copies}

        # Try to extract JSON from response
        data = extract_json_from_response(raw_response)
        if data is None:
            parse_errors.append("No JSON object found in response")
            logger.warning(f"Failed to parse JSON. Response length: {len(raw_response) if raw_response else 0}")
            logger.debug(f"Raw response (first 500 chars): {raw_response[:500] if raw_response else 'EMPTY'}")
        else:
            try:
                # Parse copies
                for copy_data in data.get('copies', []):
                    copy_index = copy_data.get('copy_index', 0)
                    student_name = copy_data.get('student_name')

                    # Parse questions
                    questions = {}
                    for qid, qdata in copy_data.get('questions', {}).items():
                        # Skip if qdata is None (malformed response)
                        if qdata is None:
                            logger.warning(f"Question {qid} has None data, skipping")
                            continue
                        questions[qid] = {
                            'student_answer_read': qdata.get('student_answer_read', ''),
                            'grade': _parse_grade_value(qdata.get('grade', 0)),
                            'max_points': _parse_grade_value(qdata.get('max_points', 1)),
                            'confidence': float(qdata.get('confidence', 0.8)),
                            'reasoning': qdata.get('reasoning', ''),
                            'feedback': qdata.get('feedback', '')
                        }

                    # Get page info from input copy if available
                    input_copy = copies_by_index.get(copy_index, {})
                    pdf_start_page = input_copy.get('start_page')
                    pdf_end_page = input_copy.get('end_page')

                    # In batch mode, all input images are shared across detected copies
                    # (the LLM sees all pages and detects students from them)
                    copies_results.append(BatchCopyResult(
                        copy_index=copy_index,
                        student_name=student_name,
                        questions=questions,
                        overall_feedback=copy_data.get('overall_feedback', ''),
                        image_paths=all_input_images,  # All copies see all images
                        pdf_start_page=pdf_start_page,
                        pdf_end_page=pdf_end_page
                    ))

                # Parse patterns
                patterns = data.get('patterns', {})

            except (KeyError, TypeError, ValueError) as e:
                parse_errors.append(f"Parsing error: {str(e)}")

        # Ensure we have results for all input copies
        if len(copies_results) < len(copies):
            parse_errors.append(
                f"Only {len(copies_results)}/{len(copies)} copies parsed"
            )

        return BatchResult(
            copies=copies_results,
            patterns=patterns,
            raw_response=raw_response[:5000] if raw_response else "",
            parse_success=len(parse_errors) == 0 and len(copies_results) > 0,
            parse_errors=parse_errors,
            duration_ms=(time.time() - start_time) * 1000
        )


def _should_detect_students(copies: List[Dict[str, Any]]) -> bool:
    """
    Determine if student detection should be enabled.

    Returns True if:
    - Only 1 "copy" is passed (meaning 1 PDF file)
    - No student name is pre-detected

    This indicates a single PDF that may contain multiple students.
    """
    if len(copies) != 1:
        return False
    # Check if the single copy has no pre-detected student name
    first_copy = copies[0]
    return not first_copy.get('student_name')


async def grade_all_copies_in_batches(
    provider,
    copies: List[Dict[str, Any]],
    questions: Dict[str, Dict[str, Any]],
    max_pages_per_batch: int = 0,
    pages_per_copy: int = 2,
    language: str = "fr",
    progress_callback=None,
    detect_students: bool = None
) -> List[BatchResult]:
    """
    Grade all copies in batches.

    Args:
        provider: LLM provider
        copies: List of all copies to grade
        questions: Question definitions
        max_pages_per_batch: Max pages per batch (0 = no limit, all in one batch)
        pages_per_copy: Number of pages per copy (for calculating batch size)
        language: Language for prompts
        progress_callback: Optional callback for progress updates
        detect_students: If True, ask LLM to detect multiple students.
                        If None, auto-detect based on copies structure.

    Returns:
        List of BatchResult, one per batch
    """
    grader = BatchGrader(provider)
    results = []

    # Auto-detect student detection mode if not specified
    if detect_students is None:
        detect_students = _should_detect_students(copies)

    if detect_students:
        logger.info("Student detection mode enabled: LLM will detect multiple students in PDF")

    # If no limit, process all copies in one batch
    if max_pages_per_batch <= 0:
        batches = [copies]
    else:
        # Split into batches based on page count
        batches = []
        current_batch = []
        current_pages = 0

        for copy in copies:
            copy_pages = pages_per_copy  # Each copy has this many pages

            if current_pages + copy_pages > max_pages_per_batch and current_batch:
                # Start new batch
                batches.append(current_batch)
                current_batch = []
                current_pages = 0

            current_batch.append(copy)
            current_pages += copy_pages

        if current_batch:
            batches.append(current_batch)

    total_batches = len(batches)

    for batch_idx, batch_copies in enumerate(batches):
        if progress_callback:
            await progress_callback('batch_start', {
                'batch_index': batch_idx + 1,
                'total_batches': total_batches,
                'copies_in_batch': len(batch_copies)
            })

        result = await grader.grade_batch(batch_copies, questions, language, detect_students)
        results.append(result)

        if progress_callback:
            await progress_callback('batch_done', {
                'batch_index': batch_idx + 1,
                'total_batches': total_batches,
                'success': result.parse_success
            })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# POST-BATCH VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Disagreement:
    """Represents a disagreement between two LLMs on a specific question."""
    copy_index: int
    question_id: str
    student_name: str  # Student name for this copy
    llm1_name: str
    llm1_grade: float
    llm1_reasoning: str
    llm1_reading: str
    llm2_name: str
    llm2_grade: float
    llm2_reasoning: str
    llm2_reading: str
    difference: float
    max_points: float  # Frozen barème (from pre-analysis)
    image_paths: List[str]  # Paths to the copy images
    disagreement_type: str = "grade"  # "grade", "reading", or combination
    pdf_start_page: Optional[int] = None  # Page de début dans le PDF (1-based)
    pdf_end_page: Optional[int] = None    # Page de fin dans le PDF (1-based)


def detect_disagreements(
    llm1_result: BatchResult,
    llm2_result: BatchResult,
    llm1_name: str,
    llm2_name: str,
    copies_data: List[Dict[str, Any]],
    threshold: Optional[float] = None,
    pdf_page_count: Optional[int] = None
) -> List[Disagreement]:
    """
    Detect disagreements between two LLM batch results.

    Uses page-based matching when available (matches copies by overlapping
    page ranges), with fallback to index-based matching.

    Args:
        llm1_result: First LLM's batch result
        llm2_result: Second LLM's batch result
        llm1_name: Name of first LLM
        llm2_name: Name of second LLM
        copies_data: Original copies data with image paths
        threshold: Minimum difference as percentage of max_points (default from settings)
        pdf_page_count: Total pages in PDF (for validation logging)

    Returns:
        List of Disagreement objects
    """
    from utils.page_matching import match_with_fallback, compare_llm_detections

    if threshold is None:
        threshold = get_settings().grade_agreement_threshold

    # 1. VALIDATION: Log warnings if detections have issues
    if pdf_page_count:
        comparison = compare_llm_detections(
            llm1_result.copies, llm2_result.copies, pdf_page_count
        )
        if comparison['has_issues']:
            for w in comparison['llm1_warnings']:
                logger.warning(f"LLM1 detection: {w}")
            for w in comparison['llm2_warnings']:
                logger.warning(f"LLM2 detection: {w}")
            if comparison['copy_count_mismatch']:
                logger.warning(
                    f"Copy count mismatch: LLM1={comparison['llm1_copy_count']}, "
                    f"LLM2={comparison['llm2_copy_count']}"
                )

    # 2. MATCHING: Use page-based matching with fallback to index
    match_result = match_with_fallback(llm1_result.copies, llm2_result.copies)
    logger.info(
        f"Matching method: {match_result.match_method}, "
        f"matched: {len(match_result.matches)}, "
        f"LLM1 unmatched: {len(match_result.llm1_unmatched)}, "
        f"LLM2 unmatched: {len(match_result.llm2_unmatched)}"
    )

    # 3. COMPARISON: Find disagreements among matched copies
    disagreements = []

    # Build lookup for copies_data by copy_index
    copies_data_by_idx = {c['copy_index']: c for c in copies_data}

    for match in match_result.matches:
        llm1_copy = match['llm1_copy']
        llm2_copy = match['llm2_copy']
        copy_idx = llm1_copy.copy_index

        # Get student name (prefer LLM1, fallback to LLM2)
        student_name = llm1_copy.student_name or llm2_copy.student_name or f"Élève {copy_idx}"

        # Get image paths and page info for this copy
        copy_data = copies_data_by_idx.get(copy_idx)
        image_paths = copy_data.get('image_paths', []) if copy_data else []
        pdf_start_page = copy_data.get('start_page') if copy_data else None
        pdf_end_page = copy_data.get('end_page') if copy_data else None

        # Check each question
        for qid, q1_data in llm1_copy.questions.items():
            q2_data = llm2_copy.questions.get(qid)
            if not q2_data:
                continue

            grade1 = float(q1_data.get('grade', 0))
            grade2 = float(q2_data.get('grade', 0))
            # Use frozen max_points from copy_data (passed via questions)
            # Fallback to 1.0 if not available
            max_points = float(q1_data.get('max_points', 1.0))
            diff = abs(grade1 - grade2)

            # Use relative threshold (percentage of max_points)
            relative_threshold = max_points * threshold

            # Get readings (handle None values with 'or')
            reading1 = (q1_data.get('student_answer_read') or '').strip().lower()
            reading2 = (q2_data.get('student_answer_read') or '').strip().lower()

            # Detect reading disagreement (significant difference, not just case/whitespace)
            reading_disagreement = False
            if reading1 and reading2 and reading1 != reading2:
                # Check if readings are significantly different
                # Use simple string similarity check
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(None, reading1, reading2).ratio()
                reading_disagreement = similarity < 0.8  # Less than 80% similar

            # Detect grade disagreement
            grade_disagreement = diff >= relative_threshold

            # Add to disagreements if any type (no more max_points disagreement - barème is frozen)
            if grade_disagreement or reading_disagreement:
                # Determine type (can combine multiple)
                types = []
                if grade_disagreement:
                    types.append("grade")
                if reading_disagreement:
                    types.append("reading")

                if len(types) > 1:
                    disp_type = "+".join(types)
                else:
                    disp_type = types[0] if types else "grade"

                disagreements.append(Disagreement(
                    copy_index=copy_idx,
                    question_id=qid,
                    student_name=student_name,
                    llm1_name=llm1_name,
                    llm1_grade=grade1,
                    llm1_reasoning=q1_data.get('reasoning', ''),
                    llm1_reading=q1_data.get('student_answer_read', ''),
                    llm2_name=llm2_name,
                    llm2_grade=grade2,
                    llm2_reasoning=q2_data.get('reasoning', ''),
                    llm2_reading=q2_data.get('student_answer_read', ''),
                    difference=diff,
                    max_points=max_points,
                    image_paths=image_paths,
                    disagreement_type=disp_type,
                    pdf_start_page=pdf_start_page,
                    pdf_end_page=pdf_end_page
                ))

    return disagreements


async def run_dual_llm_verification(
    providers: List[tuple],
    disagreements: List[Disagreement],
    language: str = "fr",
    name_disagreements: List[Dict[str, Any]] = None,
    extra_images: List[str] = None,
    chat_manager: 'CacheManager' = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run verification with BOTH LLMs seeing each other's work (grouped mode).

    All disagreements are verified in a single call per LLM.

    Args:
        providers: List of (name, provider) tuples
        disagreements: List of disagreements to verify
        language: Language for prompts
        name_disagreements: Optional list of student name disagreements (for grouped mode)
        extra_images: Optional list of additional images (for name-only cases)
        chat_manager: Optional CacheManager for context caching mode (saves ~50% tokens)

    Returns:
        Dict with:
        - "copy_{idx}_{qid}" -> {final_grade, llm1_grade, llm2_grade, method, ...}
        - "name_{idx}" -> {llm1_new_name, llm2_new_name, resolved_name, agreement} for name verifications
    """
    return await _run_dual_llm_phase(
        providers=providers,
        disagreements=disagreements,
        language=language,
        mode="verification",
        batching="grouped",
        name_disagreements=name_disagreements,
        extra_images=extra_images,
        chat_manager=chat_manager
    )


async def run_per_question_dual_verification(
    providers: List[tuple],
    disagreements: List[Disagreement],
    language: str = "fr",
    chat_manager: 'CacheManager' = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run verification for each disagreement separately with BOTH LLMs.

    Unlike run_dual_llm_verification (which groups all disagreements),
    this makes one call per disagreement per LLM (more granular).

    Args:
        providers: List of (name, provider) tuples
        disagreements: List of disagreements to verify
        language: Language for prompts
        chat_manager: Optional CacheManager for context caching mode (saves ~50% tokens)

    Returns:
        Dict mapping "copy_{idx}_{qid}" -> {final_grade, llm1_grade, llm2_grade, method, ...}
    """
    return await _run_dual_llm_phase(
        providers=providers,
        disagreements=disagreements,
        language=language,
        mode="verification",
        batching="per_question",
        chat_manager=chat_manager
    )


def _parse_verification_response(raw_response: str) -> Tuple[Dict[str, Dict], Dict[int, Dict]]:
    """
    Parse a verification response from an LLM.

    Returns:
        Tuple of (question_results, name_results) where:
        - question_results: Dict mapping "copy_{idx}_{qid}" -> verification data for grades/readings/max_points
        - name_results: Dict mapping copy_index -> name verification data
    """
    question_results = {}
    name_results = {}
    if not raw_response:
        return question_results, name_results

    data = extract_json_from_response(raw_response)
    if data is None:
        logger.error("Failed to extract JSON from verification response")
        return question_results, name_results

    try:
        # Parse question verifications (grades, readings, max_points)
        for v in data.get('verifications', []):
            key = f"copy_{v.get('copy_index')}_{v.get('question_id')}"
            question_results[key] = {
                'my_new_grade': float(v.get('my_new_grade', 0)),
                'my_new_max_points': float(v.get('my_new_max_points', 0)) if v.get('my_new_max_points') else None,
                'my_new_reading': v.get('my_new_reading'),
                'reasoning': v.get('reasoning', ''),
                'feedback': v.get('feedback', ''),
                'confidence': float(v.get('confidence', 0.8))
            }

        # Parse name verifications (if present)
        for nv in data.get('name_verifications', []):
            copy_idx = nv.get('copy_index')
            if copy_idx is not None:
                name_results[copy_idx] = {
                    'my_new_name': nv.get('my_new_name', ''),
                    'confidence': float(nv.get('confidence', 0.8))
                }
    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"Failed to parse verification response: {e}")

    return question_results, name_results


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS FOR DUAL LLM PHASES (Refactored)
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_disagreement_images(
    disagreements: List[Union[Disagreement, Dict]],
    copy_filter: Optional[int] = None
) -> List[str]:
    """
    Collect unique image paths from disagreements.

    Args:
        disagreements: List of Disagreement objects or dicts
        copy_filter: If set, only collect images for this copy index

    Returns:
        List of unique image paths
    """
    all_images = []
    seen = set()

    for d in disagreements:
        # Handle both Disagreement objects and dicts
        if isinstance(d, Disagreement):
            copy_idx = d.copy_index
            image_paths = d.image_paths
        else:
            copy_idx = d.get('copy_index')
            image_paths = d.get('image_paths', [])

        # Filter by copy if specified
        if copy_filter is not None and copy_idx != copy_filter:
            continue

        for img in image_paths:
            if img not in seen:
                all_images.append(img)
                seen.add(img)

    return all_images


def _build_dual_prompts(
    disagreements: List,
    llm1_name: str,
    llm2_name: str,
    language: str,
    mode: str,
    name_disagreements: List[Dict] = None,
    extra_images: List[str] = None
) -> Tuple[str, str]:
    """
    Build prompts for both LLMs based on mode.

    Args:
        disagreements: List of disagreements
        llm1_name: Name of first LLM
        llm2_name: Name of second LLM
        language: Language for prompts
        mode: "verification" or "ultimatum"
        name_disagreements: Optional name disagreements (verification only)
        extra_images: Optional extra images (verification only)

    Returns:
        Tuple of (prompt1, prompt2)
    """
    if mode == "verification":
        # Add provider info to name disagreements for prompt building
        name_disags_with_providers = None
        if name_disagreements:
            name_disags_with_providers = [
                {**nd, 'llm1_provider': llm1_name, 'llm2_provider': llm2_name}
                for nd in name_disagreements
            ]

        prompt1 = build_dual_llm_verification_prompt(
            disagreements, llm1_name, llm2_name, is_own_perspective=True,
            language=language, name_disagreements=name_disags_with_providers
        )
        prompt2 = build_dual_llm_verification_prompt(
            disagreements, llm2_name, llm1_name, is_own_perspective=True,
            language=language, name_disagreements=name_disags_with_providers
        )
    else:  # ultimatum
        # Convert to dict format if needed
        dict_disagreements = []
        for d in disagreements:
            if isinstance(d, Disagreement):
                dict_disagreements.append({
                    'copy_index': d.copy_index,
                    'question_id': d.question_id,
                    'llm1_grade': d.llm1_grade,
                    'llm2_grade': d.llm2_grade,
                    'max_points': d.max_points,  # Frozen barème
                    'image_paths': d.image_paths,
                    'llm1_reading': d.llm1_reading,
                    'llm2_reading': d.llm2_reading,
                })
            else:
                dict_disagreements.append(d)

        prompt1 = build_ultimatum_prompt(
            dict_disagreements, llm1_name, llm2_name, language=language
        )
        prompt2 = build_ultimatum_prompt(
            dict_disagreements, llm2_name, llm1_name, language=language
        )

    return prompt1, prompt2


async def _call_dual_providers(
    providers: List[Tuple[str, Any]],
    prompt1: str,
    prompt2: str,
    images: List[str]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Call both providers in parallel.

    Args:
        providers: List of (name, provider) tuples
        prompt1: Prompt for first provider
        prompt2: Prompt for second provider
        images: List of image paths

    Returns:
        Tuple of (response1, response2)
    """
    _, provider1 = providers[0]
    _, provider2 = providers[1]

    response1, response2 = await asyncio.gather(
        _call_provider_vision(provider1, prompt1, images),
        _call_provider_vision(provider2, prompt2, images)
    )

    return response1, response2


def _parse_dual_responses(
    response1: Optional[str],
    response2: Optional[str],
    mode: str
) -> Tuple[Dict, Dict]:
    """
    Parse responses based on mode.

    Args:
        response1: Response from first LLM
        response2: Response from second LLM
        mode: "verification" or "ultimatum"

    Returns:
        Tuple of (results1, results2) - parsed results for each LLM
    """
    if mode == "verification":
        # Verification returns tuple of (question_results, name_results)
        results1 = _parse_verification_response(response1)
        results2 = _parse_verification_response(response2)
        return (results1, results2)
    else:  # ultimatum
        return (_parse_ultimatum_response(response1), _parse_ultimatum_response(response2))


def _consolidate_feedback(
    llm1_feedback: str,
    llm2_feedback: str
) -> str:
    """
    Consolidate feedback from both LLMs.

    Prefers non-empty feedback, with LLM1 as primary.
    """
    if llm1_feedback and llm2_feedback:
        return llm1_feedback  # Use LLM1's as primary
    elif llm1_feedback:
        return llm1_feedback
    elif llm2_feedback:
        return llm2_feedback
    return ''


def _detect_flip_flop(
    initial_llm1: float,
    initial_llm2: float,
    final_llm1: float,
    final_llm2: float,
    max_points: float
) -> bool:
    """
    Detect if LLMs have swapped their positions (flip-flop).

    Args:
        initial_llm1: Initial grade from LLM1
        initial_llm2: Initial grade from LLM2
        final_llm1: Final grade from LLM1
        final_llm2: Final grade from LLM2
        max_points: Max points for threshold calculation

    Returns:
        True if flip-flop detected
    """
    initial_diff = initial_llm1 - initial_llm2
    final_diff = final_llm1 - final_llm2

    # Check if signs are opposite (positions crossed)
    is_swap = (
        (initial_diff > 0 and final_diff < 0) or
        (initial_diff < 0 and final_diff > 0)
    )

    # Use configurable threshold
    significant_diff = get_flip_flop_threshold(max_points)

    return (
        is_swap and
        abs(initial_diff) >= significant_diff and
        abs(final_diff) >= significant_diff
    )


def _resolve_verification_grade(
    disagreement: Union[Disagreement, Dict],
    llm1_question_results: Dict,
    llm2_question_results: Dict,
    key: str
) -> Dict[str, Any]:
    """
    Resolve a single disagreement from verification phase.

    Args:
        disagreement: The disagreement object or dict
        llm1_question_results: Parsed results from LLM1
        llm2_question_results: Parsed results from LLM2
        key: The result key (e.g., "copy_1_Q1")

    Returns:
        Dict with resolved grade, max_points, feedback, etc.
    """
    # Extract values from disagreement
    if isinstance(disagreement, Disagreement):
        llm1_grade = disagreement.llm1_grade
        llm2_grade = disagreement.llm2_grade
        max_points = disagreement.max_points  # Frozen barème
    else:
        llm1_grade = disagreement.get('llm1_grade', 0)
        llm2_grade = disagreement.get('llm2_grade', 0)
        max_points = disagreement.get('max_points', 1.0)

    # Get new grades
    llm1_new = llm1_question_results.get(key, {}).get('my_new_grade', llm1_grade)
    llm2_new = llm2_question_results.get(key, {}).get('my_new_grade', llm2_grade)

    # Use frozen max_points from disagreement
    resolved_max_points = max_points

    # Resolve: consensus or average
    if abs(llm1_new - llm2_new) < get_agreement_threshold(max_points):
        final_grade = (llm1_new + llm2_new) / 2
        method = "verification_consensus"
    else:
        final_grade = (llm1_new + llm2_new) / 2
        method = "verification_average"

    # Get feedbacks
    llm1_feedback = llm1_question_results.get(key, {}).get('feedback', '')
    llm2_feedback = llm2_question_results.get(key, {}).get('feedback', '')
    final_feedback = _consolidate_feedback(llm1_feedback, llm2_feedback)

    # Get readings
    llm1_new_reading = llm1_question_results.get(key, {}).get('my_new_reading')
    llm2_new_reading = llm2_question_results.get(key, {}).get('my_new_reading')

    return {
        'final_grade': final_grade,
        'llm1_new_grade': llm1_new,
        'llm2_new_grade': llm2_new,
        'resolved_max_points': resolved_max_points,  # Frozen barème
        'llm1_new_reading': llm1_new_reading,
        'llm2_new_reading': llm2_new_reading,
        'llm1_reasoning': llm1_question_results.get(key, {}).get('reasoning', ''),
        'llm2_reasoning': llm2_question_results.get(key, {}).get('reasoning', ''),
        'llm1_feedback': llm1_feedback,
        'llm2_feedback': llm2_feedback,
        'final_feedback': final_feedback,
        'method': method,
        'max_points': max_points,  # Frozen barème
        'confidence': max(
            llm1_question_results.get(key, {}).get('confidence', 0.8),
            llm2_question_results.get(key, {}).get('confidence', 0.8)
        )
    }


def _resolve_ultimatum_grade(
    disagreement: Dict,
    llm1_results: Dict,
    llm2_results: Dict,
    key: str
) -> Dict[str, Any]:
    """
    Resolve a single disagreement from ultimatum phase.

    Args:
        disagreement: The disagreement dict
        llm1_results: Parsed results from LLM1
        llm2_results: Parsed results from LLM2
        key: The result key (e.g., "copy_1_Q1")

    Returns:
        Dict with resolved grade, max_points, feedback, flip_flop detection, etc.
    """
    # Extract values from disagreement (handles both dict and Disagreement object)
    if isinstance(disagreement, Disagreement):
        max_pts = disagreement.max_points
        llm1_grade = disagreement.llm1_grade
        llm2_grade = disagreement.llm2_grade
    else:
        max_pts = disagreement.get('max_points', 1)
        llm1_grade = disagreement.get('llm1_grade', 0)
        llm2_grade = disagreement.get('llm2_grade', 0)

    # Get final grades
    llm1_final = llm1_results.get(key, {}).get('my_final_grade', llm1_grade)
    llm2_final = llm2_results.get(key, {}).get('my_final_grade', llm2_grade)

    # Resolve: consensus or average
    if abs(llm1_final - llm2_final) < get_agreement_threshold(max_pts):
        final_grade = (llm1_final + llm2_final) / 2
        method = "ultimatum_consensus"
    else:
        final_grade = (llm1_final + llm2_final) / 2
        method = "ultimatum_average"

    # Detect flip-flop
    flip_flop = _detect_flip_flop(llm1_grade, llm2_grade, llm1_final, llm2_final, max_pts)

    # Get feedbacks
    llm1_feedback = llm1_results.get(key, {}).get('feedback', '')
    llm2_feedback = llm2_results.get(key, {}).get('feedback', '')
    final_feedback = _consolidate_feedback(llm1_feedback, llm2_feedback)

    # Get final readings
    llm1_final_reading = llm1_results.get(key, {}).get('my_final_reading')
    llm2_final_reading = llm2_results.get(key, {}).get('my_final_reading')

    # Check parsing success
    llm1_parsed = key in llm1_results
    llm2_parsed = key in llm2_results

    result = {
        'final_grade': final_grade,
        'llm1_final_grade': llm1_final,
        'llm2_final_grade': llm2_final,
        'llm1_final_reading': llm1_final_reading,
        'llm2_final_reading': llm2_final_reading,
        'llm1_decision': llm1_results.get(key, {}).get('decision', 'unknown'),
        'llm2_decision': llm2_results.get(key, {}).get('decision', 'unknown'),
        'llm1_reasoning': llm1_results.get(key, {}).get('reasoning', ''),
        'llm2_reasoning': llm2_results.get(key, {}).get('reasoning', ''),
        'llm1_feedback': llm1_feedback,
        'llm2_feedback': llm2_feedback,
        'final_feedback': final_feedback,
        'llm1_parse_success': llm1_parsed,
        'llm2_parse_success': llm2_parsed,
        'method': method,
        'flip_flop_detected': flip_flop,
        'confidence': max(
            llm1_results.get(key, {}).get('confidence', 0.8),
            llm2_results.get(key, {}).get('confidence', 0.8)
        ),
        'resolved_max_points': max_pts  # Frozen barème used
    }

    return result


async def _run_dual_llm_phase(
    providers: List[Tuple[str, Any]],
    disagreements: List[Union[Disagreement, Dict]],
    language: str,
    mode: str,
    batching: str,
    name_disagreements: List[Dict] = None,
    extra_images: List[str] = None,
    chat_manager: 'CacheManager' = None
) -> Dict[str, Dict[str, Any]]:
    """
    Unified function for all verification/ultimatum phases.

    This consolidates the 6 similar functions (verification/ultimatum × grouped/per_question/per_copy)
    into one parameterized function.

    Args:
        providers: List of (name, provider) tuples
        disagreements: List of Disagreement objects or dicts
        language: "fr" or "en"
        mode: "verification" or "ultimatum"
        batching: "grouped", "per_question", or "per_copy"
        name_disagreements: Optional name disagreements (verification grouped mode only)
        extra_images: Optional extra images (verification grouped mode only)
        chat_manager: Optional CacheManager for context caching mode (saves ~50% tokens)

    Returns:
        Dict with results and optional metadata
    """
    from collections import defaultdict

    if not disagreements and not name_disagreements:
        return {}

    llm1_name, _ = providers[0]
    llm2_name, _ = providers[1]
    results = {}

    # Track parsing failures for ultimatum
    llm1_parse_failed = False
    llm2_parse_failed = False

    # Check if we should use cache manager
    use_cache = chat_manager is not None

    # If cache manager is provided, delegate to it
    if use_cache:
        logger.info(f"Using context caching for {mode} phase ({batching} mode)")

        if mode == "verification":
            return await chat_manager.run_dual_llm_verification_with_cache(
                disagreements, language
            )
        else:  # ultimatum
            return await chat_manager.run_dual_llm_ultimatum_with_cache(
                disagreements, language
            )

    if batching == "grouped":
        # ===== GROUPED: All disagreements in one call =====
        all_images = _collect_disagreement_images(disagreements)
        if extra_images:
            seen = set(all_images)
            for img in extra_images:
                if img not in seen:
                    all_images.append(img)

        prompt1, prompt2 = _build_dual_prompts(
            disagreements, llm1_name, llm2_name, language, mode,
            name_disagreements, extra_images
        )
        response1, response2 = await _call_dual_providers(providers, prompt1, prompt2, all_images)

        if mode == "verification":
            (llm1_question_results, llm1_name_results), (llm2_question_results, llm2_name_results) = \
                _parse_dual_responses(response1, response2, mode)

            # Resolve each disagreement
            for d in disagreements:
                # Handle both Disagreement objects and dicts
                if isinstance(d, Disagreement):
                    key = f"copy_{d.copy_index}_{d.question_id}"
                else:
                    key = f"copy_{d['copy_index']}_{d['question_id']}"
                results[key] = _resolve_verification_grade(
                    d, llm1_question_results, llm2_question_results, key
                )
                # Preserve image_paths for potential ultimatum
                if isinstance(d, Disagreement):
                    results[key]['image_paths'] = d.image_paths

            # Process name verifications if present
            if name_disagreements:
                for nd in name_disagreements:
                    copy_idx = nd['copy_index']
                    name_key = f"name_{copy_idx}"

                    llm1_name_result = llm1_name_results.get(copy_idx, {})
                    llm2_name_result = llm2_name_results.get(copy_idx, {})

                    llm1_new_name = llm1_name_result.get('my_new_name', nd.get('llm1_name', ''))
                    llm2_new_name = llm2_name_result.get('my_new_name', nd.get('llm2_name', ''))

                    n1_normalized = llm1_new_name.lower().strip()
                    n2_normalized = llm2_new_name.lower().strip()
                    agreement = n1_normalized == n2_normalized and n1_normalized != ''

                    results[name_key] = {
                        'llm1_new_name': llm1_new_name,
                        'llm2_new_name': llm2_new_name,
                        'resolved_name': llm1_new_name if agreement else None,
                        'agreement': agreement,
                        'confidence': max(
                            llm1_name_result.get('confidence', 0.8),
                            llm2_name_result.get('confidence', 0.8)
                        )
                    }
        else:  # ultimatum
            llm1_results, llm2_results = _parse_dual_responses(response1, response2, mode)

            # Track parsing failures
            llm1_parse_failed = len(llm1_results) == 0 and response1
            llm2_parse_failed = len(llm2_results) == 0 and response2

            for d in disagreements:
                key = f"copy_{d['copy_index']}_{d['question_id']}"
                results[key] = _resolve_ultimatum_grade(d, llm1_results, llm2_results, key)

            # Add overall parse status
            results['_parse_status'] = {
                'llm1_parse_failed': llm1_parse_failed,
                'llm2_parse_failed': llm2_parse_failed
            }

    elif batching == "per_question":
        # ===== PER-QUESTION: One call per disagreement =====
        for d in disagreements:
            if isinstance(d, Disagreement):
                key = f"copy_{d.copy_index}_{d.question_id}"
                images = d.image_paths
            else:
                key = f"copy_{d['copy_index']}_{d['question_id']}"
                images = d.get('image_paths', [])

            prompt1, prompt2 = _build_dual_prompts(
                [d], llm1_name, llm2_name, language, mode
            )
            response1, response2 = await _call_dual_providers(providers, prompt1, prompt2, images)

            if mode == "verification":
                (llm1_question_results, _), (llm2_question_results, _) = \
                    _parse_dual_responses(response1, response2, mode)

                results[key] = _resolve_verification_grade(
                    d, llm1_question_results, llm2_question_results, key
                )
                results[key]['image_paths'] = images
            else:  # ultimatum
                llm1_results, llm2_results = _parse_dual_responses(response1, response2, mode)

                if len(llm1_results) == 0 and response1:
                    llm1_parse_failed = True
                if len(llm2_results) == 0 and response2:
                    llm2_parse_failed = True

                results[key] = _resolve_ultimatum_grade(d, llm1_results, llm2_results, key)

        if mode == "ultimatum":
            results['_parse_status'] = {
                'llm1_parse_failed': llm1_parse_failed,
                'llm2_parse_failed': llm2_parse_failed
            }

    elif batching == "per_copy":
        # ===== PER-COPY: Group by copy, one call per copy =====
        by_copy = defaultdict(list)
        for d in disagreements:
            if isinstance(d, Disagreement):
                by_copy[d.copy_index].append(d)
            else:
                by_copy[d['copy_index']].append(d)

        # Also group name_disagreements by copy if provided
        name_by_copy = defaultdict(list)
        if name_disagreements:
            for nd in name_disagreements:
                name_by_copy[nd['copy_index']].append(nd)

        for copy_idx, copy_disagreements in by_copy.items():
            logger.info(f"Running per-copy {mode} for copy {copy_idx} ({len(copy_disagreements)} disagreements)")

            copy_images = _collect_disagreement_images(copy_disagreements, copy_filter=copy_idx)

            # Get name disagreements for this copy (verification mode only)
            copy_name_disagreements = name_by_copy.get(copy_idx, []) if mode == "verification" and name_disagreements else None

            prompt1, prompt2 = _build_dual_prompts(
                copy_disagreements, llm1_name, llm2_name, language, mode,
                name_disagreements=copy_name_disagreements
            )
            response1, response2 = await _call_dual_providers(providers, prompt1, prompt2, copy_images)

            if mode == "verification":
                (llm1_question_results, llm1_name_results), (llm2_question_results, llm2_name_results) = \
                    _parse_dual_responses(response1, response2, mode)

                for d in copy_disagreements:
                    if isinstance(d, Disagreement):
                        key = f"copy_{d.copy_index}_{d.question_id}"
                    else:
                        key = f"copy_{d['copy_index']}_{d['question_id']}"

                    results[key] = _resolve_verification_grade(
                        d, llm1_question_results, llm2_question_results, key
                    )
                    results[key]['image_paths'] = copy_images

                # Process name verifications for this copy if present
                if copy_name_disagreements:
                    for nd in copy_name_disagreements:
                        name_key = f"name_{copy_idx}"

                        llm1_name_result = llm1_name_results.get(copy_idx, {})
                        llm2_name_result = llm2_name_results.get(copy_idx, {})

                        llm1_new_name = llm1_name_result.get('my_new_name', nd.get('llm1_name', ''))
                        llm2_new_name = llm2_name_result.get('my_new_name', nd.get('llm2_name', ''))

                        n1_normalized = llm1_new_name.lower().strip()
                        n2_normalized = llm2_new_name.lower().strip()
                        agreement = n1_normalized == n2_normalized and n1_normalized != ''

                        results[name_key] = {
                            'llm1_new_name': llm1_new_name,
                            'llm2_new_name': llm2_new_name,
                            'resolved_name': llm1_new_name if agreement else None,
                            'agreement': agreement,
                            'confidence': max(
                                llm1_name_result.get('confidence', 0.8),
                                llm2_name_result.get('confidence', 0.8)
                            )
                        }
            else:  # ultimatum
                llm1_results, llm2_results = _parse_dual_responses(response1, response2, mode)

                if len(llm1_results) == 0 and response1:
                    llm1_parse_failed = True
                if len(llm2_results) == 0 and response2:
                    llm2_parse_failed = True

                for d in copy_disagreements:
                    key = f"copy_{d['copy_index']}_{d['question_id']}"
                    results[key] = _resolve_ultimatum_grade(d, llm1_results, llm2_results, key)

        if mode == "ultimatum":
            results['_parse_status'] = {
                'llm1_parse_failed': llm1_parse_failed,
                'llm2_parse_failed': llm2_parse_failed
            }

    return results


async def run_per_copy_dual_verification(
    providers: List[tuple],
    disagreements: List[Disagreement],
    language: str = "fr",
    name_disagreements: List[Dict] = None,
    chat_manager: 'CacheManager' = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run verification grouped by copy with BOTH LLMs.

    Each copy's disagreements are verified together in one call per LLM.
    This is a middle ground between grouped (all in one) and per-question
    (one per disagreement).

    Args:
        providers: List of (name, provider) tuples
        disagreements: List of disagreements to verify
        language: Language for prompts
        name_disagreements: Optional list of student name disagreements
        chat_manager: Optional CacheManager for context caching mode (saves ~50% tokens)

    Returns:
        Dict mapping "copy_{idx}_{qid}" -> {final_grade, llm1_grade, llm2_grade, method, ...}
        Also includes "name_{idx}" entries for name verifications
    """
    return await _run_dual_llm_phase(
        providers=providers,
        disagreements=disagreements,
        language=language,
        mode="verification",
        batching="per_copy",
        name_disagreements=name_disagreements,
        chat_manager=chat_manager
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ULTIMATUM ROUND
# ═══════════════════════════════════════════════════════════════════════════════

async def run_dual_llm_ultimatum(
    providers: List[tuple],
    persistent_disagreements: List[Dict[str, Any]],
    language: str = "fr",
    chat_manager: 'CacheManager' = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run ultimatum round with BOTH LLMs for persistent disagreements (grouped mode).

    All disagreements are resolved in a single call per LLM.

    Args:
        providers: List of (name, provider) tuples
        persistent_disagreements: List of disagreements that persist after verification
        language: Language for prompts
        chat_manager: Optional CacheManager for context caching mode (saves ~50% tokens)

    Returns:
        Dict mapping "copy_{idx}_{qid}" -> {final_grade, llm1_grade, llm2_grade, method, ...}
    """
    return await _run_dual_llm_phase(
        providers=providers,
        disagreements=persistent_disagreements,
        language=language,
        mode="ultimatum",
        batching="grouped",
        chat_manager=chat_manager
    )


async def run_per_question_dual_ultimatum(
    providers: List[tuple],
    persistent_disagreements: List[Dict[str, Any]],
    language: str = "fr",
    chat_manager: 'CacheManager' = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run ultimatum for each disagreement separately with BOTH LLMs.

    Unlike run_dual_llm_ultimatum (which groups all disagreements),
    this makes one call per disagreement per LLM (more granular).

    Args:
        providers: List of (name, provider) tuples
        persistent_disagreements: List of disagreements that persist after verification
        language: Language for prompts
        chat_manager: Optional CacheManager for context caching mode (saves ~50% tokens)

    Returns:
        Dict mapping "copy_{idx}_{qid}" -> {final_grade, llm1_grade, llm2_grade, method, ...}
    """
    return await _run_dual_llm_phase(
        providers=providers,
        disagreements=persistent_disagreements,
        language=language,
        mode="ultimatum",
        batching="per_question",
        chat_manager=chat_manager
    )


async def run_per_copy_dual_ultimatum(
    providers: List[tuple],
    persistent_disagreements: List[Dict[str, Any]],
    language: str = "fr",
    chat_manager: 'CacheManager' = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run ultimatum grouped by copy with BOTH LLMs.

    Each copy's persistent disagreements are sent together in one call per LLM.
    This is a middle ground between grouped (all in one) and per-question
    (one per disagreement).

    Args:
        providers: List of (name, provider) tuples
        persistent_disagreements: List of disagreements that persist after verification
        language: Language for prompts
        chat_manager: Optional CacheManager for context caching mode (saves ~50% tokens)

    Returns:
        Dict mapping "copy_{idx}_{qid}" -> {final_grade, llm1_grade, llm2_grade, method, ...}
    """
    return await _run_dual_llm_phase(
        providers=providers,
        disagreements=persistent_disagreements,
        language=language,
        mode="ultimatum",
        batching="per_copy",
        chat_manager=chat_manager
    )


def _parse_ultimatum_response(raw_response: str) -> Dict[str, Dict]:
    """Parse an ultimatum response from an LLM."""
    results = {}
    if not raw_response:
        return results

    data = extract_json_from_response(raw_response)
    if data is None:
        logger.error("Failed to extract JSON from ultimatum response")
        return results

    try:
        for u in data.get('ultimatum_decisions', []):
            key = f"copy_{u.get('copy_index')}_{u.get('question_id')}"
            results[key] = {
                'my_final_grade': float(u.get('my_final_grade', 0)),
                'my_final_max_points': float(u.get('my_final_max_points')) if u.get('my_final_max_points') else None,
                'my_final_reading': u.get('my_final_reading'),
                'decision': u.get('decision', 'unknown'),
                'reasoning': u.get('reasoning', ''),
                'feedback': u.get('feedback', ''),
                'confidence': float(u.get('confidence', 0.8))
            }
    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"Failed to parse ultimatum response: {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STUDENT NAME VERIFICATION & ULTIMATUM
# ═══════════════════════════════════════════════════════════════════════════════

async def run_student_name_verification(
    providers: List[tuple],
    name_disagreements: List[Dict[str, Any]],
    image_paths: List[str],
    language: str = "fr"
) -> Dict[int, Dict[str, Any]]:
    """
    Run verification round for student name disagreements.

    Args:
        providers: List of (name, provider) tuples
        name_disagreements: List of dicts with copy_index, llm1_name, llm2_name
        image_paths: List of image paths for the copies
        language: Language for prompts

    Returns:
        Dict mapping copy_index -> {llm1_new_name, llm2_new_name, resolved_name}
    """
    if not name_disagreements:
        return {}

    from prompts.batch import build_student_name_verification_prompt

    llm1_name, llm1_provider = providers[0]
    llm2_name, llm2_provider = providers[1]

    # Build prompts for each LLM
    llm1_prompt = build_student_name_verification_prompt(
        name_disagreements, llm1_name, llm2_name, language=language
    )
    llm2_prompt = build_student_name_verification_prompt(
        name_disagreements, llm2_name, llm1_name, language=language
    )

    # Call both LLMs in parallel
    llm1_response, llm2_response = await asyncio.gather(
        _call_provider_vision(llm1_provider, llm1_prompt, image_paths),
        _call_provider_vision(llm2_provider, llm2_prompt, image_paths)
    )

    # Parse responses
    llm1_results = _parse_student_name_verification_response(llm1_response)
    llm2_results = _parse_student_name_verification_response(llm2_response)

    # Merge results
    results = {}
    for d in name_disagreements:
        copy_idx = d['copy_index']

        llm1_new = llm1_results.get(copy_idx, {}).get('my_new_name', d['llm1_name'])
        llm2_new = llm2_results.get(copy_idx, {}).get('my_new_name', d['llm2_name'])

        # Check if names now agree (case-insensitive)
        n1_normalized = llm1_new.lower().strip() if llm1_new else ""
        n2_normalized = llm2_new.lower().strip() if llm2_new else ""

        if n1_normalized == n2_normalized:
            resolved = llm1_new  # They agree
            agreement = True
        else:
            # Still disagree - use LLM1 as base for now
            resolved = llm1_new
            agreement = False

        results[copy_idx] = {
            'llm1_new_name': llm1_new,
            'llm2_new_name': llm2_new,
            'resolved_name': resolved,
            'agreement': agreement,
            'llm1_confidence': llm1_results.get(copy_idx, {}).get('confidence', 0.8),
            'llm2_confidence': llm2_results.get(copy_idx, {}).get('confidence', 0.8)
        }

    return results


def _parse_student_name_verification_response(raw_response: str) -> Dict[int, Dict]:
    """Parse a student name verification response from an LLM."""
    results = {}
    if not raw_response:
        return results

    data = extract_json_from_response(raw_response)
    if data is None:
        logger.error("Failed to extract JSON from student name verification response")
        return results

    try:
        for v in data.get('name_verifications', []):
            copy_idx = v.get('copy_index')
            results[copy_idx] = {
                'my_new_name': v.get('my_new_name'),
                'confidence': float(v.get('confidence', 0.8))
            }
    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"Failed to parse student name verification response: {e}")

    return results


async def run_student_name_ultimatum(
    providers: List[tuple],
    persistent_disagreements: List[Dict[str, Any]],
    image_paths: List[str],
    language: str = "fr"
) -> Dict[int, Dict[str, Any]]:
    """
    Run ultimatum round for persistent student name disagreements.

    Args:
        providers: List of (name, provider) tuples
        persistent_disagreements: List of disagreements that persist after verification
        image_paths: List of image paths for the copies
        language: Language for prompts

    Returns:
        Dict mapping copy_index -> {llm1_final_name, llm2_final_name, resolved_name}
    """
    if not persistent_disagreements:
        return {}

    from prompts.batch import build_student_name_ultimatum_prompt

    llm1_name, llm1_provider = providers[0]
    llm2_name, llm2_provider = providers[1]

    # Build prompts for each LLM
    llm1_prompt = build_student_name_ultimatum_prompt(
        persistent_disagreements, llm1_name, llm2_name, language=language
    )
    llm2_prompt = build_student_name_ultimatum_prompt(
        persistent_disagreements, llm2_name, llm1_name, language=language
    )

    # Call both LLMs in parallel
    llm1_response, llm2_response = await asyncio.gather(
        _call_provider_vision(llm1_provider, llm1_prompt, image_paths),
        _call_provider_vision(llm2_provider, llm2_prompt, image_paths)
    )

    # Parse responses
    llm1_results = _parse_student_name_ultimatum_response(llm1_response)
    llm2_results = _parse_student_name_ultimatum_response(llm2_response)

    # Merge results
    results = {}
    for d in persistent_disagreements:
        copy_idx = d['copy_index']

        llm1_final = llm1_results.get(copy_idx, {}).get('my_final_name', d['llm1_name'])
        llm2_final = llm2_results.get(copy_idx, {}).get('my_final_name', d['llm2_name'])

        # Check if names now agree (case-insensitive)
        n1_normalized = llm1_final.lower().strip() if llm1_final else ""
        n2_normalized = llm2_final.lower().strip() if llm2_final else ""

        if n1_normalized == n2_normalized:
            resolved = llm1_final
            agreement = True
        else:
            # Still disagree after ultimatum
            resolved = llm1_final  # Use LLM1 as default
            agreement = False

        results[copy_idx] = {
            'llm1_final_name': llm1_final,
            'llm2_final_name': llm2_final,
            'resolved_name': resolved,
            'agreement': agreement,
            'llm1_confidence': llm1_results.get(copy_idx, {}).get('confidence', 0.8),
            'llm2_confidence': llm2_results.get(copy_idx, {}).get('confidence', 0.8),
            'needs_user_resolution': not agreement
        }

    return results


def _parse_student_name_ultimatum_response(raw_response: str) -> Dict[int, Dict]:
    """Parse a student name ultimatum response from an LLM."""
    results = {}
    if not raw_response:
        return results

    data = extract_json_from_response(raw_response)
    if data is None:
        logger.error("Failed to extract JSON from student name ultimatum response")
        return results

    try:
        for u in data.get('name_ultimatum_decisions', []):
            copy_idx = u.get('copy_index')
            results[copy_idx] = {
                'my_final_name': u.get('my_final_name'),
                'confidence': float(u.get('confidence', 0.8))
            }
    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"Failed to parse student name ultimatum response: {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# IMPLICIT CACHING FOR VERIFICATION/ULTIMATUM
# ═══════════════════════════════════════════════════════════════════════════════

# Gemini 2.5 implicit caching minimum tokens
# Flash: 1,024 tokens | Pro: 2,048 tokens
MIN_IMPLICIT_CACHE_TOKENS_FLASH = 1024
MIN_IMPLICIT_CACHE_TOKENS_PRO = 2048


class CacheManager:
    """
    Manages implicit caching for verification/ultimatum phases.

    Uses Gemini 2.5's implicit caching feature which automatically caches
    common prefixes in API requests. No explicit cache creation needed.

    How it works:
    - Gemini automatically caches when requests share the same prefix
    - Minimum tokens: 1,024 (Flash) or 2,048 (Pro)
    - Cost savings: ~75% on cached tokens
    - TTL: System-managed (typically 5-10 minutes)

    Best practices:
    - Keep stable content (system prompt + images) at the START
    - Add dynamic content (questions) at the END
    - Send requests within short time windows for cache hits
    """

    def __init__(self, providers: List[Tuple[str, Any]], cache_mode: str = "shared"):
        """
        Initialize CacheManager for implicit caching.

        Args:
            providers: List of (name, provider) tuples
            cache_mode: Ignored (kept for backward compatibility)
        """
        self.providers = providers
        self.cache_mode = cache_mode
        # Store images by copy for building prompts
        self._images_by_copy: Dict[int, List[str]] = {}
        # Store the common prefix (system prompt + images)
        self._prefix_content: Dict[int, Tuple[str, List[str]]] = {}  # copy_idx -> (prompt, images)
        # Track cache statistics
        self._cache_stats: Dict[str, Any] = {
            "calls": 0,              # Total API calls made
            "cached_tokens": 0,      # Total tokens served from cache
            "total_tokens": 0,       # Total tokens processed
            "estimated_savings": 0,  # Estimated token savings (75% of cached)
        }

    async def create_sessions(
        self,
        copies_data: List[Dict[str, Any]],
        questions: Dict[str, Dict[str, Any]] = None,
        language: str = "fr",
        initial_prompt: str = None
    ):
        """
        Prepare prefix content for implicit caching.

        Builds and stores a common prefix (role + rules + barème) that will be
        shared across initial grading, verification, and ultimatum phases.
        This enables Gemini's implicit caching to reduce token costs by ~75%.

        Args:
            copies_data: List with 'copy_index' and 'image_paths'
            questions: Dict of {question_id: {text, criteria, max_points}}
                       Required to build the common prefix for caching.
            language: Language for prompts (default: "fr")
            initial_prompt: DEPRECATED - kept for backward compatibility.
                           If provided without questions, uses this as prefix.
        """
        # Build common prefix from questions if provided
        if questions:
            common_prefix = build_common_prefix(questions, language)
            prefix_chars = len(common_prefix)
            logger.info(f"Built common prefix for caching: {prefix_chars} chars (~{prefix_chars // 4} tokens)")
        elif initial_prompt:
            # Backward compatibility: use provided prompt as prefix
            common_prefix = initial_prompt
            logger.info("Using provided initial_prompt as prefix (backward compatibility)")
        else:
            # No prefix available
            common_prefix = ""
            logger.warning("No questions or initial_prompt provided - caching disabled")

        # Store images by copy and prefix content
        for copy_data in copies_data:
            copy_idx = copy_data.get('copy_index', 0)
            images = copy_data.get('image_paths', [])
            self._images_by_copy[copy_idx] = images
            # Store prefix for this copy
            self._prefix_content[copy_idx] = (common_prefix, images)

        # Log setup info
        total_images = sum(len(imgs) for imgs in self._images_by_copy.values())
        prompt_chars = len(common_prefix) if common_prefix else 0
        estimated_tokens = (prompt_chars // 4) + (total_images * 258)

        # Determine minimum based on model
        model_name = self.providers[0][1].vision_model if self.providers else ""
        if "flash" in model_name.lower():
            min_tokens = MIN_IMPLICIT_CACHE_TOKENS_FLASH
        else:
            min_tokens = MIN_IMPLICIT_CACHE_TOKENS_PRO

        logger.info(
            f"Implicit caching ready: {len(self._images_by_copy)} copies, "
            f"{total_images} images, ~{estimated_tokens} tokens "
            f"(min {min_tokens} for cache activation)"
        )

        if estimated_tokens >= min_tokens:
            logger.info(f"✓ Prefix content should trigger implicit caching (~75% token savings)")
        else:
            logger.info(
                f"ℹ Content below cache threshold - requests will use regular pricing. "
                f"Add more content for automatic caching."
            )

    async def generate_with_prefix(
        self,
        provider_name: str,
        provider: Any,
        prefix_prompt: str,
        new_prompt: str,
        images: List[str],
        session_id: str = None
    ) -> Tuple[Optional[str], Dict[str, int]]:
        """
        Generate using prefix + new prompt for implicit caching.

        Structure: [PREFIX: system prompt + images] + [NEW: specific question]

        Args:
            provider_name: Name of the provider
            provider: Provider instance
            prefix_prompt: The stable prefix (system prompt)
            new_prompt: The dynamic part (specific question)
            images: Images to include in prefix
            session_id: Session ID for logging

        Returns:
            Tuple of (response text, usage stats dict)
        """
        # Build full prompt with prefix first, then new content
        full_prompt = f"{prefix_prompt}\n\n{new_prompt}"

        # Make the API call - implicit caching will work if prefix matches previous calls
        response = await _call_provider_vision(provider, full_prompt, images)

        # Try to extract cache usage info from response
        cache_stats = {"cached_tokens": 0, "total_tokens": 0}
        # Note: The response object might have usage_metadata with cached_content_token_count
        # This depends on the provider implementation

        return response, cache_stats

    async def generate_to_all_providers(
        self,
        prompt1: str,
        prompt2: str,
        images: List[str] = None,
        session_id: str = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate with both providers using implicit caching.

        Args:
            prompt1: Full prompt for first provider (includes prefix)
            prompt2: Full prompt for second provider (includes prefix)
            images: Images for the prefix
            session_id: Session ID for copy lookup

        Returns:
            Tuple of (response1, response2)
        """
        name1, provider1 = self.providers[0]
        name2, provider2 = self.providers[1]

        # Get images for this copy if session_id provided
        if session_id and session_id.startswith("copy_"):
            copy_idx = int(session_id.replace("copy_", ""))
            images = self._images_by_copy.get(copy_idx, images or [])

        # Make parallel calls - implicit caching will kick in if prefix matches
        results = await asyncio.gather(
            _call_provider_vision(provider1, prompt1, images),
            _call_provider_vision(provider2, prompt2, images)
        )

        # Track cache statistics (estimate based on prefix reuse)
        # Each successful call with common prefix should benefit from caching
        self._cache_stats["calls"] += 2  # Two providers called

        # Estimate tokens: prefix length + images
        common_prefix, _ = self._prefix_content.get(
            int(session_id.replace("copy_", "")) if session_id and session_id.startswith("copy_") else 0,
            ("", [])
        )
        if common_prefix:
            prefix_tokens = len(common_prefix) // 4
            image_tokens = len(images) * 258 if images else 0
            cached_tokens = prefix_tokens + image_tokens
            # Assume ~75% savings on cached tokens for subsequent calls
            self._cache_stats["cached_tokens"] += cached_tokens * 2
            self._cache_stats["total_tokens"] += cached_tokens * 2
            self._cache_stats["estimated_savings"] += int(cached_tokens * 2 * 0.75)
            logger.debug(f"Cache estimate for {session_id}: prefix={prefix_tokens} tokens, images={image_tokens} tokens, estimated savings={int(cached_tokens * 2 * 0.75)}")

        return results[0], results[1]

    async def run_dual_llm_verification_with_cache(
        self,
        disagreements: List[Disagreement],
        language: str = "fr"
    ) -> Dict[str, Dict[str, Any]]:
        """Run verification phase using implicit caching."""
        if not disagreements:
            return {}

        llm1_name, _ = self.providers[0]
        llm2_name, _ = self.providers[1]
        results = {}

        from collections import defaultdict
        by_copy = defaultdict(list)
        for d in disagreements:
            if isinstance(d, Disagreement):
                by_copy[d.copy_index].append(d)
            else:
                by_copy[d['copy_index']].append(d)

        # Debug: show what's stored
        logger.debug(f"Verification: _prefix_content keys: {list(self._prefix_content.keys())[:10]}...")
        logger.debug(f"Verification: by_copy keys: {list(by_copy.keys())[:10]}...")

        for copy_idx, copy_disagreements in by_copy.items():
            session_id = f"copy_{copy_idx}"
            images = _collect_disagreement_images(copy_disagreements, copy_filter=copy_idx)

            # Get the stored common prefix for this copy
            common_prefix, stored_images = self._prefix_content.get(copy_idx, ("", images))
            if not common_prefix:
                logger.warning(f"No common prefix stored for copy {copy_idx}, caching may not work")
            else:
                logger.debug(f"Using common prefix for copy {copy_idx}, length={len(common_prefix)}")

            # Build prompts WITH the common prefix for implicit caching
            prompt1 = build_dual_llm_verification_prompt(
                copy_disagreements, llm1_name, llm2_name, True, language,
                common_prefix=common_prefix if common_prefix else None
            )
            prompt2 = build_dual_llm_verification_prompt(
                copy_disagreements, llm2_name, llm1_name, True, language,
                common_prefix=common_prefix if common_prefix else None
            )

            response1, response2 = await self.generate_to_all_providers(prompt1, prompt2, images, session_id)

            (llm1_q, _), (llm2_q, _) = _parse_dual_responses(response1, response2, "verification")

            for d in copy_disagreements:
                if isinstance(d, Disagreement):
                    key = f"copy_{d.copy_index}_{d.question_id}"
                else:
                    key = f"copy_{d['copy_index']}_{d['question_id']}"
                results[key] = _resolve_verification_grade(d, llm1_q, llm2_q, key)
                results[key]['used_cache'] = bool(common_prefix)  # True if prefix was used

        return results

    async def run_dual_llm_ultimatum_with_cache(
        self,
        persistent_disagreements: List[Dict[str, Any]],
        language: str = "fr"
    ) -> Dict[str, Dict[str, Any]]:
        """Run ultimatum phase using implicit caching."""
        if not persistent_disagreements:
            return {}

        llm1_name, _ = self.providers[0]
        llm2_name, _ = self.providers[1]
        results = {}

        from collections import defaultdict
        by_copy = defaultdict(list)
        for d in persistent_disagreements:
            by_copy[d['copy_index']].append(d)

        llm1_parse_failed = False
        llm2_parse_failed = False

        for copy_idx, copy_disagreements in by_copy.items():
            session_id = f"copy_{copy_idx}"
            images = _collect_disagreement_images(copy_disagreements, copy_filter=copy_idx)

            # Get the stored common prefix for this copy
            common_prefix, stored_images = self._prefix_content.get(copy_idx, ("", images))
            if not common_prefix:
                logger.warning(f"No common prefix stored for copy {copy_idx}, caching may not work")

            # Build prompts WITH the common prefix for implicit caching
            prompt1 = build_ultimatum_prompt(
                copy_disagreements, llm1_name, llm2_name, language,
                common_prefix=common_prefix if common_prefix else None
            )
            prompt2 = build_ultimatum_prompt(
                copy_disagreements, llm2_name, llm1_name, language,
                common_prefix=common_prefix if common_prefix else None
            )

            response1, response2 = await self.generate_to_all_providers(prompt1, prompt2, images, session_id)

            llm1_results, llm2_results = _parse_dual_responses(response1, response2, "ultimatum")

            if len(llm1_results) == 0 and response1:
                llm1_parse_failed = True
            if len(llm2_results) == 0 and response2:
                llm2_parse_failed = True

            for d in copy_disagreements:
                key = f"copy_{d['copy_index']}_{d['question_id']}"
                results[key] = _resolve_ultimatum_grade(d, llm1_results, llm2_results, key)
                results[key]['used_cache'] = bool(common_prefix)  # True if prefix was used

        results['_parse_status'] = {'llm1_parse_failed': llm1_parse_failed, 'llm2_parse_failed': llm2_parse_failed}
        return results

    def clear(self):
        """Clear stored content and log cache statistics."""
        self._images_by_copy.clear()
        self._prefix_content.clear()

        # Log final cache statistics
        stats = self._cache_stats
        if stats["calls"] > 0:
            savings_pct = (stats["estimated_savings"] / stats["total_tokens"] * 100) if stats["total_tokens"] > 0 else 0
            logger.info(
                f"Cache statistics: {stats['calls']} calls, "
                f"{stats['cached_tokens']}/{stats['total_tokens']} tokens cached "
                f"(~{savings_pct:.0f}% savings, ~{stats['estimated_savings']} tokens saved)"
            )


class ExplicitCacheManager:
    """
    Manages explicit Gemini caching for batch grading.

    Unlike implicit caching (CacheManager), explicit caching creates a
    CachedContent resource that is shared across all verification/ultimatum
    calls for ALL copies, not just the first one.

    This solves the problem where implicit caching only works for the first
    copy (which sends images [1, 2, ..., N]) but not subsequent copies
    (which send different image ranges like [3, 4] or [5, 6]).

    Flow:
    1. Create cache with ALL images + barème → cache_id
    2. Verification/Ultimatum use cache_id + specific prompt
    3. Delete cache at the end

    Cost savings: ~50% reduction in input token costs
    """

    def __init__(self, providers: List[Tuple[str, Any]]):
        """
        Initialize ExplicitCacheManager.

        Args:
            providers: List of (name, provider) tuples
        """
        self.providers = providers
        self._cache_by_provider: Dict[str, str] = {}  # provider_name -> cache_id
        self._all_images: List[str] = []
        self._system_prompt: str = ""
        self._cache_stats: Dict[str, Any] = {
            "cache_created": False,
            "cache_hits": 0,
            "cache_misses": 0,
            "cached_tokens": 0,
        }

    async def create_cache(
        self,
        images: List[str],
        system_prompt: str,
        ttl_seconds: int = 1800
    ) -> bool:
        """
        Create explicit cache with all images for each provider.

        Args:
            images: List of all image paths to cache
            system_prompt: System prompt (barème, rules, etc.)
            ttl_seconds: Cache TTL (default: 30 minutes)

        Returns:
            True if cache was created for at least one provider
        """
        self._all_images = images
        self._system_prompt = system_prompt

        for name, provider in self.providers:
            if hasattr(provider, 'create_cached_context'):
                try:
                    cache_id = await asyncio.to_thread(
                        provider.create_cached_context,
                        system_prompt=system_prompt,
                        images=images,
                        ttl_seconds=ttl_seconds
                    )
                    if cache_id:
                        self._cache_by_provider[name] = cache_id
                        self._cache_stats["cache_created"] = True
                        logger.info(f"Created explicit cache for {name}: {cache_id}")
                    else:
                        logger.warning(f"Failed to create explicit cache for {name} (content too small?)")
                except Exception as e:
                    logger.warning(f"Failed to create explicit cache for {name}: {e}")

        if self._cache_by_provider:
            logger.info(
                f"Explicit cache ready for {len(self._cache_by_provider)} providers, "
                f"{len(images)} images, ~{len(system_prompt)//4 + len(images)*258} tokens"
            )
            return True
        else:
            logger.warning("Explicit cache creation failed for all providers")
            return False

    async def generate_with_cache(
        self,
        provider_name: str,
        prompt: str,
        images: List[str] = None
    ) -> Optional[str]:
        """
        Generate using cached context.

        Args:
            provider_name: Name of the provider to use
            prompt: User prompt (specific question)
            images: Optional additional images (usually None since all are cached)

        Returns:
            Response text, or None if cache not available
        """
        cache_id = self._cache_by_provider.get(provider_name)
        if not cache_id:
            self._cache_stats["cache_misses"] += 1
            logger.debug(f"No cache available for {provider_name}")
            return None

        # Find provider
        provider = None
        for name, p in self.providers:
            if name == provider_name:
                provider = p
                break

        if not provider:
            logger.warning(f"Provider {provider_name} not found")
            return None

        try:
            # Use generate_with_cache method
            if hasattr(provider, 'generate_with_cache'):
                response = await asyncio.to_thread(
                    provider.generate_with_cache,
                    cache_id,
                    prompt,
                    images
                )
                self._cache_stats["cache_hits"] += 1
                return response
            else:
                logger.warning(f"Provider {provider_name} doesn't support generate_with_cache")
                return None
        except Exception as e:
            logger.warning(f"Cache generation failed for {provider_name}: {e}")
            self._cache_stats["cache_misses"] += 1
            return None

    async def generate_to_all_providers(
        self,
        prompt1: str,
        prompt2: str,
        images: List[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate with both providers using explicit cache.

        Args:
            prompt1: Prompt for first provider
            prompt2: Prompt for second provider
            images: Optional additional images (usually None)

        Returns:
            Tuple of (response1, response2)
        """
        name1, _ = self.providers[0]
        name2, _ = self.providers[1]

        results = await asyncio.gather(
            self.generate_with_cache(name1, prompt1, images),
            self.generate_with_cache(name2, prompt2, images)
        )

        return results[0], results[1]

    async def cleanup(self):
        """Delete all caches from Gemini servers."""
        for name, cache_id in self._cache_by_provider.items():
            # Find provider
            provider = None
            for pname, p in self.providers:
                if pname == name:
                    provider = p
                    break

            if provider and hasattr(provider, 'delete_cached_context'):
                try:
                    await asyncio.to_thread(provider.delete_cached_context, cache_id)
                    logger.info(f"Deleted explicit cache for {name}: {cache_id}")
                except Exception as e:
                    logger.warning(f"Failed to delete cache for {name}: {e} (TTL will expire)")

        self._cache_by_provider.clear()

        # Log stats
        stats = self._cache_stats
        if stats["cache_created"]:
            logger.info(
                f"Explicit cache stats: {stats['cache_hits']} hits, "
                f"{stats['cache_misses']} misses"
            )

    @property
    def is_active(self) -> bool:
        """Check if explicit caching is active."""
        return len(self._cache_by_provider) > 0

    async def run_dual_llm_verification_with_cache(
        self,
        disagreements: List[Disagreement],
        language: str = "fr"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run verification phase using explicit caching.

        Unlike CacheManager, this uses a SINGLE shared cache for all copies,
        not per-copy prefixes. The cached content already contains all images
        and the barème, so prompts only need to include the specific disagreement
        details.

        Args:
            disagreements: List of Disagreement objects
            language: Language for prompts

        Returns:
            Dict mapping "copy_{idx}_{qid}" -> verification results
        """
        if not disagreements:
            return {}

        if not self.is_active:
            logger.warning("Explicit cache not active, cannot run verification")
            return {}

        llm1_name, provider1 = self.providers[0]
        llm2_name, provider2 = self.providers[1]
        results = {}

        from collections import defaultdict
        by_copy = defaultdict(list)
        for d in disagreements:
            if isinstance(d, Disagreement):
                by_copy[d.copy_index].append(d)
            else:
                by_copy[d['copy_index']].append(d)

        for copy_idx, copy_disagreements in by_copy.items():
            # Collect images for this copy (for fallback)
            copy_images = _collect_disagreement_images(copy_disagreements, copy_filter=copy_idx)

            # Build prompts WITHOUT common prefix (it's in the cached content)
            # The cache already has: system prompt + barème + ALL images
            prompt1 = build_dual_llm_verification_prompt(
                copy_disagreements, llm1_name, llm2_name, True, language,
                common_prefix=None  # No prefix - using explicit cache
            )
            prompt2 = build_dual_llm_verification_prompt(
                copy_disagreements, llm2_name, llm1_name, True, language,
                common_prefix=None  # No prefix - using explicit cache
            )

            # Try explicit cache first
            response1, response2 = await self.generate_to_all_providers(
                prompt1, prompt2, images=None
            )

            # Fallback to regular API calls if cache failed
            if response1 is None or response2 is None:
                logger.warning(f"Explicit cache miss for copy {copy_idx}, falling back to regular API")
                _, provider1 = self.providers[0]
                _, provider2 = self.providers[1]
                if response1 is None:
                    response1 = await asyncio.to_thread(
                        provider1.call_vision,
                        prompt1,
                        copy_images
                    )
                if response2 is None:
                    response2 = await asyncio.to_thread(
                        provider2.call_vision,
                        prompt2,
                        copy_images
                    )

            (llm1_q, _), (llm2_q, _) = _parse_dual_responses(response1, response2, "verification")

            for d in copy_disagreements:
                if isinstance(d, Disagreement):
                    key = f"copy_{d.copy_index}_{d.question_id}"
                else:
                    key = f"copy_{d['copy_index']}_{d['question_id']}"
                results[key] = _resolve_verification_grade(d, llm1_q, llm2_q, key)
                results[key]['used_explicit_cache'] = response1 is not None and response2 is not None

        return results

    async def run_dual_llm_ultimatum_with_cache(
        self,
        persistent_disagreements: List[Dict[str, Any]],
        language: str = "fr"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run ultimatum phase using explicit caching.

        Args:
            persistent_disagreements: List of disagreement dicts
            language: Language for prompts

        Returns:
            Dict mapping "copy_{idx}_{qid}" -> ultimatum results
        """
        if not persistent_disagreements:
            return {}

        if not self.is_active:
            logger.warning("Explicit cache not active, cannot run ultimatum")
            return {}

        llm1_name, _ = self.providers[0]
        llm2_name, _ = self.providers[1]
        results = {}

        from collections import defaultdict
        by_copy = defaultdict(list)
        for d in persistent_disagreements:
            by_copy[d['copy_index']].append(d)

        llm1_parse_failed = False
        llm2_parse_failed = False

        for copy_idx, copy_disagreements in by_copy.items():
            # Collect images for this copy (for fallback)
            copy_images = _collect_disagreement_images(copy_disagreements, copy_filter=copy_idx)

            # Build prompts WITHOUT common prefix (it's in the cached content)
            prompt1 = build_ultimatum_prompt(
                copy_disagreements, llm1_name, llm2_name, language,
                common_prefix=None  # No prefix - using explicit cache
            )
            prompt2 = build_ultimatum_prompt(
                copy_disagreements, llm2_name, llm1_name, language,
                common_prefix=None  # No prefix - using explicit cache
            )

            # Try explicit cache first
            response1, response2 = await self.generate_to_all_providers(
                prompt1, prompt2, images=None
            )

            # Fallback to regular API calls if cache failed
            if response1 is None or response2 is None:
                logger.warning(f"Explicit cache miss for copy {copy_idx} in ultimatum, falling back to regular API")
                _, provider1 = self.providers[0]
                _, provider2 = self.providers[1]
                if response1 is None:
                    response1 = await asyncio.to_thread(
                        provider1.call_vision,
                        prompt1,
                        copy_images
                    )
                if response2 is None:
                    response2 = await asyncio.to_thread(
                        provider2.call_vision,
                        prompt2,
                        copy_images
                    )

            llm1_results, llm2_results = _parse_dual_responses(response1, response2, "ultimatum")

            if len(llm1_results) == 0 and response1:
                llm1_parse_failed = True
            if len(llm2_results) == 0 and response2:
                llm2_parse_failed = True

            for d in copy_disagreements:
                key = f"copy_{d['copy_index']}_{d['question_id']}"
                results[key] = _resolve_ultimatum_grade(d, llm1_results, llm2_results, key)
                results[key]['used_explicit_cache'] = response1 is not None and response2 is not None

        results['_parse_status'] = {'llm1_parse_failed': llm1_parse_failed, 'llm2_parse_failed': llm2_parse_failed}
        return results

    def clear(self):
        """Clear cache (alias for cleanup for compatibility with CacheManager)."""
        # Note: cleanup is async, but clear is sync for CacheManager compatibility
        # Just clear the tracking dict, actual cleanup happens in async cleanup()
        self._cache_by_provider.clear()


# Backward compatibility alias
ChatContinuationManager = CacheManager

