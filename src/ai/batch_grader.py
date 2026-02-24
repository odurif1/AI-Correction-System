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
from typing import List, Dict, Any, Optional, Tuple

from config.settings import get_settings
from config.constants import MAX_RETRIES
from utils.json_extractor import extract_json_from_response
from prompts.batch import (
    build_batch_grading_prompt,
    build_dual_llm_verification_prompt,
    build_ultimatum_prompt,
)

logger = logging.getLogger(__name__)

# Retry configuration (base delay for exponential backoff)
RETRY_BASE_DELAY = 1.0  # seconds
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
    max_retries: int = MAX_RETRIES
) -> Optional[str]:
    """
    Call a provider's vision API with proper error handling and retry logic.

    This is a shared utility function that handles the common pattern of:
    - Calling provider.call_vision with optional images
    - Ensuring the response is a string
    - Handling exceptions gracefully
    - Retrying on temporary errors (503, 429, etc.)

    Args:
        provider: LLM provider instance
        prompt: The prompt to send
        images: Optional list of image paths
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        Raw response as string, or None if all attempts failed
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            raw_response = provider.call_vision(prompt, image_path=images or [])
            if not isinstance(raw_response, str):
                raw_response = str(raw_response)
            return raw_response
        except Exception as e:
            last_error = e
            error_str = str(e)

            # Check if we should retry
            if attempt < max_retries and _is_retryable_error(e):
                delay = RETRY_BASE_DELAY * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
            else:
                # Non-retryable error or max retries exceeded
                logger.error(f"Provider vision call failed: {e}")
                return None

    logger.error(f"All {max_retries + 1} attempts failed. Last error: {last_error}")
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "copy_index": self.copy_index,
            "student_name": self.student_name,
            "questions": self.questions,
            "overall_feedback": self.overall_feedback
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
        detect_students: bool = False
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

        Returns:
            BatchResult with all copy grades and detected patterns
        """
        start_time = time.time()

        # Build prompt
        prompt = build_batch_grading_prompt(copies, questions, language, detect_students)

        # Collect all images from all copies
        all_images = []
        for copy in copies:
            all_images.extend(copy.get('image_paths', []))

        # Call LLM with all images
        try:
            # Use call_vision directly to get raw response text
            raw_response = self.provider.call_vision(prompt, image_path=all_images)
            if not isinstance(raw_response, str):
                raw_response = str(raw_response)

        except Exception as e:
            logger.error(f"Batch grading API call failed: {e}")
            return BatchResult(
                copies=[],
                patterns={},
                raw_response=str(e),
                parse_success=False,
                parse_errors=[f"API call failed: {str(e)}"],
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

        # Try to extract JSON from response
        data = extract_json_from_response(raw_response)
        if data is None:
            parse_errors.append("No JSON object found in response")
        else:
            try:
                # Parse copies
                for copy_data in data.get('copies', []):
                    copy_index = copy_data.get('copy_index', 0)
                    student_name = copy_data.get('student_name')

                    # Parse questions
                    questions = {}
                    for qid, qdata in copy_data.get('questions', {}).items():
                        questions[qid] = {
                            'student_answer_read': qdata.get('student_answer_read', ''),
                            'grade': float(qdata.get('grade', 0)),
                            'max_points': float(qdata.get('max_points', 1)),
                            'confidence': float(qdata.get('confidence', 0.8)),
                            'reasoning': qdata.get('reasoning', ''),
                            'feedback': qdata.get('feedback', '')
                        }

                    copies_results.append(BatchCopyResult(
                        copy_index=copy_index,
                        student_name=student_name,
                        questions=questions,
                        overall_feedback=copy_data.get('overall_feedback', '')
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
    llm1_max_points: float  # Max points detected by LLM1
    llm1_reasoning: str
    llm1_reading: str
    llm2_name: str
    llm2_grade: float
    llm2_max_points: float  # Max points detected by LLM2
    llm2_reasoning: str
    llm2_reading: str
    difference: float
    max_points: float  # Max of both (for calculating relative thresholds)
    image_paths: List[str]  # Paths to the copy images
    disagreement_type: str = "grade"  # "grade", "reading", "max_points", or combination


def detect_disagreements(
    llm1_result: BatchResult,
    llm2_result: BatchResult,
    llm1_name: str,
    llm2_name: str,
    copies_data: List[Dict[str, Any]],
    threshold: Optional[float] = None
) -> List[Disagreement]:
    """
    Detect disagreements between two LLM batch results.

    Args:
        llm1_result: First LLM's batch result
        llm2_result: Second LLM's batch result
        llm1_name: Name of first LLM
        llm2_name: Name of second LLM
        copies_data: Original copies data with image paths
        threshold: Minimum difference as percentage of max_points (default from settings)

    Returns:
        List of Disagreement objects
    """
    if threshold is None:
        threshold = get_settings().grade_agreement_threshold

    disagreements = []

    # Build lookup for LLM2 results
    llm2_copies = {c.copy_index: c for c in llm2_result.copies}

    for llm1_copy in llm1_result.copies:
        copy_idx = llm1_copy.copy_index
        llm2_copy = llm2_copies.get(copy_idx)

        if not llm2_copy:
            continue

        # Get student name (prefer LLM1, fallback to LLM2)
        student_name = llm1_copy.student_name or llm2_copy.student_name or f"Élève {copy_idx}"

        # Get image paths for this copy
        copy_data = next((c for c in copies_data if c['copy_index'] == copy_idx), None)
        image_paths = copy_data.get('image_paths', []) if copy_data else []

        # Check each question
        for qid, q1_data in llm1_copy.questions.items():
            q2_data = llm2_copy.questions.get(qid)
            if not q2_data:
                continue

            grade1 = float(q1_data.get('grade', 0))
            grade2 = float(q2_data.get('grade', 0))
            llm1_max_pts = float(q1_data.get('max_points', 1.0))
            llm2_max_pts = float(q2_data.get('max_points', 1.0))
            max_points = max(llm1_max_pts, llm2_max_pts)
            diff = abs(grade1 - grade2)

            # Use relative threshold (percentage of max_points)
            relative_threshold = max_points * threshold

            # Get readings
            reading1 = q1_data.get('student_answer_read', '').strip().lower()
            reading2 = q2_data.get('student_answer_read', '').strip().lower()

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

            # Detect max_points disagreement (different barème detected)
            max_points_disagreement = llm1_max_pts != llm2_max_pts

            # Add to disagreements if any type
            if grade_disagreement or reading_disagreement or max_points_disagreement:
                # Determine type (can combine multiple)
                types = []
                if grade_disagreement:
                    types.append("grade")
                if reading_disagreement:
                    types.append("reading")
                if max_points_disagreement:
                    types.append("max_points")

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
                    llm1_max_points=llm1_max_pts,
                    llm1_reasoning=q1_data.get('reasoning', ''),
                    llm1_reading=q1_data.get('student_answer_read', ''),
                    llm2_name=llm2_name,
                    llm2_grade=grade2,
                    llm2_max_points=llm2_max_pts,
                    llm2_reasoning=q2_data.get('reasoning', ''),
                    llm2_reading=q2_data.get('student_answer_read', ''),
                    difference=diff,
                    max_points=max_points,
                    image_paths=image_paths,
                    disagreement_type=disp_type
                ))

    return disagreements


async def run_dual_llm_verification(
    providers: List[tuple],
    disagreements: List[Disagreement],
    language: str = "fr",
    name_disagreements: List[Dict[str, Any]] = None,
    extra_images: List[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run verification with BOTH LLMs seeing each other's work.

    Args:
        providers: List of (name, provider) tuples
        disagreements: List of disagreements to verify
        language: Language for prompts
        name_disagreements: Optional list of student name disagreements (for grouped mode)
        extra_images: Optional list of additional images (for name-only cases)

    Returns:
        Dict with:
        - "copy_{idx}_{qid}" -> {final_grade, llm1_grade, llm2_grade, method, ...}
        - "name_{idx}" -> {llm1_new_name, llm2_new_name, resolved_name, agreement} for name verifications
    """
    if not disagreements and not name_disagreements:
        return {}

    llm1_name, llm1_provider = providers[0]
    llm2_name, llm2_provider = providers[1]

    # Collect all unique image paths from grade disagreements
    all_images = []
    seen = set()
    for d in disagreements:
        for img in d.image_paths:
            if img not in seen:
                all_images.append(img)
                seen.add(img)

    # Add extra images (for name-only disagreements case)
    if extra_images:
        for img in extra_images:
            if img not in seen:
                all_images.append(img)
                seen.add(img)

    # Add provider info to name disagreements for prompt building
    name_disags_with_providers = []
    if name_disagreements:
        for nd in name_disagreements:
            name_disags_with_providers.append({
                **nd,
                'llm1_provider': llm1_name,
                'llm2_provider': llm2_name
            })

    # Build prompts for each LLM
    llm1_prompt = build_dual_llm_verification_prompt(
        disagreements, llm1_name, llm2_name, is_own_perspective=True, language=language,
        name_disagreements=name_disags_with_providers if name_disags_with_providers else None
    )
    llm2_prompt = build_dual_llm_verification_prompt(
        disagreements, llm2_name, llm1_name, is_own_perspective=True, language=language,
        name_disagreements=name_disags_with_providers if name_disags_with_providers else None
    )

    # Call both LLMs in parallel
    llm1_response, llm2_response = await asyncio.gather(
        _call_provider_vision(llm1_provider, llm1_prompt, all_images),
        _call_provider_vision(llm2_provider, llm2_prompt, all_images)
    )

    # Parse responses - now returns tuple of (question_results, name_results)
    llm1_question_results, llm1_name_results = _parse_verification_response(llm1_response)
    llm2_question_results, llm2_name_results = _parse_verification_response(llm2_response)

    # Merge question verification results (grades, readings, max_points)
    results = {}
    for d in disagreements:
        key = f"copy_{d.copy_index}_{d.question_id}"

        # Get new grades from each LLM
        llm1_new = llm1_question_results.get(key, {}).get('my_new_grade', d.llm1_grade)
        llm2_new = llm2_question_results.get(key, {}).get('my_new_grade', d.llm2_grade)

        # Get new max_points from each LLM (if provided)
        llm1_new_mp = llm1_question_results.get(key, {}).get('my_new_max_points')
        llm2_new_mp = llm2_question_results.get(key, {}).get('my_new_max_points')

        # Resolve max_points: use new values if provided, otherwise use original
        llm1_mp = llm1_new_mp if llm1_new_mp is not None else d.llm1_max_points
        llm2_mp = llm2_new_mp if llm2_new_mp is not None else d.llm2_max_points
        resolved_max_points = max(llm1_mp, llm2_mp)

        # Resolve: if both agree now, use that; otherwise average
        if abs(llm1_new - llm2_new) < get_agreement_threshold(d.max_points):
            final_grade = (llm1_new + llm2_new) / 2
            method = "verification_consensus"
        else:
            final_grade = (llm1_new + llm2_new) / 2
            method = "verification_average"

        # Get new readings if provided
        llm1_new_reading = llm1_question_results.get(key, {}).get('my_new_reading')
        llm2_new_reading = llm2_question_results.get(key, {}).get('my_new_reading')

        # Get feedback from both LLMs - prefer the one that changed their mind or has better feedback
        llm1_feedback = llm1_question_results.get(key, {}).get('feedback', '')
        llm2_feedback = llm2_question_results.get(key, {}).get('feedback', '')

        # Consolidate feedback: prefer non-empty, or use LLM2's if LLM1 changed their grade more
        if llm1_feedback and llm2_feedback:
            # Both have feedback - use LLM1's as primary (it's the "base" LLM)
            final_feedback = llm1_feedback
        elif llm1_feedback:
            final_feedback = llm1_feedback
        elif llm2_feedback:
            final_feedback = llm2_feedback
        else:
            final_feedback = ''

        results[key] = {
            'final_grade': final_grade,
            'llm1_new_grade': llm1_new,
            'llm2_new_grade': llm2_new,
            'llm1_new_max_points': llm1_mp,
            'llm2_new_max_points': llm2_mp,
            'resolved_max_points': resolved_max_points,
            'llm1_new_reading': llm1_new_reading,
            'llm2_new_reading': llm2_new_reading,
            'llm1_reasoning': llm1_question_results.get(key, {}).get('reasoning', ''),
            'llm2_reasoning': llm2_question_results.get(key, {}).get('reasoning', ''),
            'llm1_feedback': llm1_feedback,
            'llm2_feedback': llm2_feedback,
            'final_feedback': final_feedback,
            'method': method,
            'max_points': d.max_points,
            'confidence': max(
                llm1_question_results.get(key, {}).get('confidence', 0.8),
                llm2_question_results.get(key, {}).get('confidence', 0.8)
            )
        }

    # Process name verifications if present
    if name_disagreements:
        for nd in name_disagreements:
            copy_idx = nd['copy_index']
            name_key = f"name_{copy_idx}"

            # Get new names from each LLM
            llm1_name_result = llm1_name_results.get(copy_idx, {})
            llm2_name_result = llm2_name_results.get(copy_idx, {})

            llm1_new_name = llm1_name_result.get('my_new_name', nd.get('llm1_name', ''))
            llm2_new_name = llm2_name_result.get('my_new_name', nd.get('llm2_name', ''))

            # Check agreement (normalized comparison)
            n1_normalized = llm1_new_name.lower().strip()
            n2_normalized = llm2_new_name.lower().strip()
            agreement = n1_normalized == n2_normalized and n1_normalized != ''

            # Resolve name: if agreement, use that; otherwise keep original disagreement for ultimatum
            if agreement:
                resolved_name = llm1_new_name  # They match, use either
            else:
                resolved_name = None  # Will need ultimatum or user resolution

            results[name_key] = {
                'llm1_new_name': llm1_new_name,
                'llm2_new_name': llm2_new_name,
                'resolved_name': resolved_name,
                'agreement': agreement,
                'confidence': max(
                    llm1_name_result.get('confidence', 0.8),
                    llm2_name_result.get('confidence', 0.8)
                )
            }

    return results


async def run_per_question_dual_verification(
    providers: List[tuple],
    disagreements: List[Disagreement],
    language: str = "fr"
) -> Dict[str, Dict[str, Any]]:
    """
    Run verification for each disagreement separately with BOTH LLMs.

    Unlike run_dual_llm_verification (which groups all disagreements),
    this makes one call per disagreement per LLM (more granular).

    Args:
        providers: List of (name, provider) tuples
        disagreements: List of disagreements to verify
        language: Language for prompts

    Returns:
        Dict mapping "copy_{idx}_{qid}" -> {final_grade, llm1_grade, llm2_grade, method, ...}
    """
    if not disagreements:
        return {}

    llm1_name, llm1_provider = providers[0]
    llm2_name, llm2_provider = providers[1]

    results = {}

    # Process each disagreement separately
    for d in disagreements:
        key = f"copy_{d.copy_index}_{d.question_id}"

        # Build prompts for this single disagreement
        llm1_prompt = build_dual_llm_verification_prompt(
            [d], llm1_name, llm2_name, is_own_perspective=True, language=language
        )
        llm2_prompt = build_dual_llm_verification_prompt(
            [d], llm2_name, llm1_name, is_own_perspective=True, language=language
        )

        # Call both LLMs in parallel for this disagreement
        llm1_response, llm2_response = await asyncio.gather(
            _call_provider_vision(llm1_provider, llm1_prompt, d.image_paths),
            _call_provider_vision(llm2_provider, llm2_prompt, d.image_paths)
        )

        # Parse responses (returns tuple, we only need question_results for per-question mode)
        llm1_question_results, _ = _parse_verification_response(llm1_response)
        llm2_question_results, _ = _parse_verification_response(llm2_response)

        # Get new grades from each LLM
        llm1_new = llm1_question_results.get(key, {}).get('my_new_grade', d.llm1_grade)
        llm2_new = llm2_question_results.get(key, {}).get('my_new_grade', d.llm2_grade)

        # Get new max_points from each LLM (if provided)
        llm1_new_mp = llm1_question_results.get(key, {}).get('my_new_max_points')
        llm2_new_mp = llm2_question_results.get(key, {}).get('my_new_max_points')

        # Resolve max_points: use new values if provided, otherwise use original
        llm1_mp = llm1_new_mp if llm1_new_mp is not None else d.llm1_max_points
        llm2_mp = llm2_new_mp if llm2_new_mp is not None else d.llm2_max_points
        resolved_max_points = max(llm1_mp, llm2_mp)

        # Resolve: if both agree now, use that; otherwise average
        if abs(llm1_new - llm2_new) < get_agreement_threshold(d.max_points):
            final_grade = (llm1_new + llm2_new) / 2
            method = "verification_consensus"
        else:
            final_grade = (llm1_new + llm2_new) / 2
            method = "verification_average"

        # Get and consolidate feedback
        llm1_feedback = llm1_question_results.get(key, {}).get('feedback', '')
        llm2_feedback = llm2_question_results.get(key, {}).get('feedback', '')
        if llm1_feedback and llm2_feedback:
            final_feedback = llm1_feedback
        elif llm1_feedback:
            final_feedback = llm1_feedback
        elif llm2_feedback:
            final_feedback = llm2_feedback
        else:
            final_feedback = ''

        results[key] = {
            'final_grade': final_grade,
            'llm1_new_grade': llm1_new,
            'llm2_new_grade': llm2_new,
            'llm1_new_max_points': llm1_mp,
            'llm2_new_max_points': llm2_mp,
            'resolved_max_points': resolved_max_points,
            'llm1_reasoning': llm1_question_results.get(key, {}).get('reasoning', ''),
            'llm2_reasoning': llm2_question_results.get(key, {}).get('reasoning', ''),
            'llm1_feedback': llm1_feedback,
            'llm2_feedback': llm2_feedback,
            'final_feedback': final_feedback,
            'method': method,
            'max_points': d.max_points,
            'confidence': max(
                llm1_question_results.get(key, {}).get('confidence', 0.8),
                llm2_question_results.get(key, {}).get('confidence', 0.8)
            ),
            'image_paths': d.image_paths  # Keep for potential ultimatum
        }

    return results


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
                'my_new_grade': float(v.get('my_new_grade', v.get('my_initial_grade', 0))),
                'my_new_max_points': float(v.get('my_new_max_points', 0)) if v.get('my_new_max_points') else None,
                'my_new_reading': v.get('my_new_reading'),
                'changed': v.get('changed', False),
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
                    'changed': nv.get('changed', False),
                    'confidence': float(nv.get('confidence', 0.8))
                }
    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"Failed to parse verification response: {e}")

    return question_results, name_results


# ═══════════════════════════════════════════════════════════════════════════════
# ULTIMATUM ROUND
# ═══════════════════════════════════════════════════════════════════════════════

async def run_dual_llm_ultimatum(
    providers: List[tuple],
    persistent_disagreements: List[Dict[str, Any]],
    language: str = "fr"
) -> Dict[str, Dict[str, Any]]:
    """
    Run ultimatum round with BOTH LLMs for persistent disagreements.

    Args:
        providers: List of (name, provider) tuples
        persistent_disagreements: List of disagreements that persist after verification
        language: Language for prompts

    Returns:
        Dict mapping "copy_{idx}_{qid}" -> {final_grade, llm1_grade, llm2_grade, method, ...}
    """
    if not persistent_disagreements:
        return {}

    llm1_name, llm1_provider = providers[0]
    llm2_name, llm2_provider = providers[1]

    # Collect all unique image paths
    all_images = []
    seen = set()
    for d in persistent_disagreements:
        for img in d.get('image_paths', []):
            if img not in seen:
                all_images.append(img)
                seen.add(img)

    # Build ultimatum prompts for each LLM
    llm1_prompt = build_ultimatum_prompt(
        persistent_disagreements, llm1_name, llm2_name, language=language
    )
    llm2_prompt = build_ultimatum_prompt(
        persistent_disagreements, llm2_name, llm1_name, language=language
    )

    # Call both LLMs in parallel
    llm1_response, llm2_response = await asyncio.gather(
        _call_provider_vision(llm1_provider, llm1_prompt, all_images),
        _call_provider_vision(llm2_provider, llm2_prompt, all_images)
    )

    # Parse responses
    llm1_results = _parse_ultimatum_response(llm1_response)
    llm2_results = _parse_ultimatum_response(llm2_response)

    # Track parsing failures
    llm1_parse_failed = len(llm1_results) == 0 and llm1_response
    llm2_parse_failed = len(llm2_results) == 0 and llm2_response
    if llm1_parse_failed:
        logger.warning(f"LLM1 ultimatum response parsing failed")
    if llm2_parse_failed:
        logger.warning(f"LLM2 ultimatum response parsing failed")

    # Merge results
    results = {}
    for d in persistent_disagreements:
        key = f"copy_{d['copy_index']}_{d['question_id']}"

        # Check if parsing failed for this specific disagreement
        llm1_parsed = key in llm1_results
        llm2_parsed = key in llm2_results

        # Get final grades from each LLM
        llm1_final = llm1_results.get(key, {}).get('my_final_grade', d['llm1_grade'])
        llm2_final = llm2_results.get(key, {}).get('my_final_grade', d['llm2_grade'])

        # Get max_points for relative threshold calculation
        max_pts = d.get('max_points', 1.0)

        # Get new max_points from each LLM (if provided)
        llm1_final_mp = llm1_results.get(key, {}).get('my_final_max_points')
        llm2_final_mp = llm2_results.get(key, {}).get('my_final_max_points')

        # Resolve max_points: use new values if provided, otherwise use original
        llm1_resolved_mp = llm1_final_mp if llm1_final_mp is not None else d.get('llm1_max_points', max_pts)
        llm2_resolved_mp = llm2_final_mp if llm2_final_mp is not None else d.get('llm2_max_points', max_pts)

        # Check if there's a max_points disagreement after ultimatum
        max_points_disagreement = abs(llm1_resolved_mp - llm2_resolved_mp) > 0.01

        # Resolve: if both agree now, use that; otherwise average
        if abs(llm1_final - llm2_final) < get_agreement_threshold(max_pts):
            final_grade = (llm1_final + llm2_final) / 2
            method = "ultimatum_consensus"
        else:
            final_grade = (llm1_final + llm2_final) / 2
            method = "ultimatum_average"

        # Detect flip-flop (grades swapped sides)
        initial_diff = d['llm1_grade'] - d['llm2_grade']
        ultimatum_diff = llm1_final - llm2_final
        is_swap = (
            (initial_diff > 0 and ultimatum_diff < 0) or
            (initial_diff < 0 and ultimatum_diff > 0)
        )
        # Use configurable threshold (0 = detect any swap)
        significant_diff = get_flip_flop_threshold(max_pts)
        flip_flop = (
            is_swap and
            abs(initial_diff) >= significant_diff and
            abs(ultimatum_diff) >= significant_diff
        )

        # Get and consolidate feedback
        llm1_feedback = llm1_results.get(key, {}).get('feedback', '')
        llm2_feedback = llm2_results.get(key, {}).get('feedback', '')
        if llm1_feedback and llm2_feedback:
            final_feedback = llm1_feedback
        elif llm1_feedback:
            final_feedback = llm1_feedback
        elif llm2_feedback:
            final_feedback = llm2_feedback
        else:
            final_feedback = ''

        # Build result dict
        result = {
            'final_grade': final_grade,
            'llm1_final_grade': llm1_final,
            'llm2_final_grade': llm2_final,
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
            )
        }

        # Only include max_points fields if there was a disagreement
        if max_points_disagreement:
            result['llm1_final_max_points'] = llm1_resolved_mp
            result['llm2_final_max_points'] = llm2_resolved_mp
            result['max_points_disagreement'] = True
            result['resolved_max_points'] = max(llm1_resolved_mp, llm2_resolved_mp)
        else:
            # Include resolved max_points even when agreed
            result['resolved_max_points'] = max(llm1_resolved_mp, llm2_resolved_mp)

        results[key] = result

    # Add overall parse status
    results['_parse_status'] = {
        'llm1_parse_failed': llm1_parse_failed,
        'llm2_parse_failed': llm2_parse_failed
    }

    return results


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
