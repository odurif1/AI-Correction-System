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
from utils.json_extractor import extract_json_from_response
from prompts.batch import (
    build_batch_grading_prompt,
    build_dual_llm_verification_prompt,
    build_ultimatum_prompt,
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

    With chat continuation:
        grader = BatchGrader(provider, use_chat=True)
        result = await grader.grade_batch(copies, questions, language="fr")
        # Subsequent verification/ultimatum calls use same session
    """

    def __init__(self, provider, use_chat: bool = False):
        """
        Initialize batch grader.

        Args:
            provider: LLM provider (must support multi-image calls)
            use_chat: If True, use chat sessions for conversation continuity
        """
        self.provider = provider
        self.use_chat = use_chat and provider.supports_chat()
        # Chat sessions: {session_id: chat_session}
        # session_id format: "batch_{batch_idx}" or "copy_{copy_idx}"
        self.chat_sessions: Dict[str, Any] = {}
        # Store initial results for continuation
        self._initial_results: Dict[str, BatchResult] = {}

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

    async def grade_batch_with_chat(
        self,
        copies: List[Dict[str, Any]],
        questions: Dict[str, Dict[str, Any]],
        language: str = "fr",
        detect_students: bool = False,
        session_id: str = None
    ) -> Tuple[BatchResult, str]:
        """
        Grade a batch using chat continuation.

        This method starts or continues a chat session, allowing the LLM
        to remember previous context for verification/ultimatum phases.

        Args:
            copies: List of copy data dicts
            questions: Dict of {question_id: {text, criteria, max_points}}
            language: Language for prompts
            detect_students: If True, ask LLM to detect multiple students
            session_id: Optional session ID (default: auto-generated)

        Returns:
            Tuple of (BatchResult, session_id)
        """
        start_time = time.time()

        # Generate session ID if not provided
        if session_id is None:
            session_id = f"batch_{len(self.chat_sessions)}"

        # Build prompt
        prompt = build_batch_grading_prompt(copies, questions, language, detect_students)

        # Collect all images
        all_images = []
        for copy in copies:
            all_images.extend(copy.get('image_paths', []))

        # Start or continue chat session
        if session_id not in self.chat_sessions:
            self.chat_sessions[session_id] = self.provider.start_chat()

        chat = self.chat_sessions[session_id]

        # Send message in chat
        try:
            raw_response = self.provider.send_chat_message(chat, prompt, all_images)
            if not isinstance(raw_response, str):
                raw_response = str(raw_response)
        except Exception as e:
            logger.error(f"Chat batch grading failed: {e}")
            return BatchResult(
                copies=[],
                patterns={},
                raw_response=str(e),
                parse_success=False,
                parse_errors=[f"Chat API call failed: {str(e)}"],
                duration_ms=(time.time() - start_time) * 1000
            ), session_id

        # Parse response
        result = self._parse_batch_response(raw_response, copies, start_time)

        # Store for continuation
        self._initial_results[session_id] = result

        return result, session_id

    def get_chat_session(self, session_id: str) -> Optional[Any]:
        """Get an existing chat session by ID."""
        return self.chat_sessions.get(session_id)

    def has_chat_session(self, session_id: str) -> bool:
        """Check if a chat session exists."""
        return session_id in self.chat_sessions

    def clear_chat_sessions(self):
        """Clear all chat sessions."""
        self.chat_sessions.clear()
        self._initial_results.clear()

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
                            'grade': _parse_grade_value(qdata.get('grade', 0)),
                            'max_points': _parse_grade_value(qdata.get('max_points', 1)),
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
    extra_images: List[str] = None,
    chat_manager: 'ChatContinuationManager' = None
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
        chat_manager: Optional ChatContinuationManager for chat continuation mode

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
    chat_manager: 'ChatContinuationManager' = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run verification for each disagreement separately with BOTH LLMs.

    Unlike run_dual_llm_verification (which groups all disagreements),
    this makes one call per disagreement per LLM (more granular).

    Args:
        providers: List of (name, provider) tuples
        disagreements: List of disagreements to verify
        language: Language for prompts
        chat_manager: Optional ChatContinuationManager for chat continuation mode

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
                    'llm1_max_points': d.llm1_max_points,
                    'llm2_max_points': d.llm2_max_points,
                    'max_points': d.max_points,
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
        llm1_max_pts = disagreement.llm1_max_points
        llm2_max_pts = disagreement.llm2_max_points
        max_points = disagreement.max_points
    else:
        llm1_grade = disagreement.get('llm1_grade', 0)
        llm2_grade = disagreement.get('llm2_grade', 0)
        llm1_max_pts = disagreement.get('llm1_max_points', 1.0)
        llm2_max_pts = disagreement.get('llm2_max_points', 1.0)
        max_points = disagreement.get('max_points', 1.0)

    # Get new grades
    llm1_new = llm1_question_results.get(key, {}).get('my_new_grade', llm1_grade)
    llm2_new = llm2_question_results.get(key, {}).get('my_new_grade', llm2_grade)

    # Get new max_points
    llm1_new_mp = llm1_question_results.get(key, {}).get('my_new_max_points')
    llm2_new_mp = llm2_question_results.get(key, {}).get('my_new_max_points')

    llm1_mp = llm1_new_mp if llm1_new_mp is not None else llm1_max_pts
    llm2_mp = llm2_new_mp if llm2_new_mp is not None else llm2_max_pts
    resolved_max_points = max(llm1_mp, llm2_mp)

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
        'max_points': max_points,
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
    max_pts = disagreement.get('max_points', 1.0)
    llm1_grade = disagreement.get('llm1_grade', 0)
    llm2_grade = disagreement.get('llm2_grade', 0)

    # Get final grades
    llm1_final = llm1_results.get(key, {}).get('my_final_grade', llm1_grade)
    llm2_final = llm2_results.get(key, {}).get('my_final_grade', llm2_grade)

    # Get max_points
    llm1_final_mp = llm1_results.get(key, {}).get('my_final_max_points')
    llm2_final_mp = llm2_results.get(key, {}).get('my_final_max_points')

    llm1_resolved_mp = llm1_final_mp if llm1_final_mp is not None else disagreement.get('llm1_max_points', max_pts)
    llm2_resolved_mp = llm2_final_mp if llm2_final_mp is not None else disagreement.get('llm2_max_points', max_pts)

    max_points_disagreement = abs(llm1_resolved_mp - llm2_resolved_mp) > 0.01

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

    # Check parsing success
    llm1_parsed = key in llm1_results
    llm2_parsed = key in llm2_results

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
        ),
        'resolved_max_points': max(llm1_resolved_mp, llm2_resolved_mp)
    }

    if max_points_disagreement:
        result['llm1_final_max_points'] = llm1_resolved_mp
        result['llm2_final_max_points'] = llm2_resolved_mp
        result['max_points_disagreement'] = True

    return result


async def _run_dual_llm_phase(
    providers: List[Tuple[str, Any]],
    disagreements: List[Union[Disagreement, Dict]],
    language: str,
    mode: str,
    batching: str,
    name_disagreements: List[Dict] = None,
    extra_images: List[str] = None,
    chat_manager: 'ChatContinuationManager' = None
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
        chat_manager: Optional ChatContinuationManager for chat continuation mode

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

    # Check if we should use chat continuation
    use_chat = chat_manager is not None

    # If chat continuation is enabled, delegate to the chat manager
    if use_chat:
        logger.info(f"Using chat continuation for {mode} phase ({batching} mode)")

        if mode == "verification":
            return await chat_manager.run_dual_llm_verification_with_chat(
                disagreements, language
            )
        else:  # ultimatum
            return await chat_manager.run_dual_llm_ultimatum_with_chat(
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
                key = f"copy_{d.copy_index}_{d.question_id}"
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
    chat_manager: 'ChatContinuationManager' = None
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
        chat_manager: Optional ChatContinuationManager for chat continuation mode

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
    chat_manager: 'ChatContinuationManager' = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run ultimatum round with BOTH LLMs for persistent disagreements (grouped mode).

    All disagreements are resolved in a single call per LLM.

    Args:
        providers: List of (name, provider) tuples
        persistent_disagreements: List of disagreements that persist after verification
        language: Language for prompts
        chat_manager: Optional ChatContinuationManager for chat continuation mode

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
    chat_manager: 'ChatContinuationManager' = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run ultimatum for each disagreement separately with BOTH LLMs.

    Unlike run_dual_llm_ultimatum (which groups all disagreements),
    this makes one call per disagreement per LLM (more granular).

    Args:
        providers: List of (name, provider) tuples
        persistent_disagreements: List of disagreements that persist after verification
        language: Language for prompts
        chat_manager: Optional ChatContinuationManager for chat continuation mode

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
    chat_manager: 'ChatContinuationManager' = None
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
        chat_manager: Optional ChatContinuationManager for chat continuation mode

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
# CONTEXT CACHING + CHAT API FOR VERIFICATION/ULTIMATUM
# ═══════════════════════════════════════════════════════════════════════════════

# Minimum tokens required for Gemini context caching
MIN_CACHE_TOKENS = 2048


class ChatContinuationManager:
    """
    Combines Context Caching + Chat API for cost-efficient multi-phase grading.

    Supports two caching modes:
    - shared (batch mode): ONE cache for ALL copies (guarantees 2048+ tokens)
    - per_copy (individual mode): ONE cache PER copy (may fail if too small)

    Flow:
    1. Create cached context(s) based on mode
    2. Start chat sessions that reference the cache
    3. Verification: send message in chat (LLM sees cached context)
    4. Ultimatum: continue chat (LLM sees cached context + verification)

    Benefits:
    - Images cached = ~10x cheaper
    - Chat history = LLM remembers verification during ultimatum
    """

    def __init__(self, providers: List[Tuple[str, Any]], cache_mode: str = "shared"):
        """
        Initialize ChatContinuationManager.

        Args:
            providers: List of (name, provider) tuples
            cache_mode: "shared" for batch mode, "per_copy" for individual mode
        """
        self.providers = providers
        self.cache_mode = cache_mode
        # Cache IDs: {provider_name: {session_id: cache_id}} or {provider_name: cache_id} for shared
        self.caches_by_provider: Dict[str, Dict[str, str]] = {name: {} for name, _ in providers}
        # Chat sessions: {provider_name: {session_id: chat_session}}
        self.chats_by_provider: Dict[str, Dict[str, Any]] = {name: {} for name, _ in providers}
        self._caching_supported: Dict[str, bool] = {}
        self._images_by_copy: Dict[int, List[str]] = {}
        self._initial_prompt: str = ""

    def _check_caching_support(self, provider_name: str, provider: Any) -> bool:
        if provider_name not in self._caching_supported:
            self._caching_supported[provider_name] = provider.supports_context_caching()
        return self._caching_supported[provider_name]

    async def create_sessions(
        self,
        copies_data: List[Dict[str, Any]],
        initial_prompt: str
    ):
        """
        Create cached context(s) + start chat sessions.

        In "shared" mode: ONE cache for ALL copies
        In "per_copy" mode: ONE cache PER copy (if large enough)

        Args:
            copies_data: List with 'copy_index' and 'image_paths'
            initial_prompt: The batch grading prompt (will be cached)
        """
        self._initial_prompt = initial_prompt

        # Store images by copy
        for copy_data in copies_data:
            copy_idx = copy_data.get('copy_index', 0)
            images = copy_data.get('image_paths', [])
            self._images_by_copy[copy_idx] = images

        if self.cache_mode == "shared":
            await self._create_shared_cache(copies_data, initial_prompt)
        else:
            await self._create_per_copy_cache(copies_data, initial_prompt)

    async def _create_shared_cache(self, copies_data: List[Dict[str, Any]], initial_prompt: str):
        """Create ONE shared cache for all copies (batch mode)."""
        # Collect ALL images from ALL copies
        all_images = []
        for copy_data in copies_data:
            all_images.extend(copy_data.get('image_paths', []))

        # Create ONE shared cache per provider
        for name, provider in self.providers:
            cache_id = None
            if self._check_caching_support(name, provider):
                cache_id = provider.create_cached_context(
                    system_prompt=initial_prompt,
                    images=all_images,
                    ttl_seconds=3600
                )
                if cache_id:
                    self.caches_by_provider[name]["shared"] = cache_id
                    logger.info(f"Created shared cache for {name} ({len(all_images)} images)")

            # Start chat sessions for each copy (all reference the same cache)
            for copy_data in copies_data:
                copy_idx = copy_data.get('copy_index', 0)
                session_id = f"copy_{copy_idx}"

                try:
                    chat = provider.start_chat(
                        system_prompt=initial_prompt,
                        cached_context=cache_id
                    )
                    self.chats_by_provider[name][session_id] = chat
                except Exception as e:
                    logger.warning(f"Chat creation failed for {name}/{session_id}: {e}")

    async def _create_per_copy_cache(self, copies_data: List[Dict[str, Any]], initial_prompt: str):
        """Create ONE cache PER copy (individual mode)."""
        for copy_data in copies_data:
            copy_idx = copy_data.get('copy_index', 0)
            images = copy_data.get('image_paths', [])
            session_id = f"copy_{copy_idx}"

            for name, provider in self.providers:
                cache_id = None
                if self._check_caching_support(name, provider):
                    # Try to create cache for this copy only
                    cache_id = provider.create_cached_context(
                        system_prompt=initial_prompt,
                        images=images,
                        ttl_seconds=3600
                    )
                    if cache_id:
                        self.caches_by_provider[name][session_id] = cache_id
                        logger.info(f"Created per-copy cache for {name}/{session_id}")

                # Start chat session (with or without cache)
                try:
                    chat = provider.start_chat(
                        system_prompt=initial_prompt,
                        cached_context=cache_id
                    )
                    self.chats_by_provider[name][session_id] = chat
                except Exception as e:
                    logger.warning(f"Chat creation failed for {name}/{session_id}: {e}")

    async def send_in_chat(
        self,
        provider_name: str,
        provider: Any,
        session_id: str,
        prompt: str,
        images: List[str] = None
    ) -> Optional[str]:
        """Send message in chat session."""
        chat = self.chats_by_provider.get(provider_name, {}).get(session_id)

        if chat:
            try:
                return provider.send_chat_message(chat, prompt, images)
            except Exception as e:
                logger.warning(f"Chat message failed: {e}")

        # Fallback: regular call with all images
        copy_idx = int(session_id.replace("copy_", ""))
        all_images = self._images_by_copy.get(copy_idx, [])
        return await _call_provider_vision(provider, prompt, all_images)

    async def send_to_all_chats(
        self,
        session_id: str,
        prompt1: str,
        prompt2: str,
        images: List[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """Send to both providers' chat sessions."""
        name1, provider1 = self.providers[0]
        name2, provider2 = self.providers[1]

        return await asyncio.gather(
            self.send_in_chat(name1, provider1, session_id, prompt1, images),
            self.send_in_chat(name2, provider2, session_id, prompt2, images)
        )

    async def run_dual_llm_verification_with_chat(
        self,
        disagreements: List[Disagreement],
        language: str = "fr"
    ) -> Dict[str, Dict[str, Any]]:
        """Run verification using chat sessions."""
        if not disagreements:
            return {}

        llm1_name, _ = self.providers[0]
        llm2_name, _ = self.providers[1]
        results = {}

        from collections import defaultdict
        by_copy = defaultdict(list)
        for d in disagreements:
            by_copy[d.copy_index].append(d)

        for copy_idx, copy_disagreements in by_copy.items():
            session_id = f"copy_{copy_idx}"
            images = _collect_disagreement_images(copy_disagreements, copy_filter=copy_idx)

            prompt1 = build_dual_llm_verification_prompt(
                copy_disagreements, llm1_name, llm2_name, True, language
            )
            prompt2 = build_dual_llm_verification_prompt(
                copy_disagreements, llm2_name, llm1_name, True, language
            )

            response1, response2 = await self.send_to_all_chats(session_id, prompt1, prompt2, images)

            (llm1_q, _), (llm2_q, _) = _parse_dual_responses(response1, response2, "verification")

            for d in copy_disagreements:
                key = f"copy_{d.copy_index}_{d.question_id}"
                results[key] = _resolve_verification_grade(d, llm1_q, llm2_q, key)
                results[key]['chat_continuation'] = True

        return results

    async def run_dual_llm_ultimatum_with_chat(
        self,
        persistent_disagreements: List[Dict[str, Any]],
        language: str = "fr"
    ) -> Dict[str, Dict[str, Any]]:
        """Run ultimatum using same chat sessions (sees verification history)."""
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

            prompt1 = build_ultimatum_prompt(copy_disagreements, llm1_name, llm2_name, language)
            prompt2 = build_ultimatum_prompt(copy_disagreements, llm2_name, llm1_name, language)

            # Same chat session - LLM sees verification history
            response1, response2 = await self.send_to_all_chats(session_id, prompt1, prompt2, images)

            llm1_results, llm2_results = _parse_dual_responses(response1, response2, "ultimatum")

            if len(llm1_results) == 0 and response1:
                llm1_parse_failed = True
            if len(llm2_results) == 0 and response2:
                llm2_parse_failed = True

            for d in copy_disagreements:
                key = f"copy_{d['copy_index']}_{d['question_id']}"
                results[key] = _resolve_ultimatum_grade(d, llm1_results, llm2_results, key)
                results[key]['chat_continuation'] = True

        results['_parse_status'] = {'llm1_parse_failed': llm1_parse_failed, 'llm2_parse_failed': llm2_parse_failed}
        return results

    def clear(self):
        """Clear all caches and sessions."""
        for name in self.caches_by_provider:
            self.caches_by_provider[name].clear()
        for name in self.chats_by_provider:
            self.chats_by_provider[name].clear()
        self._images_by_copy.clear()
