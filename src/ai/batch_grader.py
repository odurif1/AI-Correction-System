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
from typing import List, Dict, Any, Optional

from config.settings import get_settings
from prompts.batch import (
    build_batch_grading_prompt,
    build_dual_llm_verification_prompt,
    build_ultimatum_prompt,
)

logger = logging.getLogger(__name__)


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
        language: str = "fr"
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

        Returns:
            BatchResult with all copy grades and detected patterns
        """
        start_time = time.time()

        # Build prompt
        prompt = build_batch_grading_prompt(copies, questions, language)

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
        try:
            # Find JSON in response
            json_match = raw_response
            if '```json' in raw_response:
                json_start = raw_response.find('```json') + 7
                json_end = raw_response.find('```', json_start)
                json_match = raw_response[json_start:json_end].strip()
            elif '```' in raw_response:
                json_start = raw_response.find('```') + 3
                json_end = raw_response.find('```', json_start)
                json_match = raw_response[json_start:json_end].strip()

            # Find the main JSON object
            brace_start = json_match.find('{')
            brace_end = json_match.rfind('}') + 1
            if brace_start >= 0 and brace_end > brace_start:
                json_str = json_match[brace_start:brace_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON object found in response")

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

        except json.JSONDecodeError as e:
            parse_errors.append(f"JSON parsing error: {str(e)}")
        except Exception as e:
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


async def grade_all_copies_in_batches(
    provider,
    copies: List[Dict[str, Any]],
    questions: Dict[str, Dict[str, Any]],
    max_pages_per_batch: int = 0,
    pages_per_copy: int = 2,
    language: str = "fr",
    progress_callback=None
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

    Returns:
        List of BatchResult, one per batch
    """
    grader = BatchGrader(provider)
    results = []

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

        result = await grader.grade_batch(batch_copies, questions, language)
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
    llm1_name: str
    llm1_grade: float
    llm1_reasoning: str
    llm1_reading: str
    llm2_name: str
    llm2_grade: float
    llm2_reasoning: str
    llm2_reading: str
    difference: float
    max_points: float  # For calculating relative thresholds
    image_paths: List[str]  # Paths to the copy images


def detect_disagreements(
    llm1_result: BatchResult,
    llm2_result: BatchResult,
    llm1_name: str,
    llm2_name: str,
    copies_data: List[Dict[str, Any]],
    threshold: float = 0.10
) -> List[Disagreement]:
    """
    Detect disagreements between two LLM batch results.

    Args:
        llm1_result: First LLM's batch result
        llm2_result: Second LLM's batch result
        llm1_name: Name of first LLM
        llm2_name: Name of second LLM
        copies_data: Original copies data with image paths
        threshold: Minimum difference as percentage of max_points (default 10%)

    Returns:
        List of Disagreement objects
    """
    disagreements = []

    # Build lookup for LLM2 results
    llm2_copies = {c.copy_index: c for c in llm2_result.copies}

    for llm1_copy in llm1_result.copies:
        copy_idx = llm1_copy.copy_index
        llm2_copy = llm2_copies.get(copy_idx)

        if not llm2_copy:
            continue

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
            max_points = max(
                float(q1_data.get('max_points', 1.0)),
                float(q2_data.get('max_points', 1.0))
            )
            diff = abs(grade1 - grade2)

            # Use relative threshold (percentage of max_points)
            relative_threshold = max_points * threshold

            if diff >= relative_threshold:
                disagreements.append(Disagreement(
                    copy_index=copy_idx,
                    question_id=qid,
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
                    image_paths=image_paths
                ))

    return disagreements


async def run_dual_llm_verification(
    providers: List[tuple],
    disagreements: List[Disagreement],
    language: str = "fr"
) -> Dict[str, Dict[str, Any]]:
    """
    Run verification with BOTH LLMs seeing each other's work.

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

    # Collect all unique image paths
    all_images = []
    seen = set()
    for d in disagreements:
        for img in d.image_paths:
            if img not in seen:
                all_images.append(img)
                seen.add(img)

    # Build prompts for each LLM
    llm1_prompt = build_dual_llm_verification_prompt(
        disagreements, llm1_name, llm2_name, is_own_perspective=True, language=language
    )
    llm2_prompt = build_dual_llm_verification_prompt(
        disagreements, llm2_name, llm1_name, is_own_perspective=True, language=language
    )

    # Call both LLMs in parallel
    async def call_provider(provider, prompt):
        try:
            raw_response = provider.call_vision(prompt, image_path=all_images)
            if not isinstance(raw_response, str):
                raw_response = str(raw_response)
            return raw_response
        except Exception as e:
            logger.error(f"Verification call failed: {e}")
            return None

    llm1_response, llm2_response = await asyncio.gather(
        call_provider(llm1_provider, llm1_prompt),
        call_provider(llm2_provider, llm2_prompt)
    )

    # Parse responses
    llm1_results = _parse_verification_response(llm1_response)
    llm2_results = _parse_verification_response(llm2_response)

    # Merge results
    results = {}
    for d in disagreements:
        key = f"copy_{d.copy_index}_{d.question_id}"

        # Get new grades from each LLM
        llm1_new = llm1_results.get(key, {}).get('my_new_grade', d.llm1_grade)
        llm2_new = llm2_results.get(key, {}).get('my_new_grade', d.llm2_grade)

        # Resolve: if both agree now, use that; otherwise average
        if abs(llm1_new - llm2_new) < get_agreement_threshold(d.max_points):
            final_grade = (llm1_new + llm2_new) / 2
            method = "verification_consensus"
        else:
            final_grade = (llm1_new + llm2_new) / 2
            method = "verification_average"

        results[key] = {
            'final_grade': final_grade,
            'llm1_new_grade': llm1_new,
            'llm2_new_grade': llm2_new,
            'llm1_reasoning': llm1_results.get(key, {}).get('reasoning', ''),
            'llm2_reasoning': llm2_results.get(key, {}).get('reasoning', ''),
            'method': method,
            'max_points': d.max_points,
            'confidence': max(
                llm1_results.get(key, {}).get('confidence', 0.8),
                llm2_results.get(key, {}).get('confidence', 0.8)
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
        async def call_provider(provider, prompt, images):
            try:
                raw_response = provider.call_vision(prompt, image_path=images)
                if not isinstance(raw_response, str):
                    raw_response = str(raw_response)
                return raw_response
            except Exception as e:
                logger.error(f"Per-question verification call failed: {e}")
                return None

        llm1_response, llm2_response = await asyncio.gather(
            call_provider(llm1_provider, llm1_prompt, d.image_paths),
            call_provider(llm2_provider, llm2_prompt, d.image_paths)
        )

        # Parse responses
        llm1_results = _parse_verification_response(llm1_response)
        llm2_results = _parse_verification_response(llm2_response)

        # Get new grades from each LLM
        llm1_new = llm1_results.get(key, {}).get('my_new_grade', d.llm1_grade)
        llm2_new = llm2_results.get(key, {}).get('my_new_grade', d.llm2_grade)

        # Resolve: if both agree now, use that; otherwise average
        if abs(llm1_new - llm2_new) < get_agreement_threshold(d.max_points):
            final_grade = (llm1_new + llm2_new) / 2
            method = "verification_consensus"
        else:
            final_grade = (llm1_new + llm2_new) / 2
            method = "verification_average"

        results[key] = {
            'final_grade': final_grade,
            'llm1_new_grade': llm1_new,
            'llm2_new_grade': llm2_new,
            'llm1_reasoning': llm1_results.get(key, {}).get('reasoning', ''),
            'llm2_reasoning': llm2_results.get(key, {}).get('reasoning', ''),
            'method': method,
            'max_points': d.max_points,
            'confidence': max(
                llm1_results.get(key, {}).get('confidence', 0.8),
                llm2_results.get(key, {}).get('confidence', 0.8)
            ),
            'image_paths': d.image_paths  # Keep for potential ultimatum
        }

    return results


def _parse_verification_response(raw_response: str) -> Dict[str, Dict]:
    """Parse a verification response from an LLM."""
    results = {}
    if not raw_response:
        return results

    try:
        json_match = raw_response
        if '```json' in raw_response:
            json_start = raw_response.find('```json') + 7
            json_end = raw_response.find('```', json_start)
            json_match = raw_response[json_start:json_end].strip()
        elif '```' in raw_response:
            json_start = raw_response.find('```') + 3
            json_end = raw_response.find('```', json_start)
            json_match = raw_response[json_start:json_end].strip()

        brace_start = json_match.find('{')
        brace_end = json_match.rfind('}') + 1
        if brace_start >= 0 and brace_end > brace_start:
            json_str = json_match[brace_start:brace_end]
            data = json.loads(json_str)

            for v in data.get('verifications', []):
                key = f"copy_{v.get('copy_index')}_{v.get('question_id')}"
                results[key] = {
                    'my_new_grade': float(v.get('my_new_grade', v.get('my_initial_grade', 0))),
                    'changed': v.get('changed', False),
                    'reasoning': v.get('reasoning', ''),
                    'confidence': float(v.get('confidence', 0.8))
                }
    except Exception as e:
        logger.error(f"Failed to parse verification response: {e}")

    return results


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
    async def call_provider(provider, prompt):
        try:
            raw_response = provider.call_vision(prompt, image_path=all_images)
            if not isinstance(raw_response, str):
                raw_response = str(raw_response)
            return raw_response
        except Exception as e:
            logger.error(f"Ultimatum call failed: {e}")
            return None

    llm1_response, llm2_response = await asyncio.gather(
        call_provider(llm1_provider, llm1_prompt),
        call_provider(llm2_provider, llm2_prompt)
    )

    # Parse responses
    llm1_results = _parse_ultimatum_response(llm1_response)
    llm2_results = _parse_ultimatum_response(llm2_response)

    # Merge results
    results = {}
    for d in persistent_disagreements:
        key = f"copy_{d['copy_index']}_{d['question_id']}"

        # Get final grades from each LLM
        llm1_final = llm1_results.get(key, {}).get('my_final_grade', d['llm1_grade'])
        llm2_final = llm2_results.get(key, {}).get('my_final_grade', d['llm2_grade'])

        # Get max_points for relative threshold calculation
        max_pts = d.get('max_points', 1.0)

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

        results[key] = {
            'final_grade': final_grade,
            'llm1_final_grade': llm1_final,
            'llm2_final_grade': llm2_final,
            'llm1_decision': llm1_results.get(key, {}).get('decision', 'unknown'),
            'llm2_decision': llm2_results.get(key, {}).get('decision', 'unknown'),
            'llm1_reasoning': llm1_results.get(key, {}).get('reasoning', ''),
            'llm2_reasoning': llm2_results.get(key, {}).get('reasoning', ''),
            'method': method,
            'flip_flop_detected': flip_flop,
            'confidence': max(
                llm1_results.get(key, {}).get('confidence', 0.8),
                llm2_results.get(key, {}).get('confidence', 0.8)
            )
        }

    return results


def _parse_ultimatum_response(raw_response: str) -> Dict[str, Dict]:
    """Parse an ultimatum response from an LLM."""
    results = {}
    if not raw_response:
        return results

    try:
        json_match = raw_response
        if '```json' in raw_response:
            json_start = raw_response.find('```json') + 7
            json_end = raw_response.find('```', json_start)
            json_match = raw_response[json_start:json_end].strip()
        elif '```' in raw_response:
            json_start = raw_response.find('```') + 3
            json_end = raw_response.find('```', json_start)
            json_match = raw_response[json_start:json_end].strip()

        brace_start = json_match.find('{')
        brace_end = json_match.rfind('}') + 1
        if brace_start >= 0 and brace_end > brace_start:
            json_str = json_match[brace_start:brace_end]
            data = json.loads(json_str)

            for u in data.get('ultimatum_decisions', []):
                key = f"copy_{u.get('copy_index')}_{u.get('question_id')}"
                results[key] = {
                    'my_final_grade': float(u.get('my_final_grade', 0)),
                    'decision': u.get('decision', 'unknown'),
                    'reasoning': u.get('reasoning', ''),
                    'confidence': float(u.get('confidence', 0.8))
                }
    except Exception as e:
        logger.error(f"Failed to parse ultimatum response: {e}")

    return results


# Keep the old functions for backward compatibility but mark as deprecated
async def run_grouped_verification(
    provider,
    disagreements: List[Disagreement],
    language: str = "fr"
) -> Dict[str, Dict[str, Any]]:
    """
    Run verification for all disagreements in a single API call.

    Args:
        provider: LLM provider
        disagreements: List of disagreements to verify
        language: Language for prompts

    Returns:
        Dict mapping "copy_{idx}_{qid}" -> {final_grade, reasoning, confidence}
    """
    if not disagreements:
        return {}

    # Collect all unique image paths
    all_images = []
    seen = set()
    for d in disagreements:
        for img in d.image_paths:
            if img not in seen:
                all_images.append(img)
                seen.add(img)

    # Build prompt
    prompt = build_grouped_verification_prompt(disagreements, language)

    # Call LLM
    try:
        raw_response = provider.call_vision(prompt, image_path=all_images)
        if not isinstance(raw_response, str):
            raw_response = str(raw_response)
    except Exception as e:
        logger.error(f"Grouped verification failed: {e}")
        return {}

    # Parse response
    results = {}
    try:
        # Extract JSON
        json_match = raw_response
        if '```json' in raw_response:
            json_start = raw_response.find('```json') + 7
            json_end = raw_response.find('```', json_start)
            json_match = raw_response[json_start:json_end].strip()
        elif '```' in raw_response:
            json_start = raw_response.find('```') + 3
            json_end = raw_response.find('```', json_start)
            json_match = raw_response[json_start:json_end].strip()

        brace_start = json_match.find('{')
        brace_end = json_match.rfind('}') + 1
        if brace_start >= 0 and brace_end > brace_start:
            json_str = json_match[brace_start:brace_end]
            data = json.loads(json_str)

            for resolution in data.get('resolutions', []):
                key = f"copy_{resolution.get('copy_index')}_{resolution.get('question_id')}"
                results[key] = {
                    'final_grade': float(resolution.get('final_grade', 0)),
                    'reasoning': resolution.get('reasoning', ''),
                    'confidence': float(resolution.get('confidence', 0.8))
                }
    except Exception as e:
        logger.error(f"Failed to parse grouped verification response: {e}")

    return results


async def run_per_question_verification(
    provider,
    disagreement: Disagreement,
    language: str = "fr"
) -> Dict[str, Any]:
    """
    Run verification for a single disagreement.

    Args:
        provider: LLM provider
        disagreement: The disagreement to verify
        language: Language for prompts

    Returns:
        Dict with final_grade, reasoning, confidence, student_answer_read
    """
    # Build prompt
    prompt = build_per_question_verification_prompt(disagreement, language)

    # Call LLM with only this copy's images
    try:
        raw_response = provider.call_vision(prompt, image_path=disagreement.image_paths)
        if not isinstance(raw_response, str):
            raw_response = str(raw_response)
    except Exception as e:
        logger.error(f"Per-question verification failed for copy {disagreement.copy_index}, {disagreement.question_id}: {e}")
        return {
            'final_grade': (disagreement.llm1_grade + disagreement.llm2_grade) / 2,
            'reasoning': 'Verification failed, using average',
            'confidence': 0.5
        }

    # Parse response
    try:
        json_match = raw_response
        if '```json' in raw_response:
            json_start = raw_response.find('```json') + 7
            json_end = raw_response.find('```', json_start)
            json_match = raw_response[json_start:json_end].strip()
        elif '```' in raw_response:
            json_start = raw_response.find('```') + 3
            json_end = raw_response.find('```', json_start)
            json_match = raw_response[json_start:json_end].strip()

        brace_start = json_match.find('{')
        brace_end = json_match.rfind('}') + 1
        if brace_start >= 0 and brace_end > brace_start:
            json_str = json_match[brace_start:brace_end]
            data = json.loads(json_str)
            return {
                'final_grade': float(data.get('final_grade', 0)),
                'reasoning': data.get('reasoning', ''),
                'confidence': float(data.get('confidence', 0.8)),
                'student_answer_read': data.get('student_answer_read', ''),
                'preferred_llm': data.get('preferred_llm', 'neither')
            }
    except Exception as e:
        logger.error(f"Failed to parse per-question verification response: {e}")

    return {
        'final_grade': (disagreement.llm1_grade + disagreement.llm2_grade) / 2,
        'reasoning': 'Parse failed, using average',
        'confidence': 0.5
    }
