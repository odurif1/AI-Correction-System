"""
Core session orchestrator.

Coordinates the entire grading workflow from PDF input to corrected output.
Provides phased workflow for interactive correction.
"""

import asyncio
import re
import tempfile
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

from core.models import (
    GradingSession, CopyDocument, GradedCopy, ClassAnswerMap,
    TeacherDecision, SessionStatus, ConfidenceLevel, generate_id
)
from core.workflow_state import CorrectionState, WorkflowPhase
from ai import create_ai_provider
from ai.provider_factory import create_comparison_provider
from config.settings import get_settings
from analysis.cross_copy import CrossCopyAnalyzer
from grading.grader import IntelligentGrader
from calibration.retroactive import RetroactiveApplier
from calibration.consistency import ConsistencyDetector
from storage.session_store import SessionStore
from vision.pdf_reader import PDFReader
from export.analytics import DataExporter, AnalyticsGenerator
from prompts import detect_language
from utils.sorting import question_sort_key
from utils.json_extractor import extract_json_from_response


class GradingSessionOrchestrator:
    """
    Main orchestrator for the grading workflow.

    Provides both:
    1. Traditional `run()` method for non-interactive use
    2. Phased methods for interactive workflow:
       - analyze_only() → confirm_scale() → grade_all() → review_doubts() → apply_decisions() → export()
    """

    def __init__(
        self,
        pdf_paths: List[str] = None,
        session_id: str = None,
        disagreement_callback: callable = None,
        name_disagreement_callback: callable = None,
        reading_disagreement_callback: callable = None,
        skip_reading_consensus: bool = False,
        force_single_llm: bool = False,
        pages_per_copy: int = None,
        second_reading: bool = False,
        parallel: int = 6,
        grading_mode: str = None,  # "individual", "batch", or "hybrid" (required)
        batch_verify: str = None,  # "per-question" or "grouped" (required for batch dual)
        auto_detect_structure: bool = False,  # Pre-analyze PDF structure with both LLMs
        workflow_state: CorrectionState = None
    ):
        """
        Initialize the orchestrator.

        Args:
            pdf_paths: List of paths to student PDF copies (optional if resuming)
            session_id: Session ID to resume (optional)
            disagreement_callback: Optional async callback for LLM disagreements
                                   Signature: async def callback(question_id, question_text,
                                                                 llm1_name, llm1_result,
                                                                 llm2_name, llm2_result,
                                                                 max_points) -> Tuple[float, str]
            name_disagreement_callback: Optional async callback for name disagreements
                                        Signature: async def callback(llm1_result, llm2_result) -> str
            reading_disagreement_callback: Optional async callback for reading disagreements
                                           Signature: async def callback(llm1_result, llm2_result,
                                                                         question_text, image_path) -> str
            skip_reading_consensus: If True, skip the reading consensus phase (default: False, enabled by default)
            force_single_llm: If True, use single LLM mode even if comparison_mode is enabled in config
            pages_per_copy: If set, activates individual reading mode (PDF pre-split by page count)
            second_reading: If True, enables second reading (2 passes for Single LLM, re-reading instruction for Dual LLM)
            parallel: Number of copies to process in parallel (default: 6)
            grading_mode: "individual" (each copy separately) or "batch" (all copies in one call)
            batch_verify: "per-question" or "grouped" for batch verification
            auto_detect_structure: If True, pre-analyze PDF with both LLMs to detect structure (copies, pages, names, barème)
            workflow_state: Optional CorrectionState for tracking workflow state
        """
        from config.constants import DATA_DIR

        self.pdf_paths = pdf_paths or []
        self.base_dir = Path(DATA_DIR)
        self._disagreement_callback = disagreement_callback
        self._name_disagreement_callback = name_disagreement_callback
        self._reading_disagreement_callback = reading_disagreement_callback
        self._skip_reading_consensus = skip_reading_consensus
        self._force_single_llm = force_single_llm
        self._pages_per_copy = pages_per_copy
        self._second_reading = second_reading
        self._parallel = parallel
        self._grading_mode = grading_mode
        self._batch_verify = batch_verify
        self._auto_detect_structure = auto_detect_structure
        self._structure_pre_detected = False  # Set to True after auto-detect-structure phase

        # Store names and barème detected during grading (for cross-verification)
        self._grading_detected_names = {}  # {copy_id: {'llm1': name, 'llm2': name}}
        self._grading_detected_bareme = {}  # {question_id: {'llm1': points, 'llm2': points}}

        # Workflow state tracking
        self._workflow_state = workflow_state

        # Initialize session
        if session_id:
            self.session_id = session_id
            self.store = SessionStore(session_id)
            self.session = self.store.load_session()
            if not self.session:
                raise ValueError(f"Session {session_id} not found")
        else:
            self.session_id = generate_id()
            self.session = GradingSession(
                session_id=self.session_id,
                created_at=datetime.now(),
                status=SessionStatus.ANALYZING,
                pages_per_copy=pages_per_copy
            )
            self.store = SessionStore(self.session_id)

        # Initialize AI provider (single or comparison mode)
        settings = get_settings()
        if settings.comparison_mode and not force_single_llm:
            self.ai = create_comparison_provider(
                disagreement_callback=disagreement_callback
            )
            self._comparison_mode = True
        else:
            self.ai = create_ai_provider()
            self._comparison_mode = False

        # Initialize components
        self.analyzer = CrossCopyAnalyzer(self.ai)

        # Grading scale: max_points per question
        # Filled from LLM detection or user confirmation
        self.grading_scale: Dict[str, float] = {}

        # Store detected questions (text content)
        self.detected_questions: Dict[str, str] = {}

        # Track if scale was confirmed by user
        self._scale_confirmed_by_user = False

        # Analysis results (available after analyze_only)
        self._analysis_complete = False
        self._grading_complete = False

        # Callback to ask user for scale (set by main.py)
        self.scale_callback: Optional[callable] = None

    @property
    def workflow_state(self) -> CorrectionState:
        """Get the workflow state, creating a default if not set."""
        if self._workflow_state is None:
            self._workflow_state = CorrectionState(
                session_id=self.session_id,
                phase=WorkflowPhase.INITIALIZATION
            )
        return self._workflow_state

    def set_workflow_state(self, state: CorrectionState) -> None:
        """Update the workflow state."""
        self._workflow_state = state

    def update_workflow_phase(self, phase: WorkflowPhase) -> None:
        """Update the workflow phase."""
        self._workflow_state = self.workflow_state.with_phase(phase)

    # ==================== COMPLETE WORKFLOW ====================

    async def run(self) -> str:
        """
        Run the complete grading workflow (non-interactive).

        Returns:
            Session ID
        """
        # Phase 1: Load and analyze copies
        await self._load_copies_phase()

        # Phase 2: Cross-copy analysis
        await self._analyze_phase()

        # Phase 3: Intelligent grading
        await self._grading_phase()

        # Phase 4: Calibration
        await self._calibration_phase()

        # Phase 5: Export
        export_path = await self._export_phase()

        # Mark complete
        self.session.status = SessionStatus.COMPLETE
        self._save_sync()

        return self.session_id

    # ==================== PHASED WORKFLOW METHODS ====================

    async def analyze_only(self) -> Dict:
        """
        Phase 1: Analyze copies without grading.

        Loads PDFs, extracts content, detects questions.
        Scale is NOT detected here - it will be detected during grading.

        In INDIVIDUAL mode (--pages-per-student), questions are NOT detected
        here - they will be detected during the grading phase by the LLM.

        Returns:
            Dict with:
            - 'questions': Dict of {question_id: question_text}
            - 'copies_count': Number of copies analyzed
            - 'language': Detected language
            - 'questions_detected_during_grading': True if in individual mode
        """
        # Load copies
        await self._load_copies_phase()

        # Run cross-copy analysis
        await self._analyze_phase()

        # Extract detected questions (no scale - detected during grading)
        detected_questions = {}
        detected_language = 'fr'

        if self.session.copies:
            first_copy = self.session.copies[0]
            detected_language = first_copy.language or 'fr'

            for key in first_copy.content_summary.keys():
                # Skip all metadata keys
                if key.startswith('_'):
                    continue
                if key.endswith('_points') or key.endswith('_points_unknown') or key.endswith('_confidence'):
                    continue

                # Only accept valid question keys: Q1, Q2, Q3, etc.
                if not (key.startswith('Q') and key[1:].isdigit()):
                    continue

                # This is a valid question key
                detected_questions[key] = f"Question {key}"

        # In INDIVIDUAL mode, questions will be detected during grading
        # Also in BATCH mode without --auto-detect-structure, structure is detected during grading
        questions_detected_during_grading = False
        if not detected_questions:
            if self._pages_per_copy:
                # Individual mode with mechanical split
                questions_detected_during_grading = True
            elif not self._auto_detect_structure:
                # Batch mode without pre-analysis - structure detected during grading
                questions_detected_during_grading = True
            # No placeholder - LLM will detect questions dynamically
            detected_questions = {}

        # Default scale of 1.0 for each question (will be detected during grading)
        detected_scale = {q: 1.0 for q in detected_questions.keys()}

        self.grading_scale = detected_scale
        self.detected_questions = detected_questions
        self.scale_detected = False  # Scale will be detected during grading

        self._analysis_complete = True

        return {
            'questions': detected_questions,
            'scale': detected_scale,  # Default scale, will be updated during grading
            'scale_detected': False,  # Scale detected during grading, not analysis
            'copies_count': len(self.session.copies),
            'language': detected_language,
            'questions_detected_during_grading': questions_detected_during_grading
        }

    async def re_analyze_low_confidence(self) -> Dict[str, any]:
        """
        Re-analyze elements with low confidence scale detection.

        Returns:
            Dict with updated scale info
        """
        if not self.session.copies:
            return {}

        first_copy = self.session.copies[0]
        low_confidence = self.get_low_confidence_elements(first_copy)
        updated = {}

        for q_id in low_confidence:
            result = await self.re_analyze_element(first_copy, q_id, focus="scale")

            if result.get('parsed'):
                parsed = result['parsed']
                points_key = f"{q_id}_points"

                if points_key in parsed:
                    try:
                        points = float(parsed[points_key])
                        self.grading_scale[q_id] = points
                        updated[q_id] = {
                            'points': points,
                            'confidence': parsed.get(f"{q_id}_confidence", "moyen")
                        }

                        # Update the copy's content summary
                        first_copy.content_summary[points_key] = str(points)
                        if f"{q_id}_confidence" in parsed:
                            first_copy.content_summary[f"{q_id}_confidence"] = parsed[f"{q_id}_confidence"]
                    except (ValueError, TypeError):
                        pass

        return updated

    def confirm_scale(self, scale: Dict[str, float]) -> None:
        """
        Phase 2: Set the grading scale.

        Must be called after analyze_only() and before grade_all().

        Args:
            scale: Dict of {question_id: max_points}
                   Can be empty {} if scale will be detected during grading
        """
        if not self._analysis_complete:
            raise RuntimeError("Must call analyze_only() before confirm_scale()")

        self.grading_scale = scale
        self._scale_confirmed_by_user = bool(scale)  # True if user provided scale

        # Update policy with scale
        self.session.policy.question_weights = scale

        # Save
        self._save_sync()

    def update_scale_from_detection(self, detected_scale: Dict[str, float]):
        """
        Update grading scale from LLM detection.

        Only updates questions that weren't already confirmed by user.

        Args:
            detected_scale: Dict of {question_id: max_points} detected by LLM
        """
        for qid, max_pts in detected_scale.items():
            if max_pts and max_pts > 0:
                # Only update if not already confirmed by user
                if not self._scale_confirmed_by_user or qid not in self.grading_scale:
                    self.grading_scale[qid] = max_pts

        self._save_sync()

    def get_max_points(self, question_id: str) -> float:
        """
        Get max points for a question.

        Returns:
            Max points from grading_scale, or 1.0 as default
        """
        return self.grading_scale.get(question_id, 1.0)

    def get_total_max_points(self) -> float:
        """
        Get total max points for all known questions.

        Returns:
            Sum of all max points in grading_scale
        """
        return sum(self.grading_scale.values()) if self.grading_scale else 0.0

    def has_complete_scale(self, question_ids: List[str]) -> bool:
        """
        Check if all questions have a scale defined.

        Args:
            question_ids: List of question IDs to check

        Returns:
            True if all questions have max_points > 0
        """
        for qid in question_ids:
            if self.grading_scale.get(qid, 0) <= 0:
                return False
        return True

    def get_unknown_scale_questions(self) -> List[str]:
        """
        Get list of questions with unknown scale.

        Returns:
            List of question IDs that need scale values
        """
        unknown = []
        if self.session.copies:
            first_copy = self.session.copies[0]
            for key in first_copy.content_summary.keys():
                if key.endswith('_points_unknown'):
                    question_id = key.replace('_points_unknown', '')
                    unknown.append(question_id)
        return unknown

    async def grade_all(
        self,
        progress_callback: callable = None
    ) -> List[GradedCopy]:
        """
        Grade all copies using stateless architecture.

        Each call is independent with explicit context and images re-sent:
        - Single-pass grading for all questions (images sent)
        - Targeted verification only for disagreements (images RE-SENT)
        - Ultimatum round if needed (images RE-SENT again)

        For single LLM mode with second_reading: runs 2 passes for self-verification.

        For batch mode: all copies graded in one API call.

        Must be called after confirm_scale().

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            List of GradedCopy objects
        """
        if not self._analysis_complete:
            raise RuntimeError("Must call analyze_only() before grade_all()")

        # confirm_scale() should have been called (can be with empty scale)
        # We track this via _scale_confirmed_by_user being set
        if not hasattr(self, '_scale_confirmed_by_user'):
            raise RuntimeError("Must call confirm_scale() before grade_all()")

        # Check for batch mode
        if self._grading_mode == "batch":
            return await self._grade_all_batch(progress_callback)

        # Check for hybrid mode (only works with dual LLM)
        if self._grading_mode == "hybrid":
            if not self._comparison_mode:
                raise RuntimeError("Hybrid mode requires dual LLM (remove --single)")
            return await self._grade_all_hybrid(progress_callback)

        if self._comparison_mode:
            return await self._grade_all_dual_llm(progress_callback)
        else:
            return await self._grade_all_single_llm(progress_callback)

    async def _grade_all_single_llm(
        self,
        progress_callback: callable = None
    ) -> List[GradedCopy]:
        """
        Grade all copies using single LLM mode.

        If second_reading is enabled, uses self-verification (2 passes).
        Uses parallel processing for copies (configurable via --parallel).
        """
        from ai.single_pass_grader import SinglePassGrader

        self.session.status = SessionStatus.GRADING

        total_copies = len(self.session.copies)

        questions = []
        for q_id in sorted(self.detected_questions.keys(), key=question_sort_key):
            q_text = self.detected_questions[q_id]
            questions.append({
                "id": q_id,
                "text": q_text,
                "criteria": "",
                "max_points": self.grading_scale.get(q_id, 1.0)
            })

        # Semaphore to limit concurrent grading
        semaphore = asyncio.Semaphore(self._parallel)

        async def grade_one_copy(i: int, copy: CopyDocument) -> Optional[GradedCopy]:
            """Grade a single copy with semaphore for concurrency control."""
            async with semaphore:
                if progress_callback:
                    await self._call_callback(progress_callback, 'copy_start', {
                        'copy_index': i + 1,
                        'total_copies': total_copies,
                        'copy_id': copy.id,
                        'student_name': copy.student_name,
                        'questions': list(self.grading_scale.keys())
                    })

                image_paths = [str(p) for p in copy.page_images] if copy.page_images else []
                if not image_paths:
                    return None

                if progress_callback:
                    await self._call_callback(progress_callback, 'single_pass_start', {
                        'num_questions': len(questions),
                        'providers': ['single_llm']
                    })

                grader = SinglePassGrader(self.ai)

                # Use self-verification if second_reading is enabled
                if self._second_reading:
                    result, verification_audit = await grader.grade_with_self_verification(
                        questions, image_paths, "fr"
                    )
                else:
                    result = await grader.grade_all_questions(
                        questions, image_paths, "fr"
                    )
                    verification_audit = None

                # Convert to GradedCopy
                graded = GradedCopy(
                    copy_id=copy.id,
                    policy_version=self.session.policy.version
                )

                # Populate in natural order (Q1, Q2, ... Q10, Q11)
                for q_id in sorted(result.questions.keys(), key=question_sort_key):
                    q_result = result.questions[q_id]
                    graded.grades[q_id] = q_result.grade
                    graded.confidence_by_question[q_id] = q_result.confidence
                    graded.student_feedback[q_id] = q_result.feedback
                    graded.readings[q_id] = q_result.student_answer_read
                    graded.max_points_by_question[q_id] = q_result.max_points
                    if q_result.reasoning:
                        graded.reasoning[q_id] = q_result.reasoning

                # Update student name from grading result (more accurate than analysis phase)
                if result.student_name:
                    copy.student_name = result.student_name

                graded.total_score = sum(graded.grades.values())
                graded.max_score = sum(graded.max_points_by_question.values())
                graded.confidence = sum(graded.confidence_by_question.values()) / len(graded.confidence_by_question) if graded.confidence_by_question else 0.5

                # Store audit data
                graded.llm_comparison = {
                    "method": "single_llm",
                    "options": {
                        "second_reading": self._second_reading
                    },
                    "self_verification": verification_audit,
                    "duration_ms": result.duration_ms
                }

                # Notify per question
                for q_id in sorted(graded.grades.keys(), key=question_sort_key):
                    if progress_callback:
                        await self._call_callback(progress_callback, 'question_done', {
                            'copy_index': i + 1,
                            'question_id': q_id,
                            'grade': graded.grades[q_id],
                            'max_points': graded.max_points_by_question.get(q_id, 1.0),
                            'method': 'single_llm_second_reading' if self._second_reading else 'single_llm',
                            'agreement': True
                        })

                if progress_callback:
                    token_usage = None
                    if hasattr(self.ai, 'get_token_usage'):
                        token_usage = self.ai.get_token_usage()

                    await self._call_callback(progress_callback, 'copy_done', {
                        'copy_index': i + 1,
                        'total_copies': total_copies,
                        'copy_id': copy.id,
                        'student_name': copy.student_name,
                        'total_score': graded.total_score,
                        'max_score': graded.max_score,
                        'confidence': graded.confidence,
                        'token_usage': token_usage
                    })

                return graded

        # Create tasks for all copies
        tasks = [
            grade_one_copy(i, copy)
            for i, copy in enumerate(self.session.copies)
        ]

        # Execute all tasks in parallel (limited by semaphore)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful results and emit error events
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                import logging
                logging.error(f"Error grading copy {i+1}: {result}")
                # Emit copy_error event for live display
                if progress_callback:
                    copy = self.session.copies[i] if i < len(self.session.copies) else None
                    await self._call_callback(progress_callback, 'copy_error', {
                        'copy_index': i + 1,
                        'total_copies': total_copies,
                        'copy_id': copy.id if copy else None,
                        'student_name': copy.student_name if copy else None,
                        'error': str(result)
                    })
            elif result is not None:
                self.session.graded_copies.append(result)
                self._save_sync()

        self._grading_complete = True
        return self.session.graded_copies

    async def _grade_all_batch(
        self,
        progress_callback: callable = None
    ) -> List[GradedCopy]:
        """
        Grade all copies in batch mode - one API call for all copies.

        Benefits:
        - Consistency: same answer = same grade (LLM sees all copies)
        - Pattern detection: LLM can identify common answers and outliers
        - Efficiency: 1 API call instead of N (or 2 with dual LLM)

        This is the recommended mode for:
        - Small classes (< 20 copies)
        - When consistency across copies is critical
        """
        from ai.batch_grader import grade_all_copies_in_batches, BatchResult
        import logging

        logger = logging.getLogger(__name__)
        total_copies = len(self.session.copies)

        # Notify start
        if progress_callback:
            await self._call_callback(progress_callback, 'batch_start', {
                'total_copies': total_copies,
                'mode': 'batch'
            })

        # Prepare copies data for batch grading
        copies_data = []
        for i, copy in enumerate(self.session.copies):
            # Get image paths for this copy
            image_paths = []
            if copy.page_images:
                image_paths = copy.page_images
            elif copy.pdf_path:
                # Generate images from PDF
                from vision.pdf_reader import PDFReader
                reader = PDFReader(copy.pdf_path)
                temp_dir = Path(self.store.session_dir) / "temp"
                temp_dir.mkdir(exist_ok=True)

                for page_num in range(copy.page_count):
                    image_path = str(temp_dir / f"batch_copy_{i+1}_page_{page_num}.png")
                    image_bytes = reader.get_page_image_bytes(page_num)
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                    image_paths.append(image_path)
                reader.close()

            copies_data.append({
                'copy_index': i + 1,
                'image_paths': image_paths,
                'student_name': copy.student_name
            })

            # Emit copy_start BEFORE API call so display shows "processing" during grading
            if progress_callback:
                await self._call_callback(progress_callback, 'copy_start', {
                    'copy_index': i + 1,
                    'total_copies': total_copies,
                    'copy_id': copy.id,
                    'student_name': copy.student_name or '???',
                    'questions': []  # Unknown until API returns
                })

        # Build questions dict
        questions = {}
        for qid, max_pts in self.grading_scale.items():
            questions[qid] = {
                'text': '',  # Will be detected from images
                'criteria': '',
                'max_points': max_pts
            }

        # Determine language
        language = 'fr'  # Default, could be from session

        graded_copies = []
        llm_comparison_data = {}  # Store comparison data for audit

        if self._comparison_mode:
            # Dual LLM batch: run both in parallel, then compare
            provider_names = [name for name, _ in self.ai.providers]
            llm1_name, llm2_name = provider_names[0], provider_names[1]

            if progress_callback:
                await self._call_callback(progress_callback, 'batch_llm_start', {
                    'providers': provider_names
                })

            # Run batch grading with both LLMs
            async def grade_with_provider(provider, name):
                from ai.batch_grader import BatchGrader
                grader = BatchGrader(provider)
                return await grader.grade_batch(copies_data, questions, language)

            tasks = [
                grade_with_provider(provider, name)
                for name, provider in self.ai.providers
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            llm1_result = results[0] if not isinstance(results[0], Exception) else None
            llm2_result = results[1] if not isinstance(results[1], Exception) else None

            if progress_callback:
                await self._call_callback(progress_callback, 'batch_llm_done', {
                    'llm1_success': llm1_result is not None and llm1_result.parse_success,
                    'llm2_success': llm2_result is not None and llm2_result.parse_success
                })

            # Check if at least one succeeded
            if not ((llm1_result and llm1_result.parse_success) or (llm2_result and llm2_result.parse_success)):
                logger.error("Both LLM batch grading failed")
                return []

            # ===== CROSS-VERIFY STUDENT NAMES =====
            from utils.name_matching import cross_verify_student_names, format_name_mismatch_message

            name_result = cross_verify_student_names(
                llm1_result.copies if llm1_result else [],
                llm2_result.copies if llm2_result else []
            )

            # Store name verification in comparison data for audit
            llm_comparison_data = {
                "options": {
                    "mode": "batch",
                    "providers": provider_names,
                    "total_copies": total_copies
                },
                "name_verification": {
                    "matches": name_result.matches,
                    "mismatches": name_result.mismatches,
                    "llm1_only": name_result.llm1_only,
                    "llm2_only": name_result.llm2_only,
                    "all_matched": name_result.all_matched
                },
                "llm_comparison": {}
            }

            # If names don't match, stop and inform user (unless auto-confirm)
            if name_result.requires_user_action:
                from core.exceptions import StudentNameMismatchError

                msg = format_name_mismatch_message(name_result, language)

                # Check if we're in auto mode
                is_auto_mode = self.workflow_state.auto_mode if self.workflow_state else False

                if is_auto_mode:
                    # Auto mode: print warning and continue
                    print(f"\n{msg}")
                    print("\n[WARNING] Continuing despite name mismatch (--auto-confirm mode)")
                    llm_comparison_data["name_verification_warning"] = True
                else:
                    # Interactive mode: raise exception to stop execution
                    raise StudentNameMismatchError(
                        msg,
                        mismatches=name_result.mismatches,
                        llm1_only=name_result.llm1_only,
                        llm2_only=name_result.llm2_only
                    )

            # Build merged results with comparison data (preserve name_verification)
            batch_result = None
            name_verification = llm_comparison_data.get("name_verification", {})
            llm_comparison_data = {
                "options": {
                    "mode": "batch",
                    "providers": provider_names,
                    "total_copies": total_copies
                },
                "name_verification": name_verification,
                "llm_comparison": {}
            }

            # Use LLM1 as base if available, else LLM2
            if llm1_result and llm1_result.parse_success:
                batch_result = llm1_result
            else:
                batch_result = llm2_result

            # For each copy, build comparison data
            for copy_result in batch_result.copies:
                copy_idx = copy_result.copy_index
                llm_comparison_data["llm_comparison"][f"copy_{copy_idx}"] = {
                    "student_name": copy_result.student_name,
                    "questions": {}
                }

                # Find corresponding result from other LLM
                llm2_copy = None
                if llm2_result and llm2_result.parse_success:
                    for c in llm2_result.copies:
                        if c.copy_index == copy_idx:
                            llm2_copy = c
                            break

                # For each question, store both LLM responses
                for qid, qdata in copy_result.questions.items():
                    llm1_qdata = {
                        "grade": qdata.get('grade', 0),
                        "max_points": qdata.get('max_points', 1),
                        "reading": qdata.get('student_answer_read', ''),
                        "reasoning": qdata.get('reasoning', ''),
                        "feedback": qdata.get('feedback', ''),
                        "confidence": qdata.get('confidence', 0.8)
                    }

                    llm2_qdata = {}
                    if llm2_copy and qid in llm2_copy.questions:
                        q2 = llm2_copy.questions[qid]
                        llm2_qdata = {
                            "grade": q2.get('grade', 0),
                            "max_points": q2.get('max_points', 1),
                            "reading": q2.get('student_answer_read', ''),
                            "reasoning": q2.get('reasoning', ''),
                            "feedback": q2.get('feedback', ''),
                            "confidence": q2.get('confidence', 0.8)
                        }

                    # Calculate final grade (average if both available, else use one)
                    if llm2_qdata:
                        final_grade = (llm1_qdata["grade"] + llm2_qdata["grade"]) / 2
                        # Agreement threshold: configurable % of max_points (relative threshold)
                        max_pts = max(llm1_qdata.get("max_points", 1.0), llm2_qdata.get("max_points", 1.0))
                        threshold = max_pts * get_settings().grade_agreement_threshold
                        grade_agreement = abs(llm1_qdata["grade"] - llm2_qdata["grade"]) < threshold
                        # Also check max_points agreement (barème must match)
                        max_points_agreement = abs(llm1_qdata.get("max_points", 1.0) - llm2_qdata.get("max_points", 1.0)) < 0.01
                        agreement = grade_agreement and max_points_agreement
                    else:
                        final_grade = llm1_qdata["grade"]
                        max_pts = llm1_qdata.get("max_points", 1.0)
                        agreement = True

                    # Build question data - structure: max_points, LLM1, LLM2, (verification), (ultimatum), final
                    # We'll add final at the end after all processing
                    question_data = {
                        "max_points": max_pts,  # Max points at root level
                        llm1_name: llm1_qdata,
                    }
                    if llm2_qdata:
                        question_data[llm2_name] = llm2_qdata

                    # Store initial final data (will be moved to end later)
                    question_data["_initial_final"] = {
                        "grade": final_grade,
                        "max_points": max_pts,
                        "method": "average" if llm2_qdata else "single_llm",
                        "agreement": agreement
                    }

                    llm_comparison_data["llm_comparison"][f"copy_{copy_idx}"]["questions"][qid] = question_data

                    # NOTE: Do NOT update qdata['grade'] here!
                    # We need the original grades for Disagreement creation.
                    # Grades will be updated after verification/ultimatum.

            # Notify CLI that comparison data is ready (before verification)
            if progress_callback:
                # Build a display-friendly summary of the comparison
                comparison_summary = {
                    'providers': [llm1_name, llm2_name],
                    'copies': []
                }
                for copy_idx, copy_data in llm_comparison_data.get("llm_comparison", {}).items():
                    copy_summary = {
                        'copy_index': copy_idx,
                        'student_name': copy_data.get('student_name'),
                        'questions': {}
                    }
                    for qid, qdata in copy_data.get('questions', {}).items():
                        llm1_q = qdata.get(llm1_name, {})
                        llm2_q = qdata.get(llm2_name, {})
                        copy_summary['questions'][qid] = {
                            'llm1_grade': llm1_q.get('grade'),
                            'llm1_max_points': llm1_q.get('max_points'),
                            'llm2_grade': llm2_q.get('grade'),
                            'llm2_max_points': llm2_q.get('max_points'),
                            'agreement': qdata.get('_initial_final', {}).get('agreement', True)
                        }
                    comparison_summary['copies'].append(copy_summary)

                await self._call_callback(progress_callback, 'batch_comparison_ready', comparison_summary)

            # ═══════════════════════════════════════════════════════════════════
            # POST-BATCH VERIFICATION (always run for dual LLM batch mode)
            # Follows same logic as individual mode: verification → ultimatum
            # ═══════════════════════════════════════════════════════════════════
            from ai.batch_grader import (
                detect_disagreements, run_dual_llm_verification,
                run_per_question_dual_verification, run_dual_llm_ultimatum,
                Disagreement
            )

            # Detect grade disagreements BEFORE modifying any grades
            disagreements = detect_disagreements(
                llm1_result, llm2_result,
                llm1_name, llm2_name,
                copies_data
                # threshold uses default 10% of max_points
            )

            # Detect student name disagreements (before verification)
            # In grouped mode, these will be verified together with grade disagreements
            name_disagreements = []
            if self._comparison_mode and llm1_result and llm2_result:
                for copy_result in batch_result.copies:
                    llm1_student_name = None
                    llm2_student_name = None

                    for c in llm1_result.copies:
                        if c.copy_index == copy_result.copy_index:
                            llm1_student_name = c.student_name
                            break
                    for c in llm2_result.copies:
                        if c.copy_index == copy_result.copy_index:
                            llm2_student_name = c.student_name
                            break

                    # Check for disagreement (both detected names but different)
                    if llm1_student_name and llm2_student_name:
                        n1_normalized = llm1_student_name.lower().strip()
                        n2_normalized = llm2_student_name.lower().strip()
                        if n1_normalized != n2_normalized:
                            name_disagreements.append({
                                'copy_index': copy_result.copy_index,
                                'llm1_name': llm1_student_name,
                                'llm2_name': llm2_student_name
                            })

            if disagreements or (self._batch_verify == "grouped" and name_disagreements):
                logger.info(f"Detected {len(disagreements)} disagreements, running dual LLM verification ({self._batch_verify})")

                if progress_callback:
                    await self._call_callback(progress_callback, 'batch_verification_start', {
                        'disagreements_count': len(disagreements),
                        'mode': self._batch_verify
                    })

                # ===== PHASE 1: Dual LLM Verification =====
                # Both LLMs see each other's work and can adjust their grades
                if self._batch_verify == "per-question":
                    # One call per disagreement per LLM (more granular)
                    verification_results = await run_per_question_dual_verification(
                        self.ai.providers, disagreements, language
                    )
                else:
                    # Grouped: one call per LLM (all disagreements in one prompt)
                    # In grouped mode, include name_disagreements in the same prompt

                    # Collect images from copies with name disagreements (for name-only cases)
                    extra_images = []
                    if name_disagreements and not disagreements:
                        for nd in name_disagreements:
                            for copy_result in batch_result.copies:
                                if copy_result.copy_index == nd['copy_index']:
                                    if hasattr(copy_result, 'image_paths'):
                                        extra_images.extend(copy_result.image_paths)
                                    break

                    verification_results = await run_dual_llm_verification(
                        self.ai.providers, disagreements, language,
                        name_disagreements=name_disagreements if name_disagreements else None,
                        extra_images=extra_images if extra_images else None
                    )

                # Apply verification results to batch_result
                for copy_result in batch_result.copies:
                    for qid, qdata in copy_result.questions.items():
                        key = f"copy_{copy_result.copy_index}_{qid}"
                        if key in verification_results:
                            verified = verification_results[key]
                            qdata['grade'] = verified['final_grade']
                            qdata['reasoning'] = verified.get('llm1_reasoning', qdata.get('reasoning', ''))
                            # Update feedback with consolidated feedback from verification
                            if verified.get('final_feedback'):
                                qdata['feedback'] = verified['final_feedback']

                            # Update comparison data with verification result
                            comp_key = f"copy_{copy_result.copy_index}"
                            if comp_key in llm_comparison_data.get("llm_comparison", {}):
                                if qid in llm_comparison_data["llm_comparison"][comp_key].get("questions", {}):
                                    # Get LLM data from comparison structure
                                    question_comp_data = llm_comparison_data["llm_comparison"][comp_key]["questions"][qid]
                                    llm1_data = question_comp_data.get(llm1_name, {})
                                    llm2_data = question_comp_data.get(llm2_name, {})

                                    # Get resolved max_points from verification
                                    resolved_max_points = verified.get('resolved_max_points', qdata.get('max_points', 1.0))

                                    # Check initial max_points disagreement BEFORE updating qdata
                                    # Compare llm1's original max_points with llm2's original max_points
                                    initial_mp_disagreement = abs(llm1_data.get('max_points', 1.0) - llm2_data.get('max_points', 1.0)) > 0.01 if llm2_data else False

                                    # Update qdata max_points with resolved value
                                    qdata['max_points'] = resolved_max_points

                                    # Store verification data
                                    verification_data = {
                                        "llm1_new_grade": verified.get('llm1_new_grade'),
                                        "llm2_new_grade": verified.get('llm2_new_grade'),
                                        "llm1_reasoning": verified.get('llm1_reasoning', ''),
                                        "llm2_reasoning": verified.get('llm2_reasoning', ''),
                                        "llm1_feedback": verified.get('llm1_feedback', ''),
                                        "llm2_feedback": verified.get('llm2_feedback', ''),
                                        "final_feedback": verified.get('final_feedback', ''),
                                        "confidence": verified.get('confidence', 0.8),
                                        "method": verified.get('method', 'grouped')
                                    }

                                    # Include max_points if there was an initial disagreement OR if values changed
                                    llm1_mp = verified.get('llm1_new_max_points')
                                    llm2_mp = verified.get('llm2_new_max_points')
                                    mp_values_changed = (llm1_mp is not None and abs(llm1_mp - llm1_data.get('max_points', 1.0)) > 0.01) or \
                                                       (llm2_mp is not None and abs(llm2_mp - llm2_data.get('max_points', 1.0)) > 0.01)

                                    if initial_mp_disagreement or mp_values_changed or (llm1_mp is not None and llm2_mp is not None and abs(llm1_mp - llm2_mp) > 0.01):
                                        verification_data["llm1_new_max_points"] = llm1_mp if llm1_mp is not None else llm1_data.get('max_points', 1.0)
                                        verification_data["llm2_new_max_points"] = llm2_mp if llm2_mp is not None else llm2_data.get('max_points', 1.0)
                                        verification_data["resolved_max_points"] = resolved_max_points

                                    # Only include readings if they were corrected
                                    llm1_reading = verified.get('llm1_new_reading')
                                    llm2_reading = verified.get('llm2_new_reading')
                                    if llm1_reading or llm2_reading:
                                        verification_data["llm1_new_reading"] = llm1_reading
                                        verification_data["llm2_new_reading"] = llm2_reading

                                        # Calculate and store reading similarity score
                                        if llm1_reading and llm2_reading:
                                            from difflib import SequenceMatcher
                                            reading_similarity = SequenceMatcher(
                                                None,
                                                llm1_reading.lower().strip(),
                                                llm2_reading.lower().strip()
                                            ).ratio()
                                            verification_data["reading_similarity"] = round(reading_similarity, 2)
                                            verification_data["reading_disagreement"] = reading_similarity < 0.8

                                    llm_comparison_data["llm_comparison"][comp_key]["questions"][qid]["verification"] = verification_data

                                    # Update the pending final data
                                    llm_comparison_data["llm_comparison"][comp_key]["questions"][qid]["_pending_final"] = {
                                        "grade": verified['final_grade'],
                                        "max_points": resolved_max_points,
                                        "method": f"verified_{verified.get('method', 'grouped')}"
                                    }

                # ===== NAME VERIFICATION RESULTS (grouped mode) =====
                # Process name verification results from combined verification
                name_verification_results = {}
                persistent_name_disagreements = []
                if self._batch_verify == "grouped" and name_disagreements:
                    for nd in name_disagreements:
                        copy_idx = nd['copy_index']
                        name_key = f"name_{copy_idx}"

                        if name_key in verification_results:
                            name_result = verification_results[name_key]
                            name_verification_results[copy_idx] = name_result

                            # Check if agreement was reached
                            if name_result.get('agreement'):
                                # Update student name in batch_result
                                for copy_result in batch_result.copies:
                                    if copy_result.copy_index == copy_idx:
                                        copy_result.student_name = name_result['resolved_name']
                                        break

                                # Store in llm_comparison_data
                                comp_key = f"copy_{copy_idx}"
                                if comp_key not in llm_comparison_data.get("llm_comparison", {}):
                                    llm_comparison_data["llm_comparison"][comp_key] = {}
                                llm_comparison_data["llm_comparison"][comp_key]["student_name_resolution"] = {
                                    "verification": {
                                        "llm1_new_name": name_result.get('llm1_new_name'),
                                        "llm2_new_name": name_result.get('llm2_new_name'),
                                        "agreement": True
                                    },
                                    "final_resolved_name": name_result['resolved_name']
                                }
                            else:
                                # Persistent disagreement - will need ultimatum
                                persistent_name_disagreements.append({
                                    'copy_index': copy_idx,
                                    'llm1_name': name_result.get('llm1_new_name', nd['llm1_name']),
                                    'llm2_name': name_result.get('llm2_new_name', nd['llm2_name'])
                                })

                # ===== PHASE 2: Ultimatum Round (if disagreements persist) =====
                # Check for persistent disagreements after verification

                persistent_disagreements = []
                for d in disagreements:
                    key = f"copy_{d.copy_index}_{d.question_id}"

                    if key in verification_results:
                        v = verification_results[key]
                        llm1_new = v.get('llm1_new_grade', d.llm1_grade)
                        llm2_new = v.get('llm2_new_grade', d.llm2_grade)
                        max_pts = v.get('max_points', d.max_points)
                        # Use relative threshold (10% of max_points)
                        relative_threshold = max_pts * get_settings().grade_agreement_threshold
                        gap = abs(llm1_new - llm2_new)

                        # Detect flip-flop (LLMs swapped positions on grade)
                        llm1_initial = d.llm1_grade
                        llm2_initial = d.llm2_grade
                        grade_flip_flop = (llm1_new != llm1_initial and llm2_new != llm2_initial and
                                          abs(llm1_new - llm2_initial) < 0.01 and abs(llm2_new - llm1_initial) < 0.01)

                        # Check max_points agreement and flip-flop
                        llm1_new_mp = v.get('llm1_new_max_points', d.llm1_max_points)
                        llm2_new_mp = v.get('llm2_new_max_points', d.llm2_max_points)
                        llm1_initial_mp = d.llm1_max_points
                        llm2_initial_mp = d.llm2_max_points

                        # max_points disagreement: different values after verification
                        max_points_disagreement = (llm1_new_mp is not None and llm2_new_mp is not None and
                                                  abs(llm1_new_mp - llm2_new_mp) > 0.01)

                        # max_points flip-flop: LLMs swapped their max_points values
                        max_points_flip_flop = (llm1_new_mp != llm1_initial_mp and llm2_new_mp != llm2_initial_mp and
                                               abs(llm1_new_mp - llm2_initial_mp) < 0.01 and
                                               abs(llm2_new_mp - llm1_initial_mp) < 0.01)

                        # Check if original disagreement included reading disagreement
                        original_has_reading = hasattr(d, 'disagreement_type') and 'reading' in getattr(d, 'disagreement_type', '')

                        # Check if readings still differ after verification
                        llm1_new_reading = v.get('llm1_new_reading', d.llm1_reading if hasattr(d, 'llm1_reading') else '')
                        llm2_new_reading = v.get('llm2_new_reading', d.llm2_reading if hasattr(d, 'llm2_reading') else '')
                        reading_still_differs = False
                        if llm1_new_reading and llm2_new_reading:
                            from difflib import SequenceMatcher
                            reading_similarity = SequenceMatcher(None, llm1_new_reading.lower(), llm2_new_reading.lower()).ratio()
                            reading_still_differs = reading_similarity < 0.8

                        # Persistent disagreement if: grade gap persists OR max_points disagree OR any flip-flop OR reading disagreement
                        is_persistent = (gap >= relative_threshold or
                                        max_points_disagreement or
                                        grade_flip_flop or
                                        max_points_flip_flop or
                                        (original_has_reading and reading_still_differs))

                        if is_persistent:
                            persistent_disagreements.append({
                                'copy_index': d.copy_index,
                                'question_id': d.question_id,
                                'llm1_grade': llm1_new,
                                'llm2_grade': llm2_new,
                                'llm1_initial_grade': llm1_initial,
                                'llm2_initial_grade': llm2_initial,
                                'llm1_max_points': llm1_new_mp if llm1_new_mp is not None else d.llm1_max_points,
                                'llm2_max_points': llm2_new_mp if llm2_new_mp is not None else d.llm2_max_points,
                                'llm1_initial_max_points': llm1_initial_mp,
                                'llm2_initial_max_points': llm2_initial_mp,
                                'max_points': max_pts,
                                'llm1_reasoning': v.get('llm1_reasoning', d.llm1_reasoning),
                                'llm2_reasoning': v.get('llm2_reasoning', d.llm2_reasoning),
                                'image_paths': d.image_paths,
                                'flip_flop_detected': grade_flip_flop or max_points_flip_flop,
                                'max_points_flip_flop': max_points_flip_flop,
                                'max_points_disagreement': max_points_disagreement,
                                'reading_disagreement': (original_has_reading and reading_still_differs)
                            })

                # Build verification summary for CLI display (AFTER determining persistent_disagreements)
                # Create set of keys going to ultimatum for quick lookup
                ultimatum_keys = set()
                ultimatum_reasons = {}
                for pd in persistent_disagreements:
                    key = f"copy_{pd['copy_index']}_{pd['question_id']}"
                    ultimatum_keys.add(key)
                    # Determine reason
                    reasons = []
                    if pd.get('max_points_disagreement'):
                        reasons.append('barème')
                    if pd.get('flip_flop_detected'):
                        reasons.append('flip-flop')
                    if pd.get('reading_disagreement'):
                        reasons.append('lecture')
                    if pd.get('llm1_grade') is not None and pd.get('llm2_grade') is not None:
                        gap = abs(pd['llm1_grade'] - pd['llm2_grade'])
                        if gap >= pd.get('max_points', 1) * get_settings().grade_agreement_threshold:
                            reasons.append('notes')
                    ultimatum_reasons[key] = reasons if reasons else ['autre']

                verification_summary = {
                    'resolved_count': len([k for k in verification_results if k.startswith('copy_')]),
                    'questions': []
                }
                for key, verified in verification_results.items():
                    if key.startswith('copy_') and not key.startswith('name_'):
                        parts = key.split('_')
                        if len(parts) >= 3:
                            copy_idx = int(parts[1])
                            qid = '_'.join(parts[2:])
                            goes_to_ultimatum = key in ultimatum_keys
                            verification_summary['questions'].append({
                                'copy_index': copy_idx,
                                'question_id': qid,
                                'final_grade': verified.get('final_grade'),
                                'method': verified.get('method'),
                                'goes_to_ultimatum': goes_to_ultimatum,
                                'ultimatum_reasons': ultimatum_reasons.get(key, [])
                            })

                if progress_callback:
                    await self._call_callback(progress_callback, 'batch_verification_done', verification_summary)

                if persistent_disagreements:
                    logger.info(f"Persisting {len(persistent_disagreements)} disagreements after verification, running ultimatum round")

                    if progress_callback:
                        await self._call_callback(progress_callback, 'batch_ultimatum_start', {
                            'persistent_count': len(persistent_disagreements)
                        })

                    # Run ultimatum round with both LLMs
                    ultimatum_results = await run_dual_llm_ultimatum(
                        self.ai.providers, persistent_disagreements, language
                    )

                    # Check for parsing failures
                    parse_status = ultimatum_results.pop('_parse_status', {})
                    llm1_parse_failed = parse_status.get('llm1_parse_failed', False)
                    llm2_parse_failed = parse_status.get('llm2_parse_failed', False)

                    if llm1_parse_failed or llm2_parse_failed:
                        parse_warning = []
                        if llm1_parse_failed:
                            parse_warning.append(llm1_name)
                        if llm2_parse_failed:
                            parse_warning.append(llm2_name)
                        warning_msg = f"Ultimatum parsing failed for: {', '.join(parse_warning)}"
                        logger.warning(warning_msg)

                        if progress_callback:
                            await self._call_callback(progress_callback, 'ultimatum_parse_warning', {
                                'warning': warning_msg,
                                'llm1_failed': llm1_parse_failed,
                                'llm2_failed': llm2_parse_failed
                            })

                        # Add warning to comparison data
                        llm_comparison_data['ultimatum_parse_warning'] = {
                            'llm1_parse_failed': llm1_parse_failed,
                            'llm2_parse_failed': llm2_parse_failed,
                            'warning': warning_msg
                        }

                    # Apply ultimatum results
                    for copy_result in batch_result.copies:
                        for qid, qdata in copy_result.questions.items():
                            key = f"copy_{copy_result.copy_index}_{qid}"
                            if key in ultimatum_results:
                                ultimate = ultimatum_results[key]
                                qdata['grade'] = ultimate['final_grade']
                                qdata['reasoning'] = ultimate.get('llm1_reasoning', qdata.get('reasoning', ''))
                                # Update feedback with consolidated feedback from ultimatum
                                if ultimate.get('final_feedback'):
                                    qdata['feedback'] = ultimate['final_feedback']

                                # Update comparison data with ultimatum result
                                comp_key = f"copy_{copy_result.copy_index}"
                                if comp_key in llm_comparison_data.get("llm_comparison", {}):
                                    if qid in llm_comparison_data["llm_comparison"][comp_key].get("questions", {}):
                                        # Get max_points from the question data
                                        q_max_points = qdata.get('max_points', 1.0)

                                        # Find persistent disagreement data for flip-flop
                                        grade_flip_flop = False
                                        max_points_flip_flop = False
                                        max_points_disagreement = False
                                        for pd in persistent_disagreements:
                                            if pd.get('copy_index') == copy_result.copy_index and pd.get('question_id') == qid:
                                                grade_flip_flop = pd.get('flip_flop_detected', False)
                                                max_points_flip_flop = pd.get('max_points_flip_flop', False)
                                                max_points_disagreement = pd.get('max_points_disagreement', False)
                                                break

                                        # Build ultimatum data
                                        ultimatum_data = {
                                            "llm1_final_grade": ultimate.get('llm1_final_grade'),
                                            "llm2_final_grade": ultimate.get('llm2_final_grade'),
                                            "llm1_decision": ultimate.get('llm1_decision'),
                                            "llm2_decision": ultimate.get('llm2_decision'),
                                            "llm1_reasoning": ultimate.get('llm1_reasoning', ''),
                                            "llm2_reasoning": ultimate.get('llm2_reasoning', ''),
                                            "llm1_feedback": ultimate.get('llm1_feedback', ''),
                                            "llm2_feedback": ultimate.get('llm2_feedback', ''),
                                            "final_feedback": ultimate.get('final_feedback', ''),
                                            "llm1_parse_success": ultimate.get('llm1_parse_success', True),
                                            "llm2_parse_success": ultimate.get('llm2_parse_success', True),
                                            "grade_flip_flop": grade_flip_flop,
                                            "max_points_flip_flop": max_points_flip_flop,
                                            "max_points_disagreement": max_points_disagreement,
                                            "confidence": ultimate.get('confidence', 0.8),
                                            "method": ultimate.get('method', 'ultimatum_average')
                                        }

                                        # Only include max_points if there was a disagreement
                                        if ultimate.get('max_points_disagreement'):
                                            ultimatum_data["llm1_final_max_points"] = ultimate.get('llm1_final_max_points')
                                            ultimatum_data["llm2_final_max_points"] = ultimate.get('llm2_final_max_points')
                                            ultimatum_data["resolved_max_points"] = ultimate.get('resolved_max_points')

                                        llm_comparison_data["llm_comparison"][comp_key]["questions"][qid]["ultimatum"] = ultimatum_data

                                        # Get resolved max_points from ultimatum
                                        ultimatum_resolved_mp = ultimate.get('resolved_max_points', q_max_points)

                                        # Update pending final data
                                        llm_comparison_data["llm_comparison"][comp_key]["questions"][qid]["_pending_final"] = {
                                            "grade": ultimate['final_grade'],
                                            "max_points": ultimatum_resolved_mp,
                                            "method": ultimate.get('method', 'ultimatum_average')
                                        }

                                        # Also update qdata max_points with resolved value
                                        qdata['max_points'] = ultimatum_resolved_mp

                    # Build ultimatum summary for CLI display
                    ultimatum_summary = {
                        'resolved_count': len(ultimatum_results),
                        'questions': []
                    }
                    for key, ultimate in ultimatum_results.items():
                        if key.startswith('copy_'):
                            parts = key.split('_')
                            if len(parts) >= 3:
                                copy_idx = int(parts[1])
                                qid = '_'.join(parts[2:])
                                ultimatum_summary['questions'].append({
                                    'copy_index': copy_idx,
                                    'question_id': qid,
                                    'llm1_final': ultimate.get('llm1_final_grade'),
                                    'llm2_final': ultimate.get('llm2_final_grade'),
                                    'final_grade': ultimate.get('final_grade'),
                                    'method': ultimate.get('method')
                                })

                    if progress_callback:
                        await self._call_callback(progress_callback, 'batch_ultimatum_done', ultimatum_summary)

                    # ===== PHASE 3: Ask user for max_points disagreements after ultimatum =====
                    max_points_disagreements = []
                    for copy_result in batch_result.copies:
                        for qid, qdata in copy_result.questions.items():
                            key = f"copy_{copy_result.copy_index}_{qid}"
                            if key in ultimatum_results:
                                ultimate = ultimatum_results[key]
                                if ultimate.get('max_points_disagreement'):
                                    max_points_disagreements.append({
                                        'copy_index': copy_result.copy_index,
                                        'question_id': qid,
                                        'llm1_max_points': ultimate.get('llm1_final_max_points'),
                                        'llm2_max_points': ultimate.get('llm2_final_max_points'),
                                        'llm1_grade': ultimate.get('llm1_final_grade'),
                                        'llm2_grade': ultimate.get('llm2_final_grade')
                                    })

                    if max_points_disagreements:
                        logger.warning(f"{len(max_points_disagreements)} max_points disagreements after ultimatum, asking user")

                        if progress_callback:
                            await self._call_callback(progress_callback, 'max_points_disagreement', {
                                'disagreements': max_points_disagreements
                            })

                        # Ask user to resolve each max_points disagreement
                        for mpd in max_points_disagreements:
                            comp_key = f"copy_{mpd['copy_index']}"
                            qid = mpd['question_id']

                            # Use the interactive method to ask user
                            resolved_max_points = await self._ask_max_points_resolution(mpd, llm1_name, llm2_name)

                            # Update the max_points in the question data
                            if comp_key in llm_comparison_data.get("llm_comparison", {}):
                                if qid in llm_comparison_data["llm_comparison"][comp_key].get("questions", {}):
                                    q_data = llm_comparison_data["llm_comparison"][comp_key]["questions"][qid]

                                    # Update ultimatum section with user resolution
                                    if "ultimatum" in q_data:
                                        q_data["ultimatum"]["user_resolved_max_points"] = resolved_max_points

                                    # Update the copy_result max_points (used for max_points_per_question summary)
                                    for copy_result in batch_result.copies:
                                        if copy_result.copy_index == mpd['copy_index'] and qid in copy_result.questions:
                                            copy_result.questions[qid]['max_points'] = resolved_max_points

                # ===== NAME ULTIMATUM (grouped mode) =====
                # If name disagreements persist after combined verification in grouped mode
                if self._batch_verify == "grouped" and persistent_name_disagreements:
                    from ai.batch_grader import run_student_name_ultimatum

                    logger.info(f"{len(persistent_name_disagreements)} name disagreements persist after verification, running ultimatum")

                    if progress_callback:
                        await self._call_callback(progress_callback, 'name_ultimatum_start', {
                            'persistent_count': len(persistent_name_disagreements)
                        })

                    # Collect all image paths for copies with name disagreements
                    name_disagreement_images = []
                    for copy_result in batch_result.copies:
                        for d in persistent_name_disagreements:
                            if d['copy_index'] == copy_result.copy_index:
                                if hasattr(copy_result, 'image_paths'):
                                    name_disagreement_images.extend(copy_result.image_paths)
                                break

                    name_ultimatum_results = await run_student_name_ultimatum(
                        self.ai.providers, persistent_name_disagreements, name_disagreement_images, language
                    )

                    if progress_callback:
                        await self._call_callback(progress_callback, 'name_ultimatum_done', {
                            'resolved_count': len(name_ultimatum_results)
                        })

                    # Apply name ultimatum results
                    for copy_idx, result in name_ultimatum_results.items():
                        resolved_name = result.get('resolved_name')

                        # Update batch_result
                        for copy_result in batch_result.copies:
                            if copy_result.copy_index == copy_idx:
                                copy_result.student_name = resolved_name
                                break

                        # Store in llm_comparison_data for audit
                        comp_key = f"copy_{copy_idx}"
                        if comp_key in llm_comparison_data.get("llm_comparison", {}):
                            if "student_name_resolution" not in llm_comparison_data["llm_comparison"][comp_key]:
                                llm_comparison_data["llm_comparison"][comp_key]["student_name_resolution"] = {}
                            llm_comparison_data["llm_comparison"][comp_key]["student_name_resolution"]["ultimatum"] = result
                            llm_comparison_data["llm_comparison"][comp_key]["student_name_resolution"]["final_resolved_name"] = resolved_name

                    # Ask user for any still-unresolved names (or auto-resolve in auto mode)
                    is_auto_mode = self.workflow_state.auto_mode if self.workflow_state else False

                    for copy_idx, result in name_ultimatum_results.items():
                        if result.get('needs_user_resolution'):
                            if is_auto_mode:
                                # Auto mode: pick the name with higher confidence, or LLM1's name as fallback
                                llm1_conf = result.get('llm1_confidence', 0.5)
                                llm2_conf = result.get('llm2_confidence', 0.5)
                                llm1_final = result.get('llm1_final_name', '')
                                llm2_final = result.get('llm2_final_name', '')

                                if llm1_conf >= llm2_conf:
                                    resolved_name = llm1_final
                                else:
                                    resolved_name = llm2_final

                                logger.info(f"Auto-resolved name for copy {copy_idx}: {resolved_name} (auto-confirm mode)")
                            else:
                                # Interactive mode: ask user
                                resolved_name = await self._ask_student_name_resolution(result, llm1_name, llm2_name)

                            # Update with resolution
                            for copy_result in batch_result.copies:
                                if copy_result.copy_index == copy_idx:
                                    copy_result.student_name = resolved_name
                                    break

                            comp_key = f"copy_{copy_idx}"
                            if comp_key in llm_comparison_data.get("llm_comparison", {}):
                                llm_comparison_data["llm_comparison"][comp_key]["student_name_resolution"]["user_resolved_name"] = resolved_name
                                llm_comparison_data["llm_comparison"][comp_key]["student_name_resolution"]["final_resolved_name"] = resolved_name

            # ===== STUDENT NAME VERIFICATION & ULTIMATUM (per-question mode only) =====
            # In grouped mode, name verification was already done in the combined verification prompt
            # In per-question mode, run separate name verification calls
            if self._batch_verify == "per-question" and name_disagreements:
                from ai.batch_grader import run_student_name_verification, run_student_name_ultimatum

                logger.info(f"{len(name_disagreements)} student name disagreements detected, running verification")

                if progress_callback:
                    await self._call_callback(progress_callback, 'name_verification_start', {
                        'disagreements_count': len(name_disagreements)
                    })

                # Collect all image paths for the copies with name disagreements
                name_disagreement_images = []
                for copy_result in batch_result.copies:
                    for d in name_disagreements:
                        if d['copy_index'] == copy_result.copy_index:
                            # Get images for this copy
                            if hasattr(copy_result, 'image_paths'):
                                name_disagreement_images.extend(copy_result.image_paths)
                            break

                # Run verification
                name_verification_results = await run_student_name_verification(
                    self.ai.providers, name_disagreements, name_disagreement_images, language
                )

                # Check for persistent disagreements
                persistent_name_disagreements = []
                for copy_idx, result in name_verification_results.items():
                    if not result.get('agreement'):
                        # Find original disagreement
                        orig = next((d for d in name_disagreements if d['copy_index'] == copy_idx), None)
                        if orig:
                            persistent_name_disagreements.append({
                                'copy_index': copy_idx,
                                'llm1_name': result['llm1_new_name'],
                                'llm2_name': result['llm2_new_name']
                            })

                # Run ultimatum if persistent disagreements
                name_ultimatum_results = {}
                if persistent_name_disagreements:
                    logger.info(f"{len(persistent_name_disagreements)} name disagreements persist, running ultimatum")

                    if progress_callback:
                        await self._call_callback(progress_callback, 'name_ultimatum_start', {
                            'persistent_count': len(persistent_name_disagreements)
                        })

                    name_ultimatum_results = await run_student_name_ultimatum(
                        self.ai.providers, persistent_name_disagreements, name_disagreement_images, language
                    )

                    if progress_callback:
                        await self._call_callback(progress_callback, 'name_ultimatum_done', {
                            'resolved_count': len(name_ultimatum_results)
                        })

                # Apply resolved names to batch_result and llm_comparison_data
                all_name_results = {**name_verification_results, **name_ultimatum_results}
                for copy_idx, result in all_name_results.items():
                    resolved_name = result.get('resolved_name')

                    # Update batch_result
                    for copy_result in batch_result.copies:
                        if copy_result.copy_index == copy_idx:
                            copy_result.student_name = resolved_name
                            break

                    # Store in llm_comparison_data for audit
                    comp_key = f"copy_{copy_idx}"
                    if comp_key not in llm_comparison_data.get("llm_comparison", {}):
                        llm_comparison_data["llm_comparison"][comp_key] = {}

                    llm_comparison_data["llm_comparison"][comp_key]["student_name_resolution"] = {
                        "verification": {
                            "llm1_new_name": name_verification_results.get(copy_idx, {}).get('llm1_new_name'),
                            "llm2_new_name": name_verification_results.get(copy_idx, {}).get('llm2_new_name'),
                            "agreement": name_verification_results.get(copy_idx, {}).get('agreement')
                        },
                        "ultimatum": name_ultimatum_results.get(copy_idx) if copy_idx in name_ultimatum_results else None,
                        "final_resolved_name": resolved_name,
                        "needs_user_resolution": result.get('needs_user_resolution', False)
                    }

                # Ask user for any still-unresolved names (or auto-resolve in auto mode)
                is_auto_mode = self.workflow_state.auto_mode if self.workflow_state else False

                for copy_idx, result in all_name_results.items():
                    if result.get('needs_user_resolution'):
                        if is_auto_mode:
                            # Auto mode: pick the name with higher confidence, or LLM1's name as fallback
                            llm1_conf = result.get('llm1_confidence', 0.5)
                            llm2_conf = result.get('llm2_confidence', 0.5)
                            llm1_final = result.get('llm1_final_name', '')
                            llm2_final = result.get('llm2_final_name', '')

                            if llm1_conf >= llm2_conf:
                                resolved_name = llm1_final
                            else:
                                resolved_name = llm2_final

                            logger.info(f"Auto-resolved name for copy {copy_idx}: {resolved_name} (auto-confirm mode)")
                        else:
                            # Interactive mode: ask user
                            resolved_name = await self._ask_student_name_resolution(result, llm1_name, llm2_name)

                        # Update with resolution
                        for copy_result in batch_result.copies:
                            if copy_result.copy_index == copy_idx:
                                copy_result.student_name = resolved_name
                                break

                        comp_key = f"copy_{copy_idx}"
                        if comp_key in llm_comparison_data.get("llm_comparison", {}):
                            llm_comparison_data["llm_comparison"][comp_key]["student_name_resolution"]["user_resolved_name"] = resolved_name

                if progress_callback:
                    await self._call_callback(progress_callback, 'name_verification_done', {
                        'resolved_count': len(name_verification_results)
                    })

            # ===== FINALIZE: Rebuild question data in correct order =====
            # Order: LLM1, LLM2, verification, ultimatum, final (with max_points in final)
            for copy_idx in range(len(batch_result.copies)):
                comp_key = f"copy_{copy_idx + 1}"
                if comp_key in llm_comparison_data.get("llm_comparison", {}):
                    questions = llm_comparison_data["llm_comparison"][comp_key].get("questions", {})
                    for qid, qdata in questions.items():
                        # Get all components
                        llm1_data = qdata.get(llm1_name)
                        llm2_data = qdata.get(llm2_name)
                        verification_data = qdata.get("verification")
                        ultimatum_data = qdata.get("ultimatum")

                        # Determine final data (already contains max_points from _initial_final or _pending_final)
                        if "_pending_final" in qdata:
                            final_data = qdata["_pending_final"].copy()
                        elif "_initial_final" in qdata:
                            final_data = qdata["_initial_final"].copy()
                        else:
                            # Fallback: construct minimal final data
                            llm1_mp = llm1_data.get("max_points", 1.0) if llm1_data else 1.0
                            llm2_mp = llm2_data.get("max_points", 1.0) if llm2_data else 1.0
                            # Use resolved max_points from verification if available, otherwise consensus
                            resolved_mp = qdata.get("max_points")
                            if resolved_mp is None:
                                # Consensus: prefer agreed value or max as fallback
                                resolved_mp = llm1_mp if llm1_mp == llm2_mp else max(llm1_mp, llm2_mp)
                            final_data = {
                                "grade": 0,
                                "max_points": resolved_mp,
                                "method": "unknown"
                            }

                        # Ensure max_points is present (should be, but safety check)
                        if "max_points" not in final_data:
                            final_data["max_points"] = qdata.get("max_points", 1.0)

                        # Rebuild in correct order
                        rebuilt = {}
                        if llm1_data:
                            rebuilt[llm1_name] = llm1_data
                        if llm2_data:
                            rebuilt[llm2_name] = llm2_data
                        if verification_data:
                            rebuilt["verification"] = verification_data
                        if ultimatum_data:
                            rebuilt["ultimatum"] = ultimatum_data
                        rebuilt["final"] = final_data

                        # Replace the question data
                        llm_comparison_data["llm_comparison"][comp_key]["questions"][qid] = rebuilt

            else:
                # No disagreements - apply averaged grades now
                for copy_result in batch_result.copies:
                    for qid, qdata in copy_result.questions.items():
                        comp_key = f"copy_{copy_result.copy_index}"
                        if comp_key in llm_comparison_data.get("llm_comparison", {}):
                            if qid in llm_comparison_data["llm_comparison"][comp_key].get("questions", {}):
                                q_data = llm_comparison_data["llm_comparison"][comp_key]["questions"][qid]
                                # Get final from _initial_final
                                final_data = q_data.get("_initial_final", {})
                                qdata['grade'] = final_data.get('grade', qdata.get('grade', 0))

        else:
            # Single LLM batch
            from ai.batch_grader import BatchGrader
            grader = BatchGrader(self.ai)
            batch_result = await grader.grade_batch(copies_data, questions, language)

            if not batch_result.parse_success:
                logger.error(f"Batch grading failed: {batch_result.parse_errors}")
                return []

        # Track student detection for audit
        student_detection_info = {
            'mode': 'multi_student_detection' if len(batch_result.copies) > len(self.session.copies) else 'standard',
            'input_copies': len(self.session.copies),
            'detected_copies': len(batch_result.copies),
            'students': [],
            'llm_detections': {}  # Individual LLM detections
        }

        # Capture individual LLM student name detections for audit
        if self._comparison_mode and llm1_result and llm2_result:
            llm1_name, llm2_name = provider_names[0], provider_names[1]
            student_detection_info['llm_detections'] = {
                llm1_name: [
                    {'copy_index': c.copy_index, 'student_name': c.student_name}
                    for c in llm1_result.copies
                ],
                llm2_name: [
                    {'copy_index': c.copy_index, 'student_name': c.student_name}
                    for c in llm2_result.copies
                ]
            }

        # If LLM detected more students than we have copies, create new CopyDocument objects
        if len(batch_result.copies) > len(self.session.copies):
            logger.info(f"LLM detected {len(batch_result.copies)} students, creating additional copies")
            first_copy = self.session.copies[0] if self.session.copies else None
            for i in range(len(self.session.copies), len(batch_result.copies)):
                new_copy = CopyDocument(
                    pdf_path=first_copy.pdf_path if first_copy else None,
                    page_count=first_copy.page_count if first_copy else 0,
                    student_name=None,  # Will be set from batch result
                    language=first_copy.language if first_copy else 'fr'
                )
                self.session.copies.append(new_copy)
                logger.info(f"Created new copy {i+1} for detected student")
            total_copies = len(self.session.copies)

        # Add student detection info to llm_comparison_data for audit
        if llm_comparison_data:
            llm_comparison_data['student_detection'] = student_detection_info
        else:
            # Single LLM case - create comparison data with detection info
            llm_comparison_data = {
                "options": {
                    "mode": "batch",
                    "providers": [getattr(self.ai, 'model_name', 'unknown')],
                    "total_copies": total_copies
                },
                "student_detection": student_detection_info,
                "llm_comparison": {}
            }

        # Convert batch results to GradedCopy objects
        for copy_result in batch_result.copies:
            copy_idx = copy_result.copy_index - 1
            if copy_idx < 0 or copy_idx >= len(self.session.copies):
                logger.warning(f"Copy index {copy_result.copy_index} out of bounds, skipping")
                continue

            original_copy = self.session.copies[copy_idx]

            # Update student name if detected (do this FIRST for live display)
            detected_name = copy_result.student_name
            if detected_name and not original_copy.student_name:
                original_copy.student_name = detected_name

            # Track student info for this copy (add to llm_comparison for this copy)
            copy_key = f"copy_{copy_result.copy_index}"
            if copy_key not in llm_comparison_data.get("llm_comparison", {}):
                llm_comparison_data["llm_comparison"][copy_key] = {"questions": {}}

            # Get individual LLM student names for this copy
            llm1_student_name = None
            llm2_student_name = None
            if self._comparison_mode and llm1_result and llm2_result:
                for c in llm1_result.copies:
                    if c.copy_index == copy_result.copy_index:
                        llm1_student_name = c.student_name
                        break
                for c in llm2_result.copies:
                    if c.copy_index == copy_result.copy_index:
                        llm2_student_name = c.student_name
                        break

            # Determine final resolved name (after any verification/ultimatum if applicable)
            # For now, use LLM1's name as the resolved name (or detected_name from batch_result)
            final_resolved_name = detected_name

            llm_comparison_data["llm_comparison"][copy_key]["student_detection"] = {
                'copy_index': copy_result.copy_index,
                'student_name': detected_name,
                'llm1_student_name': llm1_student_name,
                'llm2_student_name': llm2_student_name,
                'final_resolved_name': final_resolved_name,
                'pages': getattr(copy_result, 'pages', None)
            }

            # Check for name disagreement - EXACT MATCH REQUIRED
            if llm1_student_name and llm2_student_name:
                n1_normalized = llm1_student_name.lower().strip()
                n2_normalized = llm2_student_name.lower().strip()
                if n1_normalized != n2_normalized:
                    llm_comparison_data["llm_comparison"][copy_key]["student_detection"]["name_disagreement"] = {
                        "llm1_name": llm1_student_name,
                        "llm2_name": llm2_student_name,
                        "resolved_name": final_resolved_name,
                        "resolution_method": "llm1_as_base"
                    }

            # Track student info for global summary
            student_detection_info['students'].append({
                'copy_index': copy_result.copy_index,
                'student_name': detected_name,
                'pages': getattr(copy_result, 'pages', None)
            })

            # Emit copy_start with detected name for live display update
            if progress_callback:
                await self._call_callback(progress_callback, 'copy_start', {
                    'copy_index': copy_result.copy_index,
                    'total_copies': total_copies,
                    'copy_id': original_copy.id,
                    'student_name': detected_name or original_copy.student_name,
                    'questions': list(copy_result.questions.keys())
                })

            # Build grades dict and extract detected scale
            grades = {}
            detected_scale = {}
            reasoning = {}
            student_feedback = {}
            readings = {}
            confidence_by_question = {}

            # Pre-fetch comparison data for this copy to check for feedback from other LLM
            copy_key = f"copy_{copy_result.copy_index}"
            copy_comparison_questions = None
            if llm_comparison_data and copy_key in llm_comparison_data.get("llm_comparison", {}):
                copy_comparison_questions = llm_comparison_data["llm_comparison"][copy_key].get("questions", {})

            for qid, qdata in copy_result.questions.items():
                grades[qid] = qdata.get('grade', 0)
                max_pts = float(qdata.get('max_points', 0))
                if max_pts > 0:
                    detected_scale[qid] = max_pts

                # Populate detailed fields for audit
                if qdata.get('reasoning'):
                    reasoning[qid] = qdata.get('reasoning')

                # Get feedback: prefer qdata, but check comparison data as fallback
                feedback = qdata.get('feedback')
                if not feedback and copy_comparison_questions and qid in copy_comparison_questions:
                    # Try to get feedback from either LLM in comparison data
                    q_comp = copy_comparison_questions[qid]
                    for provider_key in q_comp:
                        if provider_key not in ['max_points', '_initial_final', 'final', 'verification', 'ultimatum', '_pending_final']:
                            prov_feedback = q_comp[provider_key].get('feedback')
                            if prov_feedback:
                                feedback = prov_feedback
                                break
                if feedback:
                    student_feedback[qid] = feedback

                if qdata.get('student_answer_read'):
                    readings[qid] = qdata.get('student_answer_read')
                if qdata.get('confidence') is not None:
                    confidence_by_question[qid] = float(qdata.get('confidence', 0.8))

            # Update grading scale from detection
            self.update_scale_from_detection(detected_scale)

            # Calculate max_score from grading_scale (now updated)
            batch_max_score = self.get_total_max_points()
            # If still 0, we'll prompt user after grading in main.py

            # Build max_points_per_question summary from final resolved values
            max_points_per_question = {}
            for qid, qdata in copy_result.questions.items():
                max_pts = qdata.get('max_points', 0)
                if max_pts > 0:
                    max_points_per_question[qid] = max_pts

            # Get comparison data for this copy if available
            copy_comparison = None
            copy_key = f"copy_{copy_result.copy_index}"
            if llm_comparison_data and copy_key in llm_comparison_data.get("llm_comparison", {}):
                copy_data = llm_comparison_data["llm_comparison"][copy_key]

                # Build clean student_name section with same structure as questions
                # Structure: llm1_name, llm2_name, verification, ultimatum, final
                student_detection = copy_data.get("student_detection", {})
                student_name_resolution = copy_data.get("student_name_resolution", {})

                student_name_section = {}

                # Initial LLM names
                if "name_disagreement" in student_detection:
                    nd = student_detection["name_disagreement"]
                    student_name_section["llm1_name"] = nd.get("llm1_name")
                    student_name_section["llm2_name"] = nd.get("llm2_name")

                # Verification step (if occurred)
                if student_name_resolution.get("verification"):
                    v = student_name_resolution["verification"]
                    student_name_section["verification"] = {
                        "llm1_new_name": v.get("llm1_new_name"),
                        "llm2_new_name": v.get("llm2_new_name"),
                        "agreement": v.get("agreement"),
                        "method": v.get("method", "verification")
                    }

                # Ultimatum step (if occurred)
                if student_name_resolution.get("ultimatum"):
                    u = student_name_resolution["ultimatum"]
                    student_name_section["ultimatum"] = {
                        "llm1_final_name": u.get("llm1_final_name"),
                        "llm2_final_name": u.get("llm2_final_name"),
                        "resolved_name": u.get("resolved_name"),
                        "agreement": u.get("agreement"),
                        "llm1_confidence": u.get("llm1_confidence"),
                        "llm2_confidence": u.get("llm2_confidence")
                    }

                # User resolution (if occurred)
                if student_name_resolution.get("user_resolved_name"):
                    student_name_section["user_resolution"] = {
                        "resolved_name": student_name_resolution["user_resolved_name"]
                    }

                # Final result
                final_resolved = student_name_resolution.get("final_resolved_name") or student_detection.get("final_resolved_name")
                if final_resolved:
                    # Determine method
                    if student_name_resolution.get("user_resolved_name"):
                        method = "user_resolution"
                    elif student_name_resolution.get("ultimatum"):
                        method = "ultimatum"
                    elif student_name_resolution.get("verification", {}).get("agreement"):
                        method = "verification_consensus"
                    elif student_name_resolution.get("verification"):
                        method = "verification_average"
                    else:
                        method = "llm1_as_base"

                    student_name_section["final"] = {
                        "resolved_name": final_resolved,
                        "method": method
                    }

                # Copy index for reference
                if student_detection.get("copy_index"):
                    student_name_section["copy_index"] = student_detection["copy_index"]

                copy_comparison = {
                    "options": llm_comparison_data["options"],
                    "llm_comparison": copy_data.get("questions", {}),
                    "student_name": student_name_section
                }
            elif llm_comparison_data and "student_detection" in llm_comparison_data:
                # Single LLM case - still add student detection info
                copy_comparison = {
                    "options": llm_comparison_data.get("options", {}),
                    "student_detection": llm_comparison_data["student_detection"]
                }

            # Create GradedCopy with full audit data
            graded = GradedCopy(
                copy_id=original_copy.id,
                grades=grades,
                total_score=sum(grades.values()),
                max_score=batch_max_score,
                confidence=sum(q.get('confidence', 0.8) for q in copy_result.questions.values()) / max(len(copy_result.questions), 1),
                feedback=copy_result.overall_feedback or "",
                reasoning=reasoning,
                student_feedback=student_feedback,
                readings=readings,
                confidence_by_question=confidence_by_question,
                max_points_by_question=max_points_per_question,
                llm_comparison=copy_comparison
            )

            graded_copies.append(graded)
            self.session.graded_copies.append(graded)

            if progress_callback:
                # Build final questions summary for display
                final_questions = {}
                for qid in sorted(grades.keys(), key=lambda x: (int(x.replace('Q', '')) if x.replace('Q', '').isdigit() else 999)):
                    final_questions[qid] = {
                        'grade': grades[qid],
                        'max_points': max_points_per_question.get(qid, 1.0),
                        'confidence': confidence_by_question.get(qid, 0.8)
                    }

                await self._call_callback(progress_callback, 'copy_done', {
                    'copy_index': copy_result.copy_index,
                    'total_copies': total_copies,
                    'copy_id': original_copy.id,
                    'student_name': copy_result.student_name,
                    'total_score': graded.total_score,
                    'max_score': graded.max_score,
                    'confidence': graded.confidence,
                    'final_questions': final_questions
                })

        # Save and notify completion
        self._save_sync()

        if progress_callback:
            await self._call_callback(progress_callback, 'batch_done', {
                'total_copies': total_copies,
                'graded_copies': len(graded_copies),
                'patterns': batch_result.patterns
            })

        self._grading_complete = True
        return graded_copies

    async def _grade_all_hybrid(
        self,
        progress_callback: callable = None
    ) -> List[GradedCopy]:
        """
        Grade all copies using hybrid mode: LLM1=batch, LLM2=individual.

        This combines the benefits of both approaches:
        - Batch (LLM1): Consistency, pattern detection, sees all copies
        - Individual (LLM2): Independent verification, no contamination

        Then compares and resolves disagreements.
        """
        from ai.batch_grader import BatchGrader, BatchResult
        from ai.single_pass_grader import SinglePassGrader
        import logging

        logger = logging.getLogger(__name__)
        total_copies = len(self.session.copies)

        if not hasattr(self.ai, 'providers') or len(self.ai.providers) < 2:
            raise RuntimeError("Hybrid mode requires dual LLM with 2 providers")

        provider_names = [name for name, _ in self.ai.providers]
        llm1_name, llm1_provider = self.ai.providers[0]
        llm2_name, llm2_provider = self.ai.providers[1]

        # Notify start
        if progress_callback:
            await self._call_callback(progress_callback, 'hybrid_start', {
                'total_copies': total_copies,
                'llm1': llm1_name,
                'llm2': llm2_name,
                'llm1_mode': 'batch',
                'llm2_mode': 'individual'
            })

        # Prepare copies data for batch grading
        copies_data = []
        for i, copy in enumerate(self.session.copies):
            image_paths = []
            if copy.page_images:
                image_paths = copy.page_images
            elif copy.pdf_path:
                from vision.pdf_reader import PDFReader
                reader = PDFReader(copy.pdf_path)
                temp_dir = Path(self.store.session_dir) / "temp"
                temp_dir.mkdir(exist_ok=True)

                for page_num in range(copy.page_count):
                    image_path = str(temp_dir / f"hybrid_copy_{i+1}_page_{page_num}.png")
                    image_bytes = reader.get_page_image_bytes(page_num)
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                    image_paths.append(image_path)
                reader.close()

            copies_data.append({
                'copy_index': i + 1,
                'image_paths': image_paths,
                'student_name': copy.student_name
            })

        # Build questions dict
        questions = {}
        for qid, max_pts in self.grading_scale.items():
            questions[qid] = {
                'text': '',
                'criteria': '',
                'max_points': max_pts
            }

        language = 'fr'

        # ===== PARALLEL: LLM1 (batch) + LLM2 (individual) =====
        async def run_llm1_batch():
            """LLM1 grades all copies in one batch call."""
            grader = BatchGrader(llm1_provider)
            return await grader.grade_batch(copies_data, questions, language)

        async def run_llm2_individual():
            """LLM2 grades each copy individually in parallel."""
            results = []
            semaphore = asyncio.Semaphore(self._parallel)

            async def grade_one_copy(i: int, copy):
                async with semaphore:
                    grader = SinglePassGrader(llm2_provider)
                    copy_data = copies_data[i]

                    q_list = [{'id': qid, 'text': '', 'criteria': '', 'max_points': qdata['max_points']}
                              for qid, qdata in questions.items()]

                    result = await grader.grade_all_questions(
                        q_list,
                        copy_data['image_paths'],
                        language
                    )
                    return (i, result)

            tasks = [grade_one_copy(i, copy) for i, copy in enumerate(self.session.copies)]
            individual_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in individual_results:
                if not isinstance(result, Exception):
                    results.append(result)
            return results

        # Run both in parallel
        if progress_callback:
            await self._call_callback(progress_callback, 'hybrid_grading', {
                'status': 'running_both'
            })

        batch_result, individual_results = await asyncio.gather(
            run_llm1_batch(),
            run_llm2_individual(),
            return_exceptions=True
        )

        # Handle errors
        if isinstance(batch_result, Exception):
            logger.error(f"LLM1 batch failed: {batch_result}")
            batch_result = None
        if isinstance(individual_results, Exception):
            logger.error(f"LLM2 individual failed: {individual_results}")
            individual_results = []

        # ===== MERGE AND COMPARE =====
        graded_copies = []

        for i, original_copy in enumerate(self.session.copies):
            # Get LLM1 (batch) result for this copy
            llm1_grades = {}
            llm1_name_detected = None
            if batch_result and batch_result.parse_success:
                for copy_result in batch_result.copies:
                    if copy_result.copy_index == i + 1:
                        llm1_grades = {qid: qdata['grade'] for qid, qdata in copy_result.questions.items()}
                        llm1_name_detected = copy_result.student_name
                        break

            # Get LLM2 (individual) result for this copy
            llm2_grades = {}
            llm2_name_detected = None
            for idx, result in individual_results:
                if idx == i and result and result.parse_success:
                    llm2_grades = {qid: qdata.get('grade', 0) for qid, qdata in result.questions.items()}
                    llm2_name_detected = result.student_name
                    break

            # Compare and merge grades
            final_grades = {}
            disagreements = []
            llm_comparison_data = {}

            for qid in self.grading_scale.keys():
                g1 = llm1_grades.get(qid, 0)
                g2 = llm2_grades.get(qid, 0)
                max_pts = self.grading_scale.get(qid, 1)

                # Use relative threshold from settings
                relative_threshold = max_pts * get_settings().grade_agreement_threshold

                # Check for disagreement
                if abs(g1 - g2) >= relative_threshold:
                    disagreements.append({
                        'question_id': qid,
                        'llm1_grade': g1,
                        'llm2_grade': g2,
                        'max_points': max_pts
                    })
                    # For now, use average for disagreements
                    final_grades[qid] = (g1 + g2) / 2
                else:
                    final_grades[qid] = g1  # Use LLM1 (batch) as primary

                # Store comparison data for audit
                llm_comparison_data[qid] = {
                    llm1_name: {'grade': g1, 'max_points': max_pts, 'mode': 'batch'},
                    llm2_name: {'grade': g2, 'max_points': max_pts, 'mode': 'individual'},
                    'final': {
                        'grade': final_grades[qid],
                        'agreement': abs(g1 - g2) < relative_threshold
                    }
                }

            # Resolve student name
            student_name = original_copy.student_name or llm1_name_detected or llm2_name_detected

            # Build llm_comparison for audit
            copy_llm_comparison = {
                "options": {
                    "mode": "hybrid",
                    "providers": [llm1_name, llm2_name],
                    "llm1_mode": "batch",
                    "llm2_mode": "individual"
                },
                "llm_comparison": llm_comparison_data
            }

            # Create GradedCopy with full audit data
            graded = GradedCopy(
                copy_id=original_copy.id,
                grades=final_grades,
                total_score=sum(final_grades.values()),
                max_score=sum(self.grading_scale.values()),
                confidence=0.85 if not disagreements else 0.70,
                feedback=f"Hybrid mode. Disagreements: {len(disagreements)}",
                llm_comparison=copy_llm_comparison
            )

            if student_name:
                original_copy.student_name = student_name

            graded_copies.append(graded)
            self.session.graded_copies.append(graded)

            if progress_callback:
                await self._call_callback(progress_callback, 'copy_done', {
                    'copy_index': i + 1,
                    'total_copies': total_copies,
                    'copy_id': original_copy.id,
                    'student_name': student_name,
                    'total_score': graded.total_score,
                    'max_score': graded.max_score,
                    'disagreements': [d['question_id'] for d in disagreements]
                })

        # Save and notify completion
        self._save_sync()

        if progress_callback:
            await self._call_callback(progress_callback, 'hybrid_done', {
                'total_copies': total_copies,
                'graded_copies': len(graded_copies),
                'patterns': batch_result.patterns if batch_result else {}
            })

        self._grading_complete = True
        return graded_copies

    async def _grade_all_dual_llm(
        self,
        progress_callback: callable = None
    ) -> List[GradedCopy]:
        """
        Grade all copies using dual LLM mode.
        Uses parallel processing for copies (configurable via --parallel).
        """
        if not hasattr(self.ai, 'grade_copy'):
            raise RuntimeError("AI provider does not support stateless grading")

        self.session.status = SessionStatus.GRADING

        # Set progress callback
        if hasattr(self.ai, 'set_progress_callback') and progress_callback:
            self.ai.set_progress_callback(progress_callback)

        total_copies = len(self.session.copies)
        provider_names = [name for name, _ in self.ai.providers] if hasattr(self.ai, 'providers') else []

        questions = []
        for q_id in sorted(self.detected_questions.keys(), key=question_sort_key):
            q_text = self.detected_questions[q_id]
            questions.append({
                "id": q_id,
                "text": q_text,
                "criteria": "",
                "max_points": self.grading_scale.get(q_id, 1.0)
            })

        # Semaphore to limit concurrent grading
        semaphore = asyncio.Semaphore(self._parallel)

        async def grade_one_copy(i: int, copy: CopyDocument) -> Optional[GradedCopy]:
            """Grade a single copy with semaphore for concurrency control."""
            async with semaphore:
                # Notify copy start
                if progress_callback:
                    await self._call_callback(progress_callback, 'copy_start', {
                        'copy_index': i + 1,
                        'total_copies': total_copies,
                        'copy_id': copy.id,
                        'student_name': copy.student_name,
                        'questions': list(self.grading_scale.keys())
                    })

                # Get image paths for this copy
                image_paths = [str(p) for p in copy.page_images] if copy.page_images else []

                if not image_paths:
                    return None

                # Notify single-pass start
                if progress_callback:
                    await self._call_callback(progress_callback, 'single_pass_start', {
                        'num_questions': len(questions),
                        'providers': provider_names
                    })

                # Call stateless grading
                result = await self.ai.grade_copy(
                    questions=questions,
                    image_paths=image_paths,
                    language="fr",
                    disagreement_callback=self._disagreement_callback,
                    reading_disagreement_callback=self._reading_disagreement_callback,
                    second_reading=self._second_reading
                )

                # Extract audit info for workflow display
                audit = result.get("audit", {})
                single_pass = audit.get("single_pass", {})
                disagreement_report = audit.get("disagreement_report", {})
                verification = audit.get("verification", {})

                # Notify single-pass complete with results
                if progress_callback:
                    await self._call_callback(progress_callback, 'single_pass_complete', {
                        'providers': provider_names,
                        'single_pass': single_pass
                    })

                # Notify analysis complete
                if progress_callback:
                    await self._call_callback(progress_callback, 'analysis_complete', {
                        'agreed': disagreement_report.get('agreed', 0),
                        'flagged': disagreement_report.get('flagged', 0),
                        'total': disagreement_report.get('total_questions', 0),
                        'flagged_questions': disagreement_report.get('flagged_questions', [])
                    })

                # Notify verification for each flagged question
                for flagged in disagreement_report.get('flagged_questions', []):
                    qid = flagged.get('question_id')
                    if progress_callback:
                        await self._call_callback(progress_callback, 'verification_start', {
                            'question_id': qid,
                            'reason': flagged.get('reason'),
                            'llm1_grade': flagged.get('llm1', {}).get('grade'),
                            'llm2_grade': flagged.get('llm2', {}).get('grade')
                        })

                # Notify final results per question (in natural order)
                results_data = result.get("results", {})
                for q_id in sorted(results_data.keys(), key=question_sort_key):
                    q_result = results_data[q_id]
                    q_audit = verification.get(q_id, {})
                    method = q_audit.get('method', 'unknown')
                    agreement = q_audit.get('agreement', True)

                    # Use detected max_points from grading, fallback to question_scales
                    detected_max_points = q_result.get("max_points", self.grading_scale.get(q_id, 1.0))

                    if progress_callback:
                        await self._call_callback(progress_callback, 'question_done', {
                            'copy_index': i + 1,
                            'question_id': q_id,
                            'grade': q_result.get("grade", 0),
                            'max_points': detected_max_points,
                            'method': method,
                            'agreement': agreement
                        })

                # Convert to GradedCopy
                graded = GradedCopy(
                    copy_id=copy.id,
                    policy_version=self.session.policy.version
                )

                # Update student name from consensus (dual LLM already extracted it)
                consensus_name = result.get("student_name")
                if consensus_name:
                    copy.student_name = consensus_name

                # Extract grades from results section (already in natural order from above)
                for q_id in sorted(results_data.keys(), key=question_sort_key):
                    q_result = results_data[q_id]
                    grade = q_result.get("grade", 0)
                    feedback = q_result.get("feedback", "")
                    reading = q_result.get("reading", "")
                    reasoning_text = q_result.get("reasoning", "")

                    graded.grades[q_id] = grade
                    graded.student_feedback[q_id] = feedback
                    graded.readings[q_id] = reading
                    if reasoning_text:
                        graded.reasoning[q_id] = reasoning_text

                    # Store detected max_points
                    detected_max = q_result.get("max_points", 1.0)
                    graded.max_points_by_question[q_id] = detected_max

                # Get confidence from llm_comparison if available
                llm_comparison = result.get("llm_comparison", {})
                for q_id in llm_comparison.keys():
                    final_info = llm_comparison.get(q_id, {}).get("final", {})
                    graded.confidence_by_question[q_id] = final_info.get("confidence", 0.5)

                # Calculate totals using detected max_points
                graded.total_score = sum(g or 0 for g in graded.grades.values())
                graded.max_score = sum(graded.max_points_by_question.values()) if graded.max_points_by_question else sum(self.grading_scale.values())
                graded.confidence = sum(c or 0.5 for c in graded.confidence_by_question.values()) / len(graded.confidence_by_question) if graded.confidence_by_question else 0.5

                # Generate overall feedback from per-question feedbacks
                all_feedbacks = [fb for fb in graded.student_feedback.values() if fb]
                if all_feedbacks:
                    graded.feedback = " | ".join(all_feedbacks[:3])  # First 3 feedbacks
                    if len(all_feedbacks) > 3:
                        graded.feedback += "..."

                # Store full audit data with new structure
                graded.llm_comparison = {
                    "options": result.get("options", {}),
                    "llm_comparison": llm_comparison,
                    "student_name_info": result.get("student_name_info", {}),
                    "summary": result.get("summary", {}),
                    "timing": result.get("timing", {})
                }

                # Notify copy done
                if progress_callback:
                    token_usage = None
                    if hasattr(self.ai, 'get_token_usage'):
                        token_usage = self.ai.get_token_usage()

                    await self._call_callback(progress_callback, 'copy_done', {
                        'copy_index': i + 1,
                        'total_copies': total_copies,
                        'copy_id': copy.id,
                        'student_name': copy.student_name,
                        'total_score': graded.total_score,
                        'max_score': graded.max_score,
                        'confidence': graded.confidence,
                        'token_usage': token_usage,
                        'grading_summary': result.get("summary", {})
                    })

                return graded

        # Create tasks for all copies
        tasks = [
            grade_one_copy(i, copy)
            for i, copy in enumerate(self.session.copies)
        ]

        # Execute all tasks in parallel (limited by semaphore)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful results and emit error events
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                import logging
                logging.error(f"Error grading copy {i+1}: {result}")
                # Emit copy_error event for live display
                if progress_callback:
                    copy = self.session.copies[i] if i < len(self.session.copies) else None
                    await self._call_callback(progress_callback, 'copy_error', {
                        'copy_index': i + 1,
                        'total_copies': total_copies,
                        'copy_id': copy.id if copy else None,
                        'student_name': copy.student_name if copy else None,
                        'error': str(result)
                    })
            elif result is not None:
                self.session.graded_copies.append(result)
                self._save_sync()

        self._grading_complete = True
        return self.session.graded_copies

    async def _call_callback(self, callback: callable, event_type: str, data: dict):
        """Safely call a callback, handling both sync and async."""
        if callback is None:
            return
        try:
            result = callback(event_type, data)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            # Don't let callback errors break grading, but log them
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Callback error for {event_type}: {e}")

    async def _ask_max_points_resolution(self, disagreement: dict, llm1_name: str, llm2_name: str) -> float:
        """
        Ask user to resolve a max_points disagreement after ultimatum.

        Args:
            disagreement: Dict with copy_index, question_id, llm1_max_points, llm2_max_points
            llm1_name: Name of first LLM
            llm2_name: Name of second LLM

        Returns:
            Resolved max_points value
        """
        from interaction.cli import console
        from rich.prompt import Prompt
        from rich.panel import Panel
        from rich.table import Table

        copy_idx = disagreement['copy_index']
        qid = disagreement['question_id']
        llm1_mp = disagreement['llm1_max_points']
        llm2_mp = disagreement['llm2_max_points']

        # Display the disagreement
        console.print(Panel(
            f"[bold yellow]Désaccord sur le barème après ultimatum[/bold yellow]\n\n"
            f"Copie {copy_idx}, {qid}\n\n"
            f"[cyan]{llm1_name}[/cyan]: {llm1_mp} pts\n"
            f"[magenta]{llm2_name}[/magenta]: {llm2_mp} pts",
            title="Résolution requise",
            border_style="yellow"
        ))

        # Calculate options
        avg_mp = (llm1_mp + llm2_mp) / 2
        max_mp = max(llm1_mp, llm2_mp)

        console.print(f"\nOptions:")
        console.print(f"  1. {llm1_name}: {llm1_mp} pts")
        console.print(f"  2. {llm2_name}: {llm2_mp} pts")
        console.print(f"  3. Moyenne: {avg_mp:.2f} pts")
        console.print(f"  4. Maximum: {max_mp} pts")
        console.print(f"  5. Personnalisé")

        choice = Prompt.ask(
            "Choisir le barème",
            choices=["1", "2", "3", "4", "5"],
            default="3"
        )

        if choice == "1":
            return llm1_mp
        elif choice == "2":
            return llm2_mp
        elif choice == "3":
            return avg_mp
        elif choice == "4":
            return max_mp
        else:
            custom = Prompt.ask("Entrer le barème personnalisé", default=str(avg_mp))
            return float(custom)

    async def _ask_student_name_resolution(self, disagreement: dict, llm1_name: str, llm2_name: str) -> str:
        """
        Ask user to resolve a student name disagreement after ultimatum.

        Args:
            disagreement: Dict with llm1_final_name, llm2_final_name
            llm1_name: Name of first LLM
            llm2_name: Name of second LLM

        Returns:
            Resolved student name
        """
        from interaction.cli import console
        from rich.prompt import Prompt
        from rich.panel import Panel

        llm1_student = disagreement.get('llm1_final_name', '')
        llm2_student = disagreement.get('llm2_final_name', '')

        # Display the disagreement
        console.print(Panel(
            f"[bold yellow]Désaccord sur le nom de l'étudiant après ultimatum[/bold yellow]\n\n"
            f"[cyan]{llm1_name}[/cyan]: {llm1_student}\n"
            f"[magenta]{llm2_name}[/magenta]: {llm2_student}",
            title="Résolution requise",
            border_style="yellow"
        ))

        console.print(f"\nOptions:")
        console.print(f"  1. {llm1_name}: {llm1_student}")
        console.print(f"  2. {llm2_name}: {llm2_student}")
        console.print(f"  3. Personnalisé")

        choice = Prompt.ask(
            "Choisir le nom de l'étudiant",
            choices=["1", "2", "3"],
            default="1"
        )

        if choice == "1":
            return llm1_student
        elif choice == "2":
            return llm2_student
        else:
            custom = Prompt.ask("Entrer le nom", default=llm1_student)
            return custom

    def get_doubts(self, threshold: float = 0.7) -> List[Tuple[CopyDocument, GradedCopy, str, float]]:
        """
        Get list of doubtful cases (low confidence grades).

        Args:
            threshold: Confidence threshold below which grades are considered doubtful

        Returns:
            List of (copy, graded_copy, question_id, confidence) tuples
        """
        doubts = []

        for graded in self.session.graded_copies:
            for q_id, confidence in graded.confidence_by_question.items():
                if confidence < threshold:
                    # Find corresponding copy
                    copy = next(
                        (c for c in self.session.copies if c.id == graded.copy_id),
                        None
                    )
                    if copy:
                        doubts.append((copy, graded, q_id, confidence))

        return doubts

    async def apply_decisions(self, decisions: List) -> None:
        """
        Phase 5: Apply user decisions and propagate to similar copies.

        Args:
            decisions: List of Decision objects from CLI.review_doubts()
        """
        for decision in decisions:
            # Find the graded copy
            graded = next(
                (g for g in self.session.graded_copies if g.copy_id == decision.copy_id),
                None
            )
            if graded and decision.question_id in graded.grades:
                # Update the grade
                old_grade = graded.grades[decision.question_id]
                new_grade = decision.new_grade

                # Warn if grades are None - this indicates a pipeline problem
                if old_grade is None:
                    import logging
                    logging.warning(f"Grade is None for {decision.question_id} in copy {copy_id} - pipeline issue?")
                    old_grade = 0
                if new_grade is None:
                    import logging
                    logging.warning(f"New grade is None for {decision.question_id} - decision issue?")
                    new_grade = 0

                graded.grades[decision.question_id] = new_grade

                # Update total score
                graded.total_score = (graded.total_score or 0) - old_grade + new_grade

                # Propagate if requested
                if decision.propagate and decision.similar_copy_ids:
                    await self._propagate_decision(decision)

        self._save_sync()

    async def _propagate_decision(self, decision) -> int:
        """
        Propagate a decision to similar copies.

        Args:
            decision: Decision object

        Returns:
            Number of copies updated
        """
        count = 0
        for copy_id in decision.similar_copy_ids:
            graded = next(
                (g for g in self.session.graded_copies if g.copy_id == copy_id),
                None
            )
            if graded and decision.question_id in graded.grades:
                old_grade = graded.grades[decision.question_id]
                new_grade = decision.new_grade

                # Warn if grades are None - this indicates a pipeline problem
                if old_grade is None:
                    import logging
                    logging.warning(f"Grade is None for {decision.question_id} in copy {copy_id} during propagation")
                    old_grade = 0
                if new_grade is None:
                    import logging
                    logging.warning(f"New grade is None for {decision.question_id} during propagation")
                    new_grade = 0

                graded.grades[decision.question_id] = new_grade
                graded.total_score = (graded.total_score or 0) - old_grade + new_grade
                count += 1

        return count

    async def export(self) -> Dict[str, str]:
        """
        Phase 6: Export results.

        Returns:
            Dict of {format: file_path}
        """
        return await self._export_phase()

    def get_analytics(self) -> Dict:
        """
        Get analytics for the session.

        Returns:
            Analytics dict with statistics
        """
        analytics = AnalyticsGenerator(self.session)
        report = analytics.generate()

        return {
            'mean_score': report.mean_score,
            'median_score': report.median_score,
            'std_dev': report.std_dev,
            'min_score': report.min_score,
            'max_score': report.max_score,
            'score_distribution': report.score_distribution,
            'question_stats': {q: {'mean': s.get('mean', 0)} for q, s in report.question_stats.items()}
        }

    # ==================== INTERNAL METHODS ====================

    async def _load_copies_phase(self):
        """Phase 0: Load PDFs and extract initial content.

        Logic:
        1. --auto-detect-structure: AI detects structure in batch (all students, names)
           - If --pages-per-copy: THEN split mechanically for grading
           - Else: use AI-detected page distribution
        2. --pages-per-copy alone: Mechanical split only, no AI (names detected during grading)
        3. Neither: Structure detected during grading (batch mode)
        """
        from rich.console import Console
        from rich.progress import track

        console = Console()

        if self._auto_detect_structure:
            # AUTO-DETECT MODE: AI detects structure (both LLMs)
            console.print("[bold cyan]🔍 Auto-détection de la structure (2 LLMs)...[/bold cyan]")
            await self._load_copies_with_dual_llm()
            # After AI detection, --pages-per-copy determines how to split for grading
            if self._pages_per_copy:
                console.print(f"[dim]Structure détectée, split {self._pages_per_copy} pages/copy pour correction[/dim]")
                # Re-split copies based on pages_per_copy (names already known)
                await self._resplit_by_pages()
        elif self._pages_per_copy:
            # MECHANICAL SPLIT ONLY: No AI, names detected during grading
            await self._load_copies_individual_mode()
        else:
            # NO PRE-ANALYSIS: Structure detected during grading
            console.print("[dim]Structure non pré-analysée (détection pendant la correction)[/dim]")
            await self._load_copies_minimal()

        self._save_sync()

    async def _load_copies_ensemble_mode(self):
        """Load copies using AI to detect students (ensemble mode)."""
        from rich.progress import track

        for pdf_path in track(self.pdf_paths, description="Loading PDFs..."):
            reader = PDFReader(pdf_path)
            page_count = reader.get_page_count()

            # Extract content for ALL students in this PDF
            students_data = await self._extract_all_students(pdf_path, reader)

            # Create a CopyDocument for each detected student
            for i, student_data in enumerate(students_data):
                copy = CopyDocument(
                    pdf_path=pdf_path,
                    page_count=page_count,
                    student_name=student_data.get('student_name'),
                    content_summary=student_data.get('content', {}),
                    language=student_data.get('language', 'fr'),
                    page_images=student_data.get('page_images', [])
                )

                self.session.copies.append(copy)

                # Extract only the pages belonging to this student
                pdf_bytes = self._extract_student_pages(pdf_path, student_data.get('page_images', []))
                self.store.save_copy(copy, pdf_bytes)

            reader.close()

    async def _load_copies_with_dual_llm(self):
        """
        Load copies using BOTH LLMs to detect structure.

        Uses cross-verification + ultimatum workflow for disagreements.
        Analyzes ENTIRE PDF to detect:
        - Number of students
        - Pages per student
        - Student names
        - Questions and barème (if visible)
        """
        from rich.progress import track
        from rich.console import Console

        console = Console()

        for pdf_path in track(self.pdf_paths, description="Analyzing PDF structure..."):
            reader = PDFReader(pdf_path)
            page_count = reader.get_page_count()

            # Check if we have a comparison provider (dual LLM)
            if hasattr(self.ai, 'providers') and len(self.ai.providers) >= 2:
                # Use both LLMs with cross-verification
                students_data = await self._extract_structure_with_consensus(pdf_path, reader)
            else:
                # Fallback to single LLM
                console.print("[yellow]⚠ Single LLM mode - no cross-verification[/yellow]")
                students_data = await self._extract_all_students(pdf_path, reader)

            # Create a CopyDocument for each detected student
            for i, student_data in enumerate(students_data):
                copy = CopyDocument(
                    pdf_path=pdf_path,
                    page_count=len(student_data.get('page_images', [])) or page_count,
                    student_name=student_data.get('student_name'),
                    content_summary=student_data.get('content', {}),
                    language=student_data.get('language', 'fr'),
                    page_images=student_data.get('page_images', [])
                )

                self.session.copies.append(copy)

                # Extract only the pages belonging to this student
                pdf_bytes = self._extract_student_pages(pdf_path, student_data.get('page_images', []))
                self.store.save_copy(copy, pdf_bytes)

            reader.close()

        # Mark structure as pre-detected (skip name/barème detection during grading)
        self._structure_pre_detected = True

    async def _extract_structure_with_consensus(self, pdf_path: str, reader) -> List[Dict]:
        """
        Extract PDF structure using both LLMs with consensus workflow.

        Workflow:
        1. Both LLMs analyze the PDF
        2. If agreement → done
        3. If disagreement → cross-verification
        4. If still disagreement → ultimatum
        5. If still disagreement → ask user (or auto-resolve in auto mode)
        """
        import hashlib
        import asyncio
        from rich.console import Console

        console = Console()
        page_count = reader.get_page_count()

        # Convert pages to images
        temp_dir = self.store.session_dir / "temp_images"
        temp_dir.mkdir(parents=True, exist_ok=True)

        page_images = []
        for page_num in range(page_count):
            image_bytes = reader.get_page_image_bytes(page_num)
            image_path = str(temp_dir / f"page_{page_num}.png")
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            page_images.append(image_path)

        # Prompt for structure detection
        prompt = f"""Analyse ce document PDF de {page_count} pages.

TA TÂCHE: Identifier la structure du document.

RÉPONDS EN JSON:
{{
  "students": [
    {{
      "name": "Nom de l'élève",
      "pages": [1, 2],
      "questions": {{"Q1": "résumé", "Q2": "résumé"}}
    }}
  ],
  "bareme": {{"Q1": 2, "Q2": 3}},
  "language": "fr"
}}

IMPORTANT:
- Détecte TOUS les élèves présents
- Indique les numéros de page pour chaque élève
- Détecte le barème si visible
"""

        # Call both LLMs in parallel
        console.print(f"  [dim]Appel LLM1 + LLM2 ({page_count} pages)...[/dim]")

        async def call_llm(name, provider):
            try:
                response = provider.call_vision(prompt, image_path=page_images)
                return (name, response)
            except Exception as e:
                console.print(f"  [red]Erreur {name}: {e}[/red]")
                return (name, None)

        tasks = [call_llm(name, provider) for name, provider in self.ai.providers]
        results = await asyncio.gather(*tasks)

        llm1_name, llm1_response = results[0]
        llm2_name, llm2_response = results[1]

        # Parse responses
        llm1_data = self._parse_structure_response(llm1_response) if llm1_response else None
        llm2_data = self._parse_structure_response(llm2_response) if llm2_response else None

        # Check for agreement
        if llm1_data and llm2_data:
            agreement = self._compare_structures(llm1_data, llm2_data)

            if agreement['agreed']:
                console.print(f"  [green]✓ Accord: {len(llm1_data.get('students', []))} élève(s)[/green]")
                return self._build_students_from_structure(llm1_data, page_images)

            # Disagreement - try cross-verification
            console.print(f"  [yellow]⚠ Désaccord: {agreement['reason']}[/yellow]")
            console.print(f"    LLM1: {llm1_data.get('students', [])}")
            console.print(f"    LLM2: {llm2_data.get('students', [])}")

            # For now, use LLM1 result (TODO: implement cross-verification + ultimatum)
            console.print(f"  [yellow]→ Utilisation de {llm1_name}[/yellow]")
            return self._build_students_from_structure(llm1_data, page_images)

        # One LLM failed - use the other
        working_data = llm1_data or llm2_data
        working_name = llm1_name if llm1_data else llm2_name

        if working_data:
            console.print(f"  [yellow]→ Un LLM échoué, utilisation de {working_name}[/yellow]")
            return self._build_students_from_structure(working_data, page_images)

        # Both failed - fallback to heuristic
        console.print(f"  [red]✗ Les deux LLMs ont échoué[/red]")
        return []

    def _parse_structure_response(self, response: str) -> Optional[Dict]:
        """Parse JSON structure response from LLM."""
        if not response:
            return None

        return extract_json_from_response(response)

    def _compare_structures(self, data1: Dict, data2: Dict) -> Dict:
        """Compare two structure detections for agreement."""
        students1 = data1.get('students', [])
        students2 = data2.get('students', [])

        # Compare number of students
        if len(students1) != len(students2):
            return {
                'agreed': False,
                'reason': f"Nombre d'élèves différent: {len(students1)} vs {len(students2)}"
            }

        # Compare names (normalized)
        names1 = set(s.get('name', '').lower().strip() for s in students1)
        names2 = set(s.get('name', '').lower().strip() for s in students2)

        if names1 != names2:
            return {
                'agreed': False,
                'reason': f"Noms différents: {names1} vs {names2}"
            }

        return {'agreed': True}

    def _build_students_from_structure(self, structure: Dict, page_images: List[str]) -> List[Dict]:
        """Build student data list from parsed structure."""
        students = []
        language = structure.get('language', 'fr')

        for student in structure.get('students', []):
            pages = student.get('pages', [])
            student_images = [page_images[p - 1] for p in pages if 0 < p <= len(page_images)]

            students.append({
                'student_name': student.get('name'),
                'content': student.get('questions', {}),
                'page_images': student_images,
                'language': language
            })

        # Store detected barème
        bareme = structure.get('bareme', {})
        if bareme:
            for q_id, points in bareme.items():
                if q_id not in self.grading_scale:
                    self.grading_scale[q_id] = points

        return students if students else self._build_default_students(page_images, language)

    def _build_default_students(self, page_images: List[str], language: str = 'fr') -> List[Dict]:
        """Build default single student when structure detection fails."""
        return [{
            'student_name': None,
            'content': {},
            'page_images': page_images,
            'language': language
        }]

    async def _load_copies_minimal(self):
        """
        Minimal loading without structure detection.

        Just creates placeholder copies - structure will be detected during grading.
        Used when neither --auto-detect-structure nor --pages-per-copy is set.
        """
        from rich.console import Console

        console = Console()

        for pdf_path in self.pdf_paths:
            reader = PDFReader(pdf_path)
            page_count = reader.get_page_count()

            # Create a single copy for the entire PDF
            # Structure (students, names) will be detected during grading
            copy = CopyDocument(
                pdf_path=pdf_path,
                page_count=page_count,
                student_name=None,  # Will be detected during grading
                content_summary={},
                language='fr'  # Will be detected during grading
            )

            self.session.copies.append(copy)

            # Save the entire PDF
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            self.store.save_copy(copy, pdf_bytes)

            reader.close()

    async def _resplit_by_pages(self):
        """
        Re-split copies after AI structure detection.

        Used when --auto-detect-structure detects students,
        then --pages-per-copy determines how to split for grading.
        """
        from vision.pdf_reader import split_pdf_by_ranges
        from rich.console import Console

        console = Console()
        pages_per_copy = self._pages_per_copy

        # Store detected names before re-splitting
        detected_names = {copy.id: copy.student_name for copy in self.session.copies}

        new_copies = []
        for copy in self.session.copies:
            reader = PDFReader(copy.pdf_path)
            total_pages = reader.get_page_count()
            reader.close()

            # Calculate ranges
            ranges = []
            for start in range(0, total_pages, pages_per_copy):
                end = min(start + pages_per_copy - 1, total_pages - 1)
                ranges.append((start, end))

            # Split PDF
            split_dir = Path(self.store.session_dir) / "splits"
            split_dir.mkdir(parents=True, exist_ok=True)
            split_paths = split_pdf_by_ranges(copy.pdf_path, str(split_dir), ranges)

            # Create new copies for each split
            for i, split_path in enumerate(split_paths):
                new_copy = CopyDocument(
                    pdf_path=split_path,
                    page_count=pages_per_copy,
                    student_name=detected_names.get(copy.id),  # Keep detected name
                    content_summary={},
                    language=copy.language
                )
                new_copies.append(new_copy)

                with open(split_path, 'rb') as f:
                    pdf_bytes = f.read()
                self.store.save_copy(new_copy, pdf_bytes)

        # Replace copies with new split copies
        self.session.copies = new_copies

    async def _load_copies_individual_mode(self):
        """
        Load copies by pre-splitting PDF (no AI needed for student detection).

        This mode:
        1. Splits the PDF into chunks of `pages_per_copy` pages each
        2. Analyzes each chunk as a separate student copy
        3. Uses AI consensus for name detection (if in comparison mode)
        """
        from vision.pdf_reader import split_pdf_by_ranges
        from rich.console import Console
        import hashlib

        console = Console()
        pages_per_copy = self._pages_per_copy

        copy_index = 0

        for pdf_path in self.pdf_paths:
            reader = PDFReader(pdf_path)
            total_pages = reader.get_page_count()
            reader.close()

            # Calculate ranges: [(0, 1), (2, 3), (4, 5), ...] for pages_per_copy=2
            ranges = []
            for start in range(0, total_pages, pages_per_copy):
                end = min(start + pages_per_copy - 1, total_pages - 1)
                ranges.append((start, end))

            num_students = len(ranges)

            # Create temporary directory for split PDFs
            split_dir = Path(self.store.session_dir) / "splits"
            split_dir.mkdir(parents=True, exist_ok=True)

            # Split PDF into individual student PDFs
            split_paths = split_pdf_by_ranges(pdf_path, str(split_dir), ranges)

            # Analyze each split PDF
            for i, split_path in enumerate(split_paths):
                copy_index += 1

                # Analyze this single copy
                copy = await self._analyze_single_copy(
                    split_path,
                    copy_index=copy_index,
                    total_copies=num_students
                )

                self.session.copies.append(copy)

                # Save the copy
                with open(split_path, 'rb') as f:
                    pdf_bytes = f.read()
                self.store.save_copy(copy, pdf_bytes)

    async def _analyze_single_copy(
        self,
        pdf_path: str,
        copy_index: int,
        total_copies: int
    ) -> CopyDocument:
        """
        Prepare a single student copy for grading.

        NO API CALL HERE - all analysis is done during grading phase.
        This method only:
        1. Converts PDF pages to images
        2. Creates a minimal CopyDocument

        The actual analysis (student name, questions, grades) happens in
        the single-pass grading phase, which is more efficient.
        """
        import hashlib

        reader = PDFReader(pdf_path)
        page_count = reader.get_page_count()

        # Create temp directory for this session's images (auto-cleanup)
        temp_dir = self.store.session_dir / "temp_images"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Convert pages to images
        page_images = []
        for page_num in range(page_count):
            image_bytes = reader.get_page_image_bytes(page_num)
            # Use session-specific temp directory instead of /tmp
            image_path = str(temp_dir / f"page_{copy_index}_{page_num}.png")
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            page_images.append(image_path)

        reader.close()

        # Create minimal CopyDocument - no API call needed
        # Student name and questions will be detected during grading
        copy = CopyDocument(
            id=f"copy_{copy_index}",
            pdf_path=pdf_path,
            page_count=page_count,
            student_name=None,  # Will be set during grading
            page_images=page_images,
            language=None,  # Will be detected during grading
            created_at=datetime.now(),
            processed=False
        )

        return copy

    def _extract_student_pages(self, pdf_path: str, page_images: List[str]) -> bytes:
        """
        Extract only the pages belonging to a specific student.

        Args:
            pdf_path: Path to the original PDF
            page_images: List of page image paths for this student
                         (e.g., ['/tmp/pdf_xxx_page0.png', '/tmp/pdf_xxx_page1.png'])

        Returns:
            bytes: PDF containing only the student's pages
        """
        import fitz  # PyMuPDF
        import re

        if not page_images:
            # No specific pages - return entire PDF
            with open(pdf_path, 'rb') as f:
                return f.read()

        # Extract page numbers from image paths (e.g., "page0.png" -> 0)
        page_numbers = []
        for img_path in page_images:
            match = re.search(r'page(\d+)\.png$', img_path)
            if match:
                page_numbers.append(int(match.group(1)))

        if not page_numbers:
            # Could not extract page numbers - return entire PDF
            with open(pdf_path, 'rb') as f:
                return f.read()

        # Create new PDF with only the selected pages
        with fitz.open(pdf_path) as src_doc:
            new_doc = fitz.open()

            for page_num in sorted(page_numbers):
                if 0 <= page_num < len(src_doc):
                    new_doc.insert_pdf(src_doc, from_page=page_num, to_page=page_num)

            # Convert to bytes
            pdf_bytes = new_doc.tobytes()
            new_doc.close()

            return pdf_bytes

    async def _extract_all_students(self, pdf_path: str, reader) -> List[Dict]:
        """
        Extract all students from a PDF using structured AI analysis.

        Uses a SINGLE API call with ALL pages so the AI can:
        1. See the full context
        2. Identify how many students are in the document
        3. Assign pages to each student
        4. Extract answers and grading scale per student

        Returns:
            List of dicts, one per student, with:
            - 'student_name': str or None
            - 'content': Dict with Q1, Q1_points, Q1_confidence, etc.
            - 'page_images': List of image paths for this student
            - 'language': 'fr' or 'en'
        """
        import hashlib

        page_count = reader.get_page_count()

        # Create temp directory for this session's images (auto-cleanup)
        temp_dir = self.store.session_dir / "temp_images"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Convert all pages to images first
        page_images = []
        for page_num in range(page_count):
            image_bytes = reader.get_page_image_bytes(page_num)
            # Use session-specific temp directory instead of /tmp
            image_path = str(temp_dir / f"page_{page_num}.png")
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            page_images.append(image_path)

        # Check cache (non-blocking file read)
        def compute_file_hash(filepath: str) -> str:
            """Compute MD5 hash of file (blocking operation)."""
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()[:8]

        first_page_hash = await asyncio.to_thread(compute_file_hash, page_images[0])
        cache_key = f"multi_student_{first_page_hash}_{page_count}"
        cached = self.store.get_cached_analysis(cache_key)
        if cached:
            return cached.get('students', [])

        # Structured prompt for multi-student detection - NO barème (done in grading phase)
        prompt = f"""Analyse ce document PDF de {page_count} pages. Il peut contenir UN ou PLUSIEURS élèves.

TU VOIS TOUTES LES {page_count} PAGES DE CE DOCUMENT.

TA TÂCHE: Identifier les élèves et leurs réponses.

POUR CHAQUE ÉLÈVE IDENTIFIÉ:
1. Son nom (ou "Inconnu" si non visible)
2. Les numéros de pages qui lui appartiennent
3. Pour CHAQUE question: résumé bref de la réponse

FORMAT DE RÉPONSE STRICT:
=== ÉLÈVE 1 ===
NOM: [nom]
PAGES: [numéros, ex: 1, 2]
Q1: [résumé de la réponse]
Q2: [résumé de la réponse]

=== ÉLÈVE 2 ===
NOM: [nom]
PAGES: [numéros]
Q1: ...
Q2: ...

IMPORTANT:
- Résume ce que l'élève a écrit
- Réponds en français sauf si le document est en anglais"""

        # SINGLE API call with ALL pages
        try:
            print(f"  [API] Calling vision with all {page_count} pages...")
            response = self.ai.call_vision(prompt, image_path=page_images)
            print(f"  [API] Response received")
        except Exception as e:
            error_msg = str(e).lower()
            if any(word in error_msg for word in ['rate', 'limit', 'quota', '429']):
                print(f"  [ERROR] Rate limit / quota exceeded")
                raise RuntimeError(f"API quota exhausted. Error: {e}")
            elif 'timeout' in error_msg:
                raise RuntimeError(f"API timeout: {e}")
            else:
                raise RuntimeError(f"API error: {e}")

        # Parse response to extract students
        students = self._parse_multi_student_response([(1, response)], page_images)

        # Determine language
        language = 'en' if any(w in response.lower() for w in ['the', 'student', 'answer']) else 'fr'
        for student in students:
            student['language'] = language

        # Verify student names with consensus (if in comparison mode)
        students = await self._verify_student_names_consensus(students)

        # Cache the result
        self.store.cache_analysis(cache_key, {'students': students})

        return students

    def _parse_multi_student_response(
        self,
        responses: List[Tuple[int, str]],
        page_images: List[str]
    ) -> List[Dict]:
        """
        Parse multi-student AI response.

        Args:
            responses: List of (page_num, response_text) tuples
            page_images: List of all page image paths

        Returns:
            List of student dicts with 'student_name', 'content', 'page_images'
        """
        students = []
        current_student = None
        all_content = {}  # Merged content across pages

        # Combine all responses
        combined_text = "\n".join([r for _, r in responses])

        for line in combined_text.split('\n'):
            line = line.strip()

            # Detect new student section
            if line.startswith('=== ÉLÈVE') or line.startswith('=== ELEVE') or line.startswith('=== Student'):
                # Save previous student if exists and has content
                if current_student and (current_student.get('student_name') or current_student.get('content')):
                    current_student['content'] = self._merge_content(
                        all_content.get(current_student.get('student_name') or 'default', {})
                    )
                    # Only add if has actual content (questions)
                    if any(k.startswith('Q') for k in current_student['content'].keys()):
                        students.append(current_student)

                current_student = {
                    'student_name': None,
                    'content': {},
                    'page_images': page_images,  # Will be filtered later
                    'pages': []
                }
                continue

            # Skip lines that are clearly not data (intro text, etc.)
            if not current_student and (line.startswith('Ce document') or line.startswith('This document') or line.startswith('**')):
                continue

            # Parse NOM: or NAME:
            if line.upper().startswith('NOM:') or line.upper().startswith('NAME:'):
                if not current_student:
                    current_student = {
                        'student_name': None,
                        'content': {},
                        'page_images': page_images,
                        'pages': []
                    }
                try:
                    colon_idx = line.index(':')
                    name = line[colon_idx + 1:].strip()
                    if name and name.lower() != 'inconnu':
                        old_name = current_student.get('student_name')
                        current_student['student_name'] = name
                        # Migrate content to new name key
                        if old_name and old_name in all_content:
                            all_content[name] = all_content.pop(old_name, {})
                        else:
                            all_content[name] = all_content.get('default', {})
                except (ValueError, IndexError):
                    pass
                continue

            # Parse PAGES:
            if line.upper().startswith('PAGES:'):
                if not current_student:
                    current_student = {
                        'student_name': None,
                        'content': {},
                        'page_images': page_images,
                        'pages': []
                    }
                try:
                    colon_idx = line.index(':')
                    pages_str = line[colon_idx + 1:].strip()
                    # Parse "1, 2, 3" or "1-2" format
                    pages = []
                    for part in pages_str.split(','):
                        part = part.strip()
                        if '-' in part:
                            start, end = part.split('-')
                            pages.extend(range(int(start), int(end) + 1))
                        elif part.isdigit():
                            pages.append(int(part))
                    current_student['pages'] = pages
                    # Filter page images for this student
                    if pages:
                        current_student['page_images'] = [
                            page_images[p - 1] for p in pages if 0 < p <= len(page_images)
                        ]
                except (ValueError, IndexError):
                    pass
                continue

            # Parse Q1:, Q2:, etc.
            if line.startswith('Q') and ':' in line:
                # Only create default student when we see actual question content
                if not current_student:
                    current_student = {
                        'student_name': None,
                        'content': {},
                        'page_images': page_images,
                        'pages': []
                    }
                    all_content['default'] = {}

                parsed = self._parse_question_line(line)
                if parsed:
                    q_id, content = parsed
                    student_name = current_student.get('student_name') or 'default'
                    if student_name not in all_content:
                        all_content[student_name] = {}
                    all_content[student_name][q_id] = content.get('answer', '')
                    # Only add points if not None (must be string for Pydantic)
                    if content.get('points') is not None:
                        all_content[student_name][f"{q_id}_points"] = str(content['points'])
                    if content.get('confidence'):
                        all_content[student_name][f"{q_id}_confidence"] = str(content['confidence'])

        # Don't forget the last student (only if it has actual content)
        if current_student:
            student_name = current_student.get('student_name') or 'default'
            current_student['content'] = self._merge_content(
                all_content.get(student_name, all_content.get('default', {}))
            )
            # Only add if has actual question content
            if any(k.startswith('Q') and not k.endswith(('_points', '_confidence', '_points_unknown'))
                   for k in current_student['content'].keys()):
                students.append(current_student)

        # If no students were detected, create one default student with all content
        if not students:
            merged_content = {}
            for page_num, response in responses:
                page_content = self._parse_content_summary(response)
                for key, value in page_content.items():
                    if key in merged_content:
                        merged_content[key] = merged_content[key] + " " + value
                    else:
                        merged_content[key] = value

            students = [{
                'student_name': merged_content.get('_student_name'),
                'content': {k: v for k, v in merged_content.items() if not k.startswith('_')},
                'page_images': page_images,
                'language': 'fr'
            }]

        return students

    async def _verify_student_names_consensus(
        self,
        students: List[Dict]
    ) -> List[Dict]:
        """
        Verify student names using LLM consensus (for comparison mode).

        Args:
            students: List of student dicts from initial analysis

        Returns:
            Updated list of students with verified names
        """
        # Only use consensus if we have a ComparisonProvider
        if not hasattr(self.ai, 'detect_student_name_with_consensus'):
            return students

        import asyncio

        for student in students:
            page_images = student.get('page_images', [])
            if not page_images:
                continue

            # Use first page for name detection
            first_page = page_images[0] if isinstance(page_images, list) else page_images
            language = student.get('language', 'fr')

            try:
                # Detect with consensus (both LLMs)
                # If both agree → returns immediately (no extra calls)
                # If one fails → uses other result (no cross-verification)
                # If disagree → cross-verification
                result = await self.ai.detect_student_name_with_consensus(
                    image_path=first_page,
                    language=language,
                    name_disagreement_callback=self._name_disagreement_callback
                )

                consensus_name = result.get('name')
                comparison = result.get('comparison', {})

                if consensus_name:
                    old_name = student.get('student_name')
                    student['student_name'] = consensus_name

                    # Log if there was a discrepancy
                    if old_name and old_name != consensus_name:
                        print(f"  [CONSENSUS] Nom: '{old_name}' → '{consensus_name}'")
                        if not result.get('consensus'):
                            print(f"    ⚠ Désaccord résolu:")
                            print(f"      - {comparison.get('llm1', {}).get('provider')}: {comparison.get('llm1', {}).get('name')}")
                            print(f"      - {comparison.get('llm2', {}).get('provider')}: {comparison.get('llm2', {}).get('name')}")
                    else:
                        print(f"  [CONSENSUS] Nom confirmé: {consensus_name}")

            except Exception as e:
                print(f"  [WARN] Erreur consensus nom: {e}")

        return students

    def _parse_question_line(self, line: str) -> Optional[Tuple[str, Dict]]:
        """Parse a question line like 'Q1: answer | BARÈME: 2 pts | confiance: haut'."""
        import re
        try:
            colon_idx = line.index(':')
            key_part = line[:colon_idx].strip()

            if not (key_part.startswith('Q') and key_part[1:].split()[0].isdigit()):
                return None

            # Extract question number
            q_num = ''.join(c for c in key_part[1:] if c.isdigit())
            q_id = f"Q{q_num}"

            value = line[colon_idx + 1:].strip()
            result = {'answer': '', 'points': None, 'bareme_confidence': 'moyen'}

            if '|' in value:
                parts = [p.strip() for p in value.split('|')]
                result['answer'] = parts[0] if parts else ''

                for part in parts[1:]:
                    part_lower = part.lower()
                    if 'confiance' in part_lower or 'confidence' in part_lower:
                        if 'haut' in part_lower or 'high' in part_lower:
                            result['bareme_confidence'] = 'haut'
                        elif 'bas' in part_lower or 'low' in part_lower:
                            result['bareme_confidence'] = 'bas'
                        else:
                            result['bareme_confidence'] = 'moyen'
                    elif 'barème' in part_lower or 'bareme' in part_lower:
                        # Extract points from "BARÈME: X pts" or "BARÈME: ?"
                        if '?' in part:
                            result['points'] = None
                        else:
                            match = re.search(r'([\d.]+)', part)
                            if match:
                                result['points'] = match.group(1)
                    elif part == '?' or 'inconnu' in part_lower:
                        result['points'] = None
                    elif result['points'] is None:
                        # Fallback: try to extract number if not already found
                        match = re.search(r'([\d.]+)', part)
                        if match:
                            result['points'] = match.group(1)
            else:
                result['answer'] = value

            return (q_id, result)

        except (ValueError, IndexError):
            return None

    def _merge_content(self, content: Dict) -> Dict:
        """Merge content dict, handling duplicate keys by concatenation."""
        merged = {}
        for key, value in content.items():
            if key in merged:
                if isinstance(merged[key], str) and isinstance(value, str):
                    merged[key] = merged[key] + " " + value
            else:
                merged[key] = value
        return merged

    async def re_analyze_element(
        self,
        copy: CopyDocument,
        element_id: str,
        focus: str = "scale"
    ) -> Dict[str, any]:
        """
        Re-analyze a specific element with focused attention.

        Called when initial analysis has low confidence on a specific element.

        Args:
            copy: The copy document
            element_id: Question ID (e.g., "Q3")
            focus: What to focus on - "scale", "answer", or "both"

        Returns:
            Dict with updated content and confidence
        """
        if not copy.page_images:
            return {"error": "No images available"}

        # Build focused prompt based on what we need
        if focus == "scale":
            prompt = f"""Regarde ATTENTIVEMENT cette copie et trouve le bareme (points) pour la question {element_id}.

Cherche:
- En haut de la page (barème général)
- A cote de la question {element_id}
- Sur le sujet de l'examen si visible

Reponds UNIQUEMENT par:
{element_id}: | [nombre de points] | [confiance: haut/moyen/bas]

Exemple: "Q3: | 2 | confiance: haut"

Si tu ne trouves pas le bareme, reponds: "{element_id}: | ? | confiance: bas"
"""
        elif focus == "answer":
            prompt = f"""Regarde ATTENTIVEMENT cette copie et resume la reponse de l'eleve pour la question {element_id}.

Reponds UNIQUEMENT par:
{element_id}: [resume precis de la reponse]

Sois precis sur ce que l'eleve a ecrit.
"""
        else:  # both
            prompt = f"""Regarde ATTENTIVEMENT cette copie pour la question {element_id}.

1. Resume precisement la reponse de l'eleve
2. Trouve le bareme (points) pour cette question

Reponds UNIQUEMENT par:
{element_id}: [resume de la reponse] | [points] | [confiance: haut/moyen/bas]
"""

        # Use the first page image for focused analysis
        result = self.ai.call_vision(prompt, image_path=copy.page_images[0])
        parsed = self._parse_content_summary(result)

        return {
            "element_id": element_id,
            "parsed": parsed,
            "raw_response": result
        }

    def get_low_confidence_elements(self, copy: CopyDocument) -> List[str]:
        """
        Get list of elements with low confidence scale detection.

        Returns:
            List of question IDs with low confidence
        """
        low_confidence = []

        for key, value in copy.content_summary.items():
            if key.endswith('_confidence'):
                if 'bas' in value.lower() or 'low' in value.lower():
                    # Extract question ID from Q1_confidence -> Q1
                    q_id = key.replace('_confidence', '')
                    low_confidence.append(q_id)

        return low_confidence

    def _parse_content_summary(self, response: str) -> Dict[str, str]:
        """
        Parse AI response into content summary dict.

        Extracts student name, answer, grading scale (points), and confidence from the response.
        Format:
            ELEVE: [name]
            Q1: answer | points | confiance: haut/moyen/bas
            Q1: answer | points (legacy format)
            Q1: answer | ? (unknown scale)
        """
        summary = {}

        for line in response.split('\n'):
            line = line.strip()

            # Skip empty lines or markdown
            if not line or line.startswith('---'):
                continue

            # Look for ELEVE: pattern (student name)
            if line.upper().startswith('ELEVE:') or line.startswith('ÉLÈVE:'):
                try:
                    colon_idx = line.index(':')
                    student_name = line[colon_idx+1:].strip()
                    if student_name and student_name.lower() != 'inconnu':
                        summary['_student_name'] = student_name
                        # Also set it on the copy if we have access to it
                except (ValueError, IndexError):
                    pass
                continue

            # Look for Q1:, Q2:, etc. pattern
            if line.startswith('Q') and ':' in line:
                try:
                    # Find colon after Q<number>
                    colon_idx = line.index(':')
                    key_part = line[:colon_idx].strip()

                    # Extract just the number from Q1, Q2, etc.
                    if key_part.startswith('Q'):
                        num_str = key_part[1:]  # Remove 'Q'
                        if num_str.isdigit():
                            key = f"Q{num_str}"
                            # Get everything after the colon as value
                            value = line[colon_idx+1:].strip()

                            # Default confidence for barème detection
                            bareme_confidence = "moyen"

                            # Check for | separators
                            if '|' in value:
                                parts = [p.strip() for p in value.split('|')]
                                value = parts[0]

                                # Process remaining parts for points and confidence
                                for part in parts[1:]:
                                    part_lower = part.lower()

                                    # Check for confidence indicator for barème detection
                                    if 'confiance' in part_lower or 'confidence' in part_lower:
                                        if 'haut' in part_lower or 'high' in part_lower:
                                            bareme_confidence = "haut"
                                        elif 'bas' in part_lower or 'low' in part_lower:
                                            bareme_confidence = "bas"
                                        else:
                                            bareme_confidence = "moyen"
                                    else:
                                        # This should be points
                                        if part == '?' or part.strip() == '?':
                                            summary[f"{key}_points_unknown"] = "true"
                                        else:
                                            # Extract numeric value from points string
                                            points_match = re.search(r'([\d.]+)', part)
                                            if points_match:
                                                points = float(points_match.group(1))
                                                summary[f"{key}_points"] = str(points)
                                                self.grading_scale[key] = points

                            # Store barème detection confidence (more explicit naming)
                            summary[f"{key}_bareme_confidence"] = bareme_confidence

                            # Store answer if not empty
                            if key and value:
                                summary[key] = value

                except (ValueError, IndexError):
                    continue

            # Fallback: any line with colon (for less structured responses)
            elif ':' in line and not line.startswith('*') and '**' not in line:
                try:
                    parts = line.split(':', 1)
                    key = parts[0].strip()
                    value = parts[1].strip()

                    # Clean key: remove markdown and spaces
                    key = key.replace('*', '').replace("'", "")
                    if "'" in key:
                        key = key.replace("'", "")

                    if key and value and len(key) < 20:  # Reasonable key length
                        summary[key] = value
                except (ValueError, IndexError):
                    continue

        return summary

    async def _analyze_phase(self):
        """Phase 1: Cross-copy analysis (OPTIONAL - skipped for speed)."""
        self.session.status = SessionStatus.ANALYZING

        # Skip cross-copy analysis for now - it's slow and not essential
        # Can be re-enabled later with a --full-analysis flag
        # self.session.class_map = await self.analyzer.analyze(self.session.copies)

        # Create a minimal class_map for grading
        from core.models import ClassAnswerMap
        self.session.class_map = ClassAnswerMap()

        self._save_sync()

        # Transition to grading
        self.session.status = SessionStatus.GRADING

    async def _calibration_phase(self):
        """Phase 3: Calibration and consistency check."""
        self.session.status = SessionStatus.CALIBRATING

        # Check for inconsistencies
        detector = ConsistencyDetector(self.session)
        reports = detector.detect_all()

        if reports:
            # Apply corrections retroactively
            applier = RetroactiveApplier(self.session, self.ai)
            await applier.apply_corrections(reports, self.store)

        self._save_sync()

    async def verify_detected_parameters(self) -> Dict[str, Any]:
        """
        Cross-verify names and barème detected during grading.

        Only called if --auto-detect-structure was NOT used.
        If structure was pre-detected, this is skipped (already verified).

        Returns:
            Dict with verification results
        """
        from rich.console import Console
        import asyncio

        console = Console()

        # Skip if structure was pre-detected (already cross-verified)
        if self._structure_pre_detected:
            console.print("[dim]Structure pré-détectée, vérification skipée[/dim]")
            return {'skipped': True, 'reason': 'pre_detected'}

        # Skip if no comparison mode
        if not self._comparison_mode:
            console.print("[dim]Mode single LLM, pas de cross-vérification[/dim]")
            return {'skipped': True, 'reason': 'single_llm'}

        results = {
            'names_verified': 0,
            'names_disagreed': 0,
            'bareme_verified': 0,
            'bareme_disagreed': 0
        }

        # TODO: Implement name cross-verification
        # This would require storing both LLMs' name detections during grading
        # in self._grading_detected_names

        # TODO: Implement barème cross-verification
        # This would require storing both LLMs' barème detections during grading
        # in self._grading_detected_bareme

        return results

    async def _export_phase(self) -> Dict[str, str]:
        """Phase 4: Export results."""
        from rich.console import Console

        console = Console()

        # Generate analytics
        analytics = AnalyticsGenerator(self.session)
        report = analytics.generate()

        # Export to session's reports directory
        reports_dir = self.store.get_reports_dir()
        reports_dir.mkdir(parents=True, exist_ok=True)

        exporter = DataExporter(self.session, str(reports_dir))
        json_path = exporter.export_json()

        # Export CSV
        csv_path = exporter.export_csv()

        # Export individual copies
        individual_paths = exporter.export_individual_copies()

        exports = {
            'json': str(json_path),
            'csv': str(csv_path),
            'individual': individual_paths
        }

        return exports

    def _save_sync(self):
        """Synchronous save of session state."""
        self.store.save_session(self.session)

        # Also save individual graded copies with their metadata
        for graded in self.session.graded_copies:
            self.store.save_graded_copy(graded, graded.copy_id)

    async def _save_session(self):
        """Save session state to storage."""
        await self.store.save_session_async(self.session)

    def get_progress(self) -> Dict[str, any]:
        """Get current progress statistics."""
        graded = len(self.session.graded_copies)
        total = len(self.session.copies)
        return {
            "session_id": self.session_id,
            "status": self.session.status,
            "copies_count": total,
            "graded_count": graded,
            "progress_percent": (graded / total * 100) if total > 0 else 0
        }

    def get_session(self) -> GradingSession:
        """Get current session object."""
        return self.session

    def export_data(self, formats: str) -> Dict[str, str]:
        """Export session data in specified formats."""
        reports_dir = self.store.get_reports_dir()
        reports_dir.mkdir(parents=True, exist_ok=True)

        exporter = DataExporter(self.session, str(reports_dir))
        exports = {}

        for fmt in formats.split(','):
            fmt = fmt.strip().lower()
            if fmt == 'json':
                exports['json'] = str(exporter.export_json())
            elif fmt == 'csv':
                exports['csv'] = str(exporter.export_csv())

        return exports
