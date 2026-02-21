"""
Correction workflow orchestration.

Encapsulates the multi-phase correction workflow with clear separation
of concerns and explicit state management.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple

from core.workflow_state import CorrectionState, WorkflowPhase
from core.models import GradedCopy, CopyDocument
from core.exceptions import GradingError


@dataclass
class WorkflowCallbacks:
    """Callbacks for workflow events."""
    on_disagreement: Optional[Callable] = None
    on_name_disagreement: Optional[Callable] = None
    on_reading_disagreement: Optional[Callable] = None
    on_progress: Optional[Callable] = None
    on_phase_change: Optional[Callable] = None


@dataclass
class WorkflowConfig:
    """Configuration for correction workflow."""
    auto_mode: bool = False
    language: str = "fr"
    pages_per_copy: int = 2
    parallel_copies: int = 6
    dual_llm_mode: bool = False
    second_reading: bool = False
    skip_reading_consensus: bool = False


class CorrectionWorkflow:
    """
    Orchestrates the multi-phase correction workflow.

    Phases:
    1. INITIALIZATION: Setup and validation
    2. PDF_LOADING: Load and parse PDF files
    3. ANALYSIS: Analyze document structure
    4. SCALE_DETECTION: Detect grading scale
    5. GRADING: Grade student copies
    6. VERIFICATION: Resolve disagreements
    7. CALIBRATION: Apply retroactive changes
    8. EXPORT: Generate reports
    9. COMPLETE: Final cleanup
    """

    def __init__(
        self,
        orchestrator,  # GradingSessionOrchestrator
        config: WorkflowConfig = None,
        callbacks: WorkflowCallbacks = None
    ):
        """
        Initialize workflow.

        Args:
            orchestrator: GradingSessionOrchestrator instance
            config: Workflow configuration
            callbacks: Event callbacks
        """
        self.orchestrator = orchestrator
        self.config = config or WorkflowConfig()
        self.callbacks = callbacks or WorkflowCallbacks()

        # Initialize state
        self.state = CorrectionState(
            language=self.config.language,
            auto_mode=self.config.auto_mode,
            phase=WorkflowPhase.INITIALIZATION
        )

        # Jurisprudence tracking (can be updated during workflow)
        self._jurisprudence: Dict[str, Dict[str, Any]] = {}

    def _set_phase(self, phase: WorkflowPhase) -> None:
        """Update workflow phase and notify."""
        self.state = self.state.with_phase(phase)
        if self.callbacks.on_phase_change:
            self.callbacks.on_phase_change(phase, self.state)

    async def _notify_progress(self, event_type: str, data: Dict[str, Any]) -> None:
        """Notify progress callback."""
        if self.callbacks.on_progress:
            result = self.callbacks.on_progress(event_type, data)
            if asyncio.iscoroutine(result):
                await result

    def get_jurisprudence(self) -> Dict[str, Dict[str, Any]]:
        """Get current jurisprudence."""
        return self._jurisprudence.copy()

    def add_jurisprudence(
        self,
        question_id: str,
        decision: float,
        reasoning: str = "",
        llm1_grade: float = None,
        llm2_grade: float = None,
        max_points: float = None,
        auto: bool = False
    ) -> None:
        """Add jurisprudence entry."""
        self._jurisprudence[question_id] = {
            'decision': decision,
            'reasoning': reasoning,
            'llm1_grade': llm1_grade,
            'llm2_grade': llm2_grade,
            'max_points': max_points,
            'auto': auto
        }

        # Update orchestrator's AI provider if available
        if hasattr(self.orchestrator.ai, 'set_jurisprudence'):
            self.orchestrator.ai.set_jurisprudence(self._jurisprudence)

        # Update state
        self.state = self.state.with_jurisprudence(
            question_id, decision, reasoning, llm1_grade, llm2_grade
        )

    async def handle_disagreement(
        self,
        question_id: str,
        question_text: str,
        llm1_name: str,
        llm1_result: Dict[str, Any],
        llm2_name: str,
        llm2_result: Dict[str, Any],
        max_points: float
    ) -> Tuple[float, str]:
        """
        Handle a grading disagreement between LLMs.

        Args:
            question_id: Question identifier
            question_text: Question text
            llm1_name: Name of first LLM
            llm1_result: Result from first LLM
            llm2_name: Name of second LLM
            llm2_result: Result from second LLM
            max_points: Maximum points for question

        Returns:
            Tuple of (chosen_grade, feedback_source)
        """
        grade1 = llm1_result.get('grade', 0) or 0
        grade2 = llm2_result.get('grade', 0) or 0
        average_grade = (grade1 + grade2) / 2

        # Check jurisprudence
        if question_id in self._jurisprudence:
            past = self._jurisprudence[question_id]
            await self._notify_progress('jurisprudence_applied', {
                'question_id': question_id,
                'past_decision': past['decision']
            })

        # Auto mode: use average
        if self.config.auto_mode:
            self.add_jurisprudence(
                question_id=question_id,
                decision=average_grade,
                llm1_grade=grade1,
                llm2_grade=grade2,
                max_points=max_points,
                auto=True,
                reasoning=f"Auto: average of {grade1} and {grade2}"
            )
            return average_grade, "merge"

        # Interactive mode: use callback
        if self.callbacks.on_disagreement:
            try:
                result = self.callbacks.on_disagreement(
                    question_id=question_id,
                    question_text=question_text,
                    llm1_name=llm1_name,
                    llm1_result=llm1_result,
                    llm2_name=llm2_name,
                    llm2_result=llm2_result,
                    max_points=max_points
                )
                if asyncio.iscoroutine(result):
                    chosen_grade, feedback_source = await result
                else:
                    chosen_grade, feedback_source = result

                self.add_jurisprudence(
                    question_id=question_id,
                    decision=chosen_grade,
                    llm1_grade=grade1,
                    llm2_grade=grade2,
                    max_points=max_points,
                    auto=False,
                    reasoning=f"User choice from {llm1_name}/{llm2_name}"
                )
                return chosen_grade, feedback_source

            except (EOFError, KeyboardInterrupt):
                # Fallback to average
                self.add_jurisprudence(
                    question_id=question_id,
                    decision=average_grade,
                    llm1_grade=grade1,
                    llm2_grade=grade2,
                    max_points=max_points,
                    auto=True,
                    reasoning="Fallback: no interactive input"
                )
                return average_grade, "merge"

        # No callback: use average
        return average_grade, "merge"

    async def handle_name_disagreement(
        self,
        llm1_result: Dict[str, Any],
        llm2_result: Dict[str, Any]
    ) -> str:
        """
        Handle a name detection disagreement.

        Args:
            llm1_result: Name result from first LLM
            llm2_result: Name result from second LLM

        Returns:
            Chosen name
        """
        # Auto mode: use higher confidence
        if self.config.auto_mode:
            if llm1_result.get('confidence', 0) >= llm2_result.get('confidence', 0):
                return llm1_result.get('name') or "Inconnu"
            return llm2_result.get('name') or "Inconnu"

        # Interactive mode: use callback
        if self.callbacks.on_name_disagreement:
            result = self.callbacks.on_name_disagreement(llm1_result, llm2_result)
            if asyncio.iscoroutine(result):
                return await result
            return result

        # Fallback: use first available
        return llm1_result.get('name') or llm2_result.get('name') or "Inconnu"

    async def handle_reading_disagreement(
        self,
        llm1_result: Dict[str, Any],
        llm2_result: Dict[str, Any],
        question_text: str,
        image_path
    ) -> str:
        """
        Handle a reading disagreement between LLMs.

        Args:
            llm1_result: Reading result from first LLM
            llm2_result: Reading result from second LLM
            question_text: Question being read
            image_path: Path to image being read

        Returns:
            Chosen reading
        """
        # Auto mode: use higher confidence
        if self.config.auto_mode:
            if llm1_result.get('confidence', 0) >= llm2_result.get('confidence', 0):
                return llm1_result.get('reading', '')
            return llm2_result.get('reading', '')

        # Interactive mode: use callback
        if self.callbacks.on_reading_disagreement:
            result = self.callbacks.on_reading_disagreement(
                llm1_result, llm2_result, question_text, image_path
            )
            if asyncio.iscoroutine(result):
                return await result
            return result

        # Fallback: use higher confidence
        if llm1_result.get('confidence', 0) >= llm2_result.get('confidence', 0):
            return llm1_result.get('reading', '')
        return llm2_result.get('reading', '')

    async def run(
        self,
        pdf_paths: List[str],
        output_dir: str = None
    ) -> Dict[str, Any]:
        """
        Run the complete correction workflow.

        Args:
            pdf_paths: List of PDF file paths
            output_dir: Output directory for results

        Returns:
            Workflow results with grades and audit trail
        """
        self._set_phase(WorkflowPhase.PDF_LOADING)

        try:
            # Phase 1: Load PDFs
            await self._notify_progress('phase_start', {
                'phase': 'loading',
                'num_files': len(pdf_paths)
            })

            await self.orchestrator.load_pdfs(pdf_paths)

            # Phase 2: Analyze
            self._set_phase(WorkflowPhase.ANALYSIS)
            await self._notify_progress('phase_start', {
                'phase': 'analysis'
            })

            await self.orchestrator.analyze_only()

            # Phase 3: Scale detection
            self._set_phase(WorkflowPhase.SCALE_DETECTION)
            detected_scale = self.orchestrator.get_detected_scale()

            # Phase 4: Grading
            self._set_phase(WorkflowPhase.GRADING)
            await self._notify_progress('phase_start', {
                'phase': 'grading',
                'num_copies': len(self.orchestrator.session.copies)
            })

            graded_copies = await self.orchestrator.grade_all(
                progress_callback=self._create_progress_callback()
            )

            # Phase 5: Export (if output_dir specified)
            if output_dir:
                self._set_phase(WorkflowPhase.EXPORT)
                await self._notify_progress('phase_start', {
                    'phase': 'export',
                    'output_dir': output_dir
                })

            # Complete
            self._set_phase(WorkflowPhase.COMPLETE)

            return {
                'success': True,
                'session_id': self.orchestrator.session.session_id,
                'graded_copies': len(graded_copies),
                'jurisprudence': self._jurisprudence,
                'state': self.state.to_dict()
            }

        except Exception as e:
            self._set_phase(WorkflowPhase.ERROR)
            self.state = self.state.with_error(e, {
                'phase': self.state.phase.value
            })
            raise

    def _create_progress_callback(self) -> Callable:
        """Create progress callback for orchestrator."""
        async def callback(event_type: str, data: Dict[str, Any]) -> None:
            # Update state progress
            if 'copy_index' in data:
                self.state = self.state.with_progress(
                    processed=data['copy_index'],
                    total=data.get('total_copies', self.state.total_copies)
                )

            # Forward to external callback
            await self._notify_progress(event_type, data)

        return callback

    @property
    def session(self):
        """Get current session."""
        return self.orchestrator.session
