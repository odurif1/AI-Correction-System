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
from audit.builder import build_audit_from_llm_comparison
from config.settings import get_settings
from analysis.cross_copy import CrossCopyAnalyzer
from core.grading.grader import IntelligentGrader
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
        user_id: str = None,
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
        use_chat_continuation: bool = False,  # Enable chat continuation for verification/ultimatum
        workflow_state: CorrectionState = None
    ):
        """
        Initialize the orchestrator.

        Args:
            pdf_paths: List of paths to student PDF copies (optional if resuming)
            session_id: Session ID to resume (optional)
            user_id: User ID for multi-tenant storage (optional)
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
            use_chat_continuation: If True, use chat sessions for verification/ultimatum (LLM remembers context)
            workflow_state: Optional CorrectionState for tracking workflow state
        """
        from config.constants import DATA_DIR

        self.pdf_paths = pdf_paths or []
        self.base_dir = Path(DATA_DIR)
        self.user_id = user_id
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
        self._use_chat_continuation = use_chat_continuation
        self._structure_pre_detected = False  # Set to True after auto-detect-structure phase

        # Store names and barème detected during grading (for cross-verification)
        self._grading_detected_names = {}  # {copy_id: {'llm1': name, 'llm2': name}}
        self._grading_detected_bareme = {}  # {question_id: {'llm1': points, 'llm2': points}}

        # Workflow state tracking
        self._workflow_state = workflow_state

        # Initialize session
        # Use default user_id for CLI usage
        effective_user_id = user_id or "cli_user"

        if session_id:
            self.session_id = session_id
            self.store = SessionStore(session_id, user_id=effective_user_id)
            self.session = self.store.load_session()
            if not self.session:
                raise ValueError(f"Session {session_id} not found")
        else:
            self.session_id = generate_id()
            self.session = GradingSession(
                session_id=self.session_id,
                created_at=datetime.now(),
                status=SessionStatus.DIAGNOSTIC,
                user_id=effective_user_id,
                pages_per_copy=pages_per_copy
            )
            self.store = SessionStore(self.session_id, user_id=effective_user_id)

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
        # grading_scale is now an alias to session.policy.question_weights
        # Use self.session.policy.question_weights directly as the single source of truth

        # Store detected questions (text content)
        self.detected_questions: Dict[str, str] = {}

        # Track if scale was confirmed by user
        self._scale_confirmed_by_user = False

        # Analysis results (available after analyze_only)
        self._analysis_complete = False
        self._grading_complete = False

        # Restore _analysis_complete if session was already analyzed
        # (session has copies and they have content, or grading_scale is set)
        if self.session.copies and any(c.content_summary for c in self.session.copies):
            self._analysis_complete = True
        elif self.grading_scale:  # If grading_scale is already set, analysis was done
            self._analysis_complete = True

        # Callback to ask user for scale (set by main.py)
        self.scale_callback: Optional[callable] = None

    @property
    def grading_scale(self) -> Dict[str, float]:
        """Alias to session.policy.question_weights (single source of truth)."""
        return self.session.policy.question_weights

    @grading_scale.setter
    def grading_scale(self, value: Dict[str, float]):
        """Set question_weights via grading_scale alias."""
        self.session.policy.question_weights = value

    @property
    def workflow_state(self) -> CorrectionState:
        """Get the workflow state, creating a default if not set."""
        if self._workflow_state is None:
            self._workflow_state = CorrectionState(
                session_id=self.session_id,
                phase=WorkflowPhase.DETECTION
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
        await self.analyze_only()

        # Phase 3: Intelligent grading
        # await self._grading_phase() # Removed, usually called explicitly or via grade_all()

        # Phase 4: Calibration
        await self._calibration_phase()

        # Phase 5: Export
        export_path = await self._export_phase()

        # Mark complete
        self.session.transition_to(SessionStatus.COMPLETE)
        self._save_sync()

        return self.session_id

    # ==================== PHASED WORKFLOW METHODS ====================

    async def analyze_only(self) -> Dict:
        """
        Phase 1: Analyze copies without grading.

        Loads PDFs, extracts content, detects questions.
        Delegated to DetectionService.

        Returns:
            Dict with detection metadata
        """
        from core.services.detection_service import DetectionService
        
        detector = DetectionService(
            session=self.session,
            store=self.store,
            ai=self.ai,
            pdf_paths=self.pdf_paths,
            pages_per_copy=self._pages_per_copy,
            grading_mode=self._grading_mode,
            comparison_mode=self._comparison_mode
        )
        
        result = await detector.analyze_only()
        
        # grading_scale is an alias to session.policy.question_weights
        self.grading_scale = result['scale']
        self.detected_questions = result['questions']
        self.scale_detected = result['scale_detected']
        self._analysis_complete = True
        self._structure_pre_detected = result.get('structure_pre_detected', False)
        
        self._save_sync()
        return result

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
                        # grading_scale is an alias to session.policy.question_weights
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
                    # grading_scale is an alias to session.policy.question_weights
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
        from core.services.grading_service import GradingService
        grader = GradingService(
            session=self.session,
            store=self.store,
            ai=self.ai,
            grading_mode=self._grading_mode,
            comparison_mode=self._comparison_mode,
            second_reading=self._second_reading,
            parallel=self._parallel,
            detected_questions=self.detected_questions,
            grading_scale=self.grading_scale,
            analysis_complete=self._analysis_complete,
            pages_per_copy=self._pages_per_copy,
            orchestrator=self
        )
        return await grader.grade_all(progress_callback)




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
    async def _calibration_phase(self):
        """Phase 3: Calibration and consistency check."""
        self.session.transition_to(SessionStatus.CORRECTION)

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

    async def export(self) -> Dict[str, str]:
        """Export session results (json + csv)."""
        return self.export_data('json,csv')

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
