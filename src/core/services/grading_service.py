import asyncio
from typing import List, Dict, Tuple
from core.models import CopyDocument, GradedCopy
from core.services.graders.base import GradingContext
from export.analytics import AnalyticsGenerator


class GradingService:
    """Thin dispatcher that delegates grading to strategy classes."""

    def __init__(self, session, store, ai, grading_mode, comparison_mode, second_reading, parallel, detected_questions, grading_scale, analysis_complete, pages_per_copy=None, orchestrator=None):
        self.session = session
        self.store = store
        self.ai = ai
        self._grading_mode = grading_mode
        self._comparison_mode = comparison_mode
        self._second_reading = second_reading
        self._parallel = parallel
        self.detected_questions = detected_questions
        self.grading_scale = grading_scale
        self._analysis_complete = analysis_complete
        self._pages_per_copy = pages_per_copy
        self._scale_confirmed_by_user = True
        self._grading_complete = False
        self._orchestrator = orchestrator

    def __getattr__(self, name):
        """Delegate missing attributes to the orchestrator."""
        if name.startswith('_') and self._orchestrator is not None:
            return getattr(self._orchestrator, name)
        raise AttributeError(f"'GradingService' object has no attribute '{name}'")

    def _save_sync(self, last_graded=None):
        """Save session state to storage."""
        self.store.save_session(self.session)
        if last_graded:
            self.store.save_graded_copy(last_graded, last_graded.copy_id)
        else:
            for graded in self.session.graded_copies:
                self.store.save_graded_copy(graded, graded.copy_id)

    def _build_grading_context(self) -> GradingContext:
        """Build the context object for grader strategies."""
        return GradingContext(
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
            orchestrator=self._orchestrator,
        )

    async def grade_all(
        self,
        progress_callback: callable = None
    ) -> List[GradedCopy]:
        """Grade all copies by delegating to the appropriate strategy."""
        if not self._analysis_complete:
            raise RuntimeError("Must call analyze_only() before grade_all()")

        ctx = self._build_grading_context()

        if self._grading_mode == "batch":
            from core.services.graders.batch_grader import BatchGrader
            grader = BatchGrader(ctx)
        elif self._grading_mode == "hybrid":
            if not self._comparison_mode:
                raise RuntimeError("Hybrid mode requires dual LLM (remove --single)")
            from core.services.graders.hybrid_grader import HybridGrader
            grader = HybridGrader(ctx)
        elif self._comparison_mode:
            from core.services.graders.dual_llm_grader import DualLLMGrader
            grader = DualLLMGrader(ctx)
        else:
            from core.services.graders.single_llm_grader import SingleLLMGrader
            grader = SingleLLMGrader(ctx)

        result = await grader.grade_all(progress_callback)
        self._grading_complete = True
        return result

    # ==================== UTILITY METHODS ====================

    async def _call_callback(self, callback, event_type, data):
        """Safely call an async or sync callback."""
        if callback is None:
            return
        try:
            result = callback(event_type, data)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            import logging
            logging.warning(f"Callback error for {event_type}: {e}")

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
        return self._orchestrator.export_data('json,csv') if self._orchestrator else {}

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
