"""
Intelligent grading engine with confidence scoring.

Grades student copies while tracking confidence and requesting
teacher help when uncertain.
"""

import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime

from core.models import (
    CopyDocument, GradedCopy, GradingPolicy, ClassAnswerMap,
    ConfidenceLevel, UncertaintyType
)
from ai import create_ai_provider
from prompts import build_grading_prompt, get_uncertainty_prompt
from config.constants import CONFIDENCE_THRESHOLD_AUTO, CONFIDENCE_THRESHOLD_FLAG


class IntelligentGrader:
    """
    The core grading engine.

    Features:
    - Confidence-based decision making
    - Cross-copy context awareness
    - Automatic flagging for review
    - Teacher request when uncertain
    """

    def __init__(
        self,
        policy: GradingPolicy,
        class_map: ClassAnswerMap = None,
        ai_provider = None,
        progress_callback: callable = None,
        reading_disagreement_callback: callable = None,
        skip_reading_consensus: bool = False
    ):
        """
        Initialize the grader.

        Args:
            policy: Current grading policy
            class_map: Cross-copy analysis results
            ai_provider: AI provider (creates default if None)
            progress_callback: Optional callback for progress updates
            reading_disagreement_callback: Optional callback for reading disagreements (enables Consensus de Lecture)
            skip_reading_consensus: If True, skip reading consensus phase (default: False)
        """
        self.policy = policy
        self.class_map = class_map
        self.ai = ai_provider or create_ai_provider()
        self.progress_callback = progress_callback
        self._reading_disagreement_callback = reading_disagreement_callback
        self._skip_reading_consensus = skip_reading_consensus

        # Thresholds from policy or defaults
        self.auto_threshold = policy.confidence_thresholds.get(
            "auto", CONFIDENCE_THRESHOLD_AUTO
        )
        self.flag_threshold = policy.confidence_thresholds.get(
            "flag", CONFIDENCE_THRESHOLD_FLAG
        )

    async def _notify_progress(self, event_type: str, data: dict):
        """Safely call progress callback."""
        if self.progress_callback is None:
            return
        try:
            result = self.progress_callback(event_type, data)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Progress callback failed: {e}")

    async def grade_copy(
        self,
        copy: CopyDocument,
        questions: Dict[str, str] = None,
        question_scales: Dict[str, float] = None
    ) -> GradedCopy:
        """
        Grade a complete student copy.

        Args:
            copy: CopyDocument to grade
            questions: Dict of {question_id: question_text}
            question_scales: Dict of {question_id: max_points} - overrides default scales

        Returns:
            GradedCopy with grades and confidence
        """
        if questions is None:
            questions = self._infer_questions(copy)

        graded = GradedCopy(
            copy_id=copy.id,
            policy_version=self.policy.version
        )

        total_score = 0.0
        max_score = 0.0
        confidences = []
        total_questions = len(questions)
        question_list = list(questions.items())

        for q_idx, (question_id, question_text) in enumerate(question_list):
            # Notify question start
            await self._notify_progress('question_start', {
                'question_id': question_id,
                'question_index': q_idx + 1,
                'total_questions': total_questions,
                'copy_id': copy.id
            })

            result = await self._grade_question(
                copy,
                question_id,
                question_text,
                question_scales
            )

            # Use provided scale - if not available, raise error to ask user
            if question_scales and question_id in question_scales:
                max_points = question_scales[question_id]
            else:
                # No scale provided - this should have been handled by asking user
                raise ValueError(
                    f"Barème manquant pour {question_id}. "
                    f"Le programme doit demander à l'utilisateur avant de noter."
                )

            graded.grades[question_id] = result.get("grade") or 0
            graded.confidence_by_question[question_id] = result.get("confidence") or 0
            graded.reasoning[question_id] = result.get("reasoning", "")
            graded.student_feedback[question_id] = result.get("student_feedback", "")

            # Store comparison data if available (dual-LLM mode)
            comp_data = result.get("comparison")
            if comp_data:
                # Build grading_audit if not already present
                if graded.grading_audit is None:
                    from audit.builder import AuditBuilder
                    from core.models import GradingAudit
                    graded.grading_audit = GradingAudit(
                        mode="dual",
                        grading_method="individual",
                        verification_mode="none",
                        providers=[],
                        questions={},
                        summary={"total_questions": 0, "agreed_initial": 0, "required_verification": 0, "required_ultimatum": 0, "final_agreement_rate": 0.0}
                    )

                # Add question result to audit
                from audit.builder import AuditBuilder
                from core.models import LLMResult, ResolutionInfo, QuestionAudit

                # Get LLM names from comparison data
                llm1_name = comp_data.get("llm1", {}).get("provider", "LLM1")
                llm2_name = comp_data.get("llm2", {}).get("provider", "LLM2")

                # Build LLM results
                llm_results = {}
                if comp_data.get("llm1"):
                    llm_results["LLM1"] = LLMResult(
                        grade=comp_data["llm1"].get("grade", 0),
                        max_points=max_points,  # Use the max_points from the question
                        reading=comp_data["llm1"].get("reading", ""),
                        reasoning=comp_data["llm1"].get("reasoning", ""),
                        feedback=comp_data["llm1"].get("student_feedback", ""),
                        confidence=comp_data["llm1"].get("confidence", 0.8)
                    )
                if comp_data.get("llm2"):
                    llm_results["LLM2"] = LLMResult(
                        grade=comp_data["llm2"].get("grade", 0),
                        max_points=max_points,  # Use the max_points from the question
                        reading=comp_data["llm2"].get("reading", ""),
                        reasoning=comp_data["llm2"].get("reasoning", ""),
                        feedback=comp_data["llm2"].get("student_feedback", ""),
                        confidence=comp_data["llm2"].get("confidence", 0.8)
                    )

                # Build resolution
                resolution = ResolutionInfo(
                    final_grade=grade,
                    final_max_points=max_points,
                    method=comp_data.get("final_method", "consensus"),
                    phases=["initial"],
                    agreement=comp_data.get("final_agreement", True)
                )

                # Add to audit
                graded.grading_audit.questions[question_id] = QuestionAudit(
                    llm_results=llm_results,
                    resolution=resolution
                )

            # Handle None grade (error case)
            grade = result.get("grade") or 0
            confidence = result.get("confidence") or 0

            total_score += grade
            max_score += max_points
            confidences.append(confidence)

            # Notify question done
            await self._notify_progress('question_done', {
                'question_id': question_id,
                'question_index': q_idx + 1,
                'total_questions': total_questions,
                'grade': grade,
                'max_points': max_points,
                'confidence': confidence,
                'agreement': comp_data.get('final_agreement', True) if comp_data else True,
                'final_method': comp_data.get('decision_path', {}).get('final_method', 'consensus') if comp_data else 'single_llm',
                'copy_id': copy.id
            })

        graded.total_score = total_score
        graded.max_score = max_score
        graded.confidence = sum(confidences) / len(confidences) if confidences else 0

        return graded

    def _infer_questions(self, copy: CopyDocument) -> Dict[str, str]:
        """Infer questions from the copy summary."""
        if hasattr(copy, "content_summary") and copy.content_summary:
            return {
                key: f"Question {key}"
                for key in copy.content_summary.keys()
                if isinstance(key, str) and key.startswith("Q")
            }
        return {"Q1": "Question 1"}

    async def _grade_question(
        self,
        copy: CopyDocument,
        question_id: str,
        question_text: str,
        question_scales: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Grade a single question.

        This implementation delegates to the configured provider and parses
        a single-question grading response.
        """
        copy_summary = ""
        if hasattr(copy, "content_summary") and copy.content_summary:
            copy_summary = copy.content_summary.get(question_id, "")

        max_points = None
        if question_scales:
            max_points = question_scales.get(question_id)

        prompt = build_grading_prompt(
            question_id=question_id,
            question_text=question_text,
            student_answer=copy_summary,
            max_points=max_points,
        )
        response = self.ai.call_text(prompt)

        from ai.response_parser import ResponseParser

        parser = ResponseParser()
        parsed = parser.parse_grading_response(response, question_id)

        if not parsed.get("confidence"):
            uncertainty_prompt = get_uncertainty_prompt(
                question_id=question_id,
                question_text=question_text,
                student_answer=copy_summary,
                proposed_grade=parsed.get("grade", 0),
                reasoning=parsed.get("reasoning", ""),
            )
            uncertainty_response = self.ai.call_text(uncertainty_prompt)
            parsed.update(parser.parse_uncertainty_response(uncertainty_response))

        parsed.setdefault("timestamp", datetime.utcnow().isoformat())
        return parsed
