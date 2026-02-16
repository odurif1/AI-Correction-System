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
from config.prompts import build_grading_prompt, get_uncertainty_prompt
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
        except Exception:
            pass

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

            graded.grades[question_id] = result["grade"]
            graded.confidence_by_question[question_id] = result["confidence"]
            graded.internal_reasoning[question_id] = result.get("internal_reasoning", "")
            graded.student_feedback[question_id] = result.get("student_feedback", "")

            # Store comparison data if available (dual-LLM mode)
            comp_data = result.get("comparison")
            if comp_data:
                if graded.llm_comparison is None:
                    graded.llm_comparison = {}
                graded.llm_comparison[question_id] = comp_data

            total_score += result["grade"]
            max_score += max_points
            confidences.append(result["confidence"])

            # Notify question done
            await self._notify_progress('question_done', {
                'question_id': question_id,
                'question_index': q_idx + 1,
                'total_questions': total_questions,
                'grade': result["grade"],
                'max_points': max_points,
                'confidence': result["confidence"],
                'agreement': comp_data.get('final_agreement', True) if comp_data else True,
                'final_method': comp_data.get('decision_path', {}).get('final_method', 'consensus') if comp_data else 'single_llm',
                'copy_id': copy.id
            })

            # Track if low confidence
            if result["confidence"] < self.flag_threshold:
                graded.uncertainty_type = UncertaintyType(
                    result.get("uncertainty_type", "other")
                )

        graded.total_score = total_score
        graded.max_score = max_score
        graded.confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Generate overall feedback using primary LLM
        await self._notify_progress('feedback_start', {
            'copy_id': copy.id,
            'student_name': copy.student_name
        })
        graded.feedback = await self._generate_overall_feedback(
            copy, graded, language=copy.language
        )
        await self._notify_progress('feedback_done', {
            'copy_id': copy.id,
            'feedback': graded.feedback
        })

        return graded

    async def _generate_overall_feedback(
        self,
        copy: CopyDocument,
        graded: 'GradedCopy',
        language: str = 'fr'
    ) -> str:
        """
        Generate overall feedback for the entire copy.

        Args:
            copy: Student's copy
            graded: Graded copy with all question results
            language: Language for feedback

        Returns:
            Overall feedback string
        """
        # Build summary of results
        results_summary = []
        for q_id in sorted(graded.grades.keys()):
            grade = graded.grades[q_id]
            # Find max points from comparison or estimate
            if graded.llm_comparison and q_id in graded.llm_comparison:
                comp = graded.llm_comparison[q_id]
                # Estimate max from the grades (rough)
                max_q = max(1.0, grade * 1.5)  # Rough estimate
            else:
                max_q = 1.0  # Default

            feedback_q = graded.student_feedback.get(q_id, "")[:100]
            results_summary.append(f"{q_id}: {grade:.1f} - {feedback_q}")

        if language == 'fr':
            prompt = f"""Rédige une appréciation générale pour cette copie d'élève (1-2 phrases maximum).

Élève: {copy.student_name or 'Anonyme'}
Note: {graded.total_score:.1f}/{graded.max_score} ({(graded.total_score/graded.max_score*100) if graded.max_score > 0 else 0:.0f}%)

Résultats par question:
{chr(10).join(results_summary)}

EXIGENCES DE STYLE:
- Ton PROFRESSIONNEL et EXIGEANT, pas enfantin
- Structure: [Bilan global] + [Point fort principal] + [Axe de progrès concret]
- Vocabulaire précis et factuel
- Éviter: "bravo", "super", "continue", "bon travail", "efforts"
- Privilégier: termes spécifiques au sujet (ex: "maîtrise des conversions", "rigueur dans le raisonnement")

EXEMPLES DE BONS STYLES:
- "Copie correcte. La notion de dilution est comprise mais les calculs de concentration manquent de rigueur."
- "Travail insuffisant. Les définitions sont imprécises et les unités ne sont pas respectées."
- "Bonne maîtrise d'ensemble. Quelques erreurs de détail sur les formules littérales."

Réponds UNIQUEMENT avec l'appréciation, sans guillemets ni formule de politesse."""
        else:
            prompt = f"""Write a professional overall comment for this student's work (1-2 sentences maximum).

Student: {copy.student_name or 'Anonymous'}
Score: {graded.total_score:.1f}/{graded.max_score} ({(graded.total_score/graded.max_score*100) if graded.max_score > 0 else 0:.0f}%)

Question results:
{chr(10).join(results_summary)}

STYLE REQUIREMENTS:
- PROFESSIONAL and DEMANDING tone, not childish
- Structure: [Overall assessment] + [Main strength] + [Concrete area for improvement]
- Precise and factual vocabulary
- Avoid: "great", "good job", "keep it up", "nice effort"
- Prefer: subject-specific terms (e.g., "mastery of conversions", "rigorous reasoning")

GOOD EXAMPLES:
- "Satisfactory work. Dilution concept is understood but concentration calculations lack rigor."
- "Insufficient work. Definitions are imprecise and units are not respected."
- "Good overall mastery. Minor errors on literal formulas."

Reply ONLY with the comment, no quotation marks or pleasantries."""

        try:
            # Get primary provider (handles ComparisonProvider too)
            if hasattr(self.ai, 'primary_provider'):
                provider = self.ai.primary_provider
            else:
                provider = self.ai

            response = provider.call_text(prompt)
            return response.strip() if response else ""
        except Exception as e:
            # Return empty on error rather than fail
            return ""

    async def _grade_question(
        self,
        copy: CopyDocument,
        question_id: str,
        question_text: str,
        question_scales: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Grade a single question.

        Args:
            copy: Student's copy
            question_id: Question identifier
            question_text: The question
            question_scales: Optional dict of question_id -> max_points

        Returns:
            Dict with grade, confidence, internal_reasoning, student_feedback
        """
        # Get max points from scales - MUST be provided
        if question_scales and question_id in question_scales:
            max_points = question_scales[question_id]
        else:
            raise ValueError(
                f"Barème manquant pour {question_id}. "
                f"L'utilisateur doit fournir le barème avant la correction."
            )

        # Get criteria from policy
        criteria = self.policy.criteria.get(question_id, "")
        if not criteria:
            criteria = f"Grade this answer fairly out of {max_points} points."

        # Get student's answer
        student_answer = copy.content_summary.get(question_id, "")

        # Get class context
        class_context = ""
        if self.class_map:
            from analysis.cross_copy import CrossCopyAnalyzer
            analyzer = CrossCopyAnalyzer(self.ai)
            class_context = analyzer.get_class_context(
                self.class_map, question_id
            )

        # Get similar answers for context
        similar_answers = self._get_similar_answers(question_id, copy.id)

        # Use vision AI to grade
        if copy.page_images:
            # Grade with vision - pass detected language and question_id
            # Pass ALL page images so AI can find the question anywhere
            result = self.ai.grade_with_vision(
                question_text=question_text,
                criteria=criteria,
                image_path=copy.page_images,  # ALL pages for this student
                max_points=max_points,
                class_context=class_context,
                language=copy.language,  # Pass detected language
                question_id=question_id,  # Pass for jurisprudence lookup
                reading_disagreement_callback=self._reading_disagreement_callback,
                skip_reading_consensus=self._skip_reading_consensus
            )
            # Handle async providers (ComparisonProvider)
            import asyncio
            if asyncio.iscoroutine(result):
                result = await result
        else:
            # Grade with text prompt - pass detected language
            prompt = build_grading_prompt(
                question_text=question_text,
                criteria=criteria,
                student_answer=student_answer,
                similar_answers=similar_answers,
                max_points=max_points,
                class_context=class_context,
                language=copy.language  # Pass detected language
            )

            response = self.ai.call_text(prompt)
            result = self.ai._parse_grading_response(response)

        # Calculate final confidence
        confidence = self._calculate_confidence(result, copy, question_id)

        return {
            "grade": result.get("grade", 0),
            "confidence": confidence,
            "internal_reasoning": result.get("internal_reasoning", ""),
            "student_feedback": result.get("student_feedback", ""),
            "uncertainty_type": result.get("uncertainty_type", "none"),
            "comparison": result.get("comparison")  # Dual-LLM comparison data
        }

    def _calculate_confidence(
        self,
        ai_result: Dict[str, Any],
        copy: CopyDocument,
        question_id: str
    ) -> float:
        """
        Calculate final confidence score.

        Factors:
        - Base confidence from AI
        - Known cluster (+)
        - Similar graded answers (+)
        - Outlier status (-)
        """
        base = ai_result.get("confidence", 0.5)

        # Known cluster increases confidence
        if copy.cluster_id is not None:
            base += 0.1

        # Similar already-graded answers increase confidence
        similar = self._find_similar_graded(copy, question_id)
        if similar:
            base += 0.15

        # Outlier decreases confidence
        if self.class_map:
            for outlier in self.class_map.outliers:
                if outlier.copy_id == copy.id and outlier.question_id == question_id:
                    base -= 0.2
                    break

        return min(max(base, 0.0), 1.0)

    def _get_similar_answers(
        self,
        question_id: str,
        copy_id: str
    ) -> List[Dict[str, Any]]:
        """Get similar answers for context."""
        if not self.class_map:
            return []

        from analysis.cross_copy import CrossCopyAnalyzer
        analyzer = CrossCopyAnalyzer(self.ai)
        return analyzer.get_similar_answers(self.class_map, question_id, copy_id)

    def _find_similar_graded(
        self,
        copy: CopyDocument,
        question_id: str
    ) -> Optional[Dict[str, Any]]:
        """Find similar already-graded answers."""
        # This would check if similar answers have been graded consistently
        # For now, return None
        return None

    def _get_max_points(self, question_id: str) -> float:
        """Get max points for a question."""
        return self.policy.question_weights.get(
            question_id,
            5.0  # Default
        )

    def _infer_questions(self, copy: CopyDocument) -> Dict[str, str]:
        """Infer questions from copy content."""
        questions = {}
        for key in copy.content_summary.keys():
            # Skip all metadata keys:
            # - Keys starting with _ (like _student_name)
            # - Keys ending with _points, _points_unknown, _confidence
            if key.startswith('_'):
                continue
            if key.endswith('_points') or key.endswith('_points_unknown') or key.endswith('_confidence'):
                continue
            # Only accept valid question keys: Q1, Q2, Q3, etc.
            if not (key.startswith('Q') and key[1:].isdigit()):
                continue
            questions[key] = f"Question {key}"
        return questions

    def needs_teacher_review(self, graded: GradedCopy) -> bool:
        """
        Check if a graded copy needs teacher review.

        Args:
            graded: GradedCopy to check

        Returns:
            True if review needed
        """
        return graded.confidence < self.flag_threshold

    def needs_review_flag(self, graded: GradedCopy) -> bool:
        """
        Check if a graded copy should be flagged (medium confidence).

        Args:
            graded: GradedCopy to check

        Returns:
            True if flag needed
        """
        return (self.flag_threshold <= graded.confidence < self.auto_threshold
                and not graded.teacher_reviewed)

    def get_confidence_level(self, graded: GradedCopy) -> ConfidenceLevel:
        """
        Get the confidence level category.

        Args:
            graded: GradedCopy

        Returns:
            ConfidenceLevel enum
        """
        if graded.confidence >= self.auto_threshold:
            return ConfidenceLevel.HIGH
        elif graded.confidence >= self.flag_threshold:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def get_uncertainty_message(self, graded: GradedCopy) -> str:
        """
        Get appropriate message for uncertainty.

        Args:
            graded: GradedCopy

        Returns:
            Uncertainty message
        """
        return get_uncertainty_prompt(graded.uncertainty_type.value)


class BatchGrader:
    """
    Grades multiple copies in parallel.

    Manages the grading workflow for all copies in a session.
    """

    def __init__(
        self,
        policy: GradingPolicy,
        class_map: ClassAnswerMap,
        ai_provider = None
    ):
        """Initialize batch grader."""
        self.policy = policy
        self.class_map = class_map
        self.ai = ai_provider or create_ai_provider()
        self.grader = IntelligentGrader(policy, class_map, ai_provider)

    async def grade_all(
        self,
        copies: List[CopyDocument],
        questions: Dict[str, str] = None
    ) -> List[GradedCopy]:
        """
        Grade all copies.

        Args:
            copies: List of copies to grade
            questions: Question definitions

        Returns:
            List of GradedCopy objects
        """
        results = []

        for copy in copies:
            graded = await self.grader.grade_copy(copy, questions)
            results.append(graded)

        return results

    def get_progress(self, total: int, completed: int) -> Dict[str, any]:
        """Get progress statistics."""
        return {
            "total": total,
            "completed": completed,
            "remaining": total - completed,
            "progress_percent": (completed / total * 100) if total > 0 else 0
        }
