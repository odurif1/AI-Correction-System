"""
Feedback generation for students.

Creates personalized, constructive feedback based on grading results.
"""

from typing import Dict, List, Optional
from datetime import datetime

from core.models import GradedCopy, GradingSession, AnalyticsReport
from ai.openai_provider import OpenAIProvider


class FeedbackGenerator:
    """
    Generates personalized feedback for students.

    Features:
    - Question-specific feedback
    - Overall performance summary
    - Comparison to class average
    - Actionable improvement suggestions
    """

    def __init__(self, ai_provider: OpenAIProvider = None):
        """
        Initialize feedback generator.

        Args:
            ai_provider: AI provider (creates default if None)
        """
        self.ai = ai_provider or OpenAIProvider()

    def generate_for_copy(
        self,
        graded: GradedCopy,
        session: GradingSession,
        student_name: str = "Student"
    ) -> str:
        """
        Generate feedback for a graded copy.

        Args:
            graded: GradedCopy
            session: Grading session with class context
            student_name: Student's name

        Returns:
            Feedback text
        """
        # Get class performance
        class_avg = self._compute_class_average(session)

        # Build graded copy dict for prompt
        graded_data = {
            "total_score": graded.total_score,
            "max_score": graded.max_score,
            "grades": [],
            "feedbacks": []
        }

        for q_id, points in graded.grades.items():
            max_points = session.policy.question_weights.get(q_id, 5.0)
            graded_data["grades"].append({
                "question": q_id,
                "points": points,
                "max": max_points,
                "feedback": graded.student_feedback.get(q_id, "")
            })

        class_perf = {
            "average": class_avg
        }

        # Use AI to generate feedback
        feedback = self.ai.generate_feedback(
            student_name=student_name,
            graded_copy=graded_data,
            class_performance=class_perf
        )

        return feedback

    def generate_for_session(
        self,
        session: GradingSession
    ) -> Dict[str, str]:
        """
        Generate feedback for all copies in a session.

        Args:
            session: Grading session

        Returns:
            Dict of {copy_id: feedback}
        """
        feedback_map = {}

        for graded in session.graded_copies:
            copy = next(
                (c for c in session.copies if c.id == graded.copy_id),
                None
            )

            if copy:
                student_name = copy.student_name or "Student"
                feedback = self.generate_for_copy(graded, session, student_name)
                feedback_map[graded.copy_id] = feedback

        return feedback_map

    def generate_question_feedback(
        self,
        graded: GradedCopy,
        question_id: str
    ) -> str:
        """
        Generate feedback for a specific question.

        Args:
            graded: GradedCopy
            question_id: Question identifier

        Returns:
            Question-specific feedback
        """
        if question_id not in graded.student_feedback:
            return ""

        student_fb = graded.student_feedback[question_id]
        grade = graded.grades.get(question_id, 0)

        feedback = f"Question {question_id}: {grade}/5 points\n"
        feedback += f"{student_fb}\n"

        return feedback

    def generate_improvement_suggestions(
        self,
        graded: GradedCopy,
        session: GradingSession
    ) -> List[str]:
        """
        Generate specific improvement suggestions.

        Args:
            graded: GradedCopy
            session: Grading session

        Returns:
            List of suggestions
        """
        suggestions = []

        # Find weak areas (below 70% of max)
        for q_id, grade in graded.grades.items():
            max_points = session.policy.question_weights.get(q_id, 5.0)
            percentage = grade / max_points if max_points > 0 else 0

            if percentage < 0.7:
                suggestions.append(
                    f"Question {q_id}: Review the concepts - "
                    f"scored {grade}/{max_points}"
                )

        return suggestions

    def generate_strengths_highlights(
        self,
        graded: GradedCopy,
        session: GradingSession
    ) -> List[str]:
        """
        Generate highlights of student's strengths.

        Args:
            graded: GradedCopy
            session: Grading session

        Returns:
            List of strength highlights
        """
        strengths = []

        # Find strong areas (above 85% of max)
        for q_id, grade in graded.grades.items():
            max_points = session.policy.question_weights.get(q_id, 5.0)
            percentage = grade / max_points if max_points > 0 else 0

            if percentage >= 0.85:
                strengths.append(
                    f"Question {q_id}: Excellent work ({grade}/{max_points})"
                )

        return strengths

    def _compute_class_average(self, session: GradingSession) -> float:
        """Compute class average score."""
        if not session.graded_copies:
            return 0.0

        total = sum(g.total_score for g in session.graded_copies)
        return total / len(session.graded_copies)


class ClassFeedbackSummary:
    """
    Generates class-level feedback summaries for teachers.
    """

    def __init__(self, session: GradingSession):
        """Initialize with grading session."""
        self.session = session

    def generate_common_mistakes(self) -> List[str]:
        """
        Identify common mistakes across the class.

        Returns:
            List of common mistake descriptions
        """
        mistakes = []

        if self.session.class_map:
            for q_id, analysis in self.session.class_map.question_analyses.items():
                if analysis.common_errors:
                    for error in analysis.common_errors[:2]:
                        mistakes.append(f"Q{q_id}: {error}")

        return mistakes

    def generate_difficulty_report(self) -> Dict[str, str]:
        """
        Generate report on question difficulty.

        Returns:
            Dict of {question_id: difficulty_level}
        """
        report = {}

        if self.session.class_map:
            for q_id, analysis in self.session.class_map.question_analyses.items():
                difficulty = analysis.difficulty_estimate

                if difficulty < 0.3:
                    level = "Easy"
                elif difficulty < 0.6:
                    level = "Medium"
                else:
                    level = "Difficult"

                report[q_id] = level

        return report

    def generate_outstanding_answers(self) -> List[str]:
        """
        Identify outstanding answers worth highlighting.

        Returns:
            List of descriptions
        """
        outstanding = []

        for graded in self.session.graded_copies:
            if graded.confidence >= 0.9 and graded.total_score >= graded.max_score * 0.9:
                copy = next(
                    (c for c in self.session.copies if c.id == graded.copy_id),
                    None
                )
                if copy:
                    name = copy.student_name or "Anonymous"
                    outstanding.append(
                        f"{name}: {graded.total_score}/{graded.max_score}"
                    )

        return outstanding

    def generate_teacher_summary(self) -> str:
        """
        Generate comprehensive summary for the teacher.

        Returns:
            Summary text
        """
        lines = [
            f"Class Summary - {self.session.session_id}",
            f"Total copies: {len(self.session.graded_copies)}",
            ""
        ]

        # Average score
        if self.session.graded_copies:
            avg = sum(g.total_score for g in self.session.graded_copies) / len(self.session.graded_copies)
            lines.append(f"Class average: {avg:.1f}")

        # Common mistakes
        mistakes = self.generate_common_mistakes()
        if mistakes:
            lines.append("\nCommon mistakes:")
            for mistake in mistakes[:5]:
                lines.append(f"  - {mistake}")

        # Difficulty report
        difficulty = self.generate_difficulty_report()
        if difficulty:
            lines.append("\nQuestion difficulty:")
            for q_id, level in difficulty.items():
                lines.append(f"  {q_id}: {level}")

        return "\n".join(lines)
