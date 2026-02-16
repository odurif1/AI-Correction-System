"""
Cross-copy analysis for intelligent grading.

Analyzes all copies together to identify patterns, ensure consistency,
and provide context for individual grading decisions.
"""

import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime

from core.models import (
    CopyDocument, GradingSession, ClassAnswerMap,
    QuestionAnalysis, AnswerOutlier
)
from ai.openai_provider import OpenAIProvider
from analysis.clustering import EmbeddingClustering


class CrossCopyAnalyzer:
    """
    Analyzes student answers across all copies.

    This is the "Phase 1" component that:
    1. Scans all copies with vision AI
    2. Identifies question structure
    3. Clusters similar answers
    4. Detects patterns and outliers
    5. Provides context for grading
    """

    def __init__(self, ai_provider: OpenAIProvider = None):
        """
        Initialize the analyzer.

        Args:
            ai_provider: OpenAI provider (creates default if None)
        """
        self.ai = ai_provider or OpenAIProvider()
        self.clustering = EmbeddingClustering()

    async def analyze(
        self,
        copies: List[CopyDocument],
        questions: Dict[str, str] = None
    ) -> ClassAnswerMap:
        """
        Perform complete cross-copy analysis.

        Args:
            copies: All student copies to analyze
            questions: Dict of {question_id: question_text}

        Returns:
            ClassAnswerMap with all analysis results
        """
        result = ClassAnswerMap()

        # If no questions provided, detect from copies
        if questions is None:
            questions = self._detect_questions(copies)

        # Analyze each question
        for question_id, question_text in questions.items():
            analysis = await self._analyze_question(
                copies,
                question_id,
                question_text
            )
            result.question_analyses[question_id] = analysis

            # Cluster answers
            clusters = self.clustering.cluster_by_question(copies, question_id)
            result.clusters[question_id] = [c.model_dump() for c in clusters]

        # Detect outliers across all questions
        result.outliers = self._detect_outliers(copies, result)

        result.analyzed_at = datetime.now()

        return result

    def _detect_questions(
        self,
        copies: List[CopyDocument]
    ) -> Dict[str, str]:
        """
        Detect question structure from copies.

        Analyzes the first few copies to identify questions.

        Args:
            copies: List of copy documents

        Returns:
            Dict of {question_id: question_text}
        """
        # For now, use a simple approach
        # In production, this would use vision AI to detect structure

        questions = {}
        sample_copy = copies[0] if copies else None

        if sample_copy and sample_copy.content_summary:
            # Extract question IDs from content summary keys
            for key in sample_copy.content_summary.keys():
                questions[key] = f"Question {key}"

        return questions

    async def _analyze_question(
        self,
        copies: List[CopyDocument],
        question_id: str,
        question_text: str
    ) -> QuestionAnalysis:
        """
        Analyze answers for a specific question.

        Args:
            copies: All copies
            question_id: Question to analyze
            question_text: The question text

        Returns:
            QuestionAnalysis object
        """
        # Collect answers
        answer_summaries = []
        for copy in copies:
            if question_id in copy.content_summary:
                answer_summaries.append((copy.id, copy.content_summary[question_id]))

        if not answer_summaries:
            return QuestionAnalysis(
                question_id=question_id,
                question_text=question_text
            )

        # Use AI to analyze patterns
        max_points = 5.0  # Default, could be configured
        analysis_result = self.ai.analyze_cross_copy(
            question_id=question_id,
            question_text=question_text,
            answer_summaries=answer_summaries,
            max_points=max_points
        )

        return QuestionAnalysis(
            question_id=question_id,
            question_text=question_text,
            common_correct=analysis_result.get("common_correct", []),
            common_errors=analysis_result.get("common_errors", []),
            unique_approaches=analysis_result.get("unique_approaches", []),
            difficulty_estimate=analysis_result.get("difficulty_estimate", 0.5),
            answer_distribution=self._compute_distribution(answer_summaries)
        )

    def _compute_distribution(
        self,
        answer_summaries: List[tuple[str, str]]
    ) -> Dict[str, int]:
        """
        Compute answer distribution for simple counting.

        Args:
            answer_summaries: List of (copy_id, answer) tuples

        Returns:
            Distribution dict
        """
        # For simple answers, count exact matches
        # In production, this would be more sophisticated
        counts = {}
        for _, answer in answer_summaries:
            key = answer[:50]  # Truncate for key
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _detect_outliers(
        self,
        copies: List[CopyDocument],
        class_map: ClassAnswerMap
    ) -> List[AnswerOutlier]:
        """
        Detect outlier answers across all questions.

        Args:
            copies: All copies
            class_map: Current class map

        Returns:
            List of AnswerOutlier objects
        """
        outliers = []

        for question_id in class_map.question_analyses.keys():
            # Use clustering to detect noise points
            outlier_ids = self.clustering.detect_outliers(copies, question_id)

            for copy_id in outlier_ids:
                copy = next((c for c in copies if c.id == copy_id), None)
                if copy and question_id in copy.content_summary:
                    outliers.append(AnswerOutlier(
                        copy_id=copy_id,
                        question_id=question_id,
                        description=copy.content_summary[question_id][:100],
                        outlier_reason="Significantly different from other answers",
                        suggested_action="review"
                    ))

        return outliers

    def get_class_context(
        self,
        class_map: ClassAnswerMap,
        question_id: str
    ) -> str:
        """
        Generate context string for grading a question.

        Args:
            class_map: Analysis results
            question_id: Question to get context for

        Returns:
            Context description string
        """
        if question_id not in class_map.question_analyses:
            return ""

        analysis = class_map.question_analyses[question_id]

        context_parts = []

        if analysis.common_correct:
            context_parts.append(
                f"Common correct approaches: {', '.join(analysis.common_correct[:3])}"
            )

        if analysis.common_errors:
            context_parts.append(
                f"Common errors: {', '.join(analysis.common_errors[:3])}"
            )

        if analysis.unique_approaches:
            context_parts.append(
                f"Unique approaches noted: {len(analysis.unique_approaches)}"
            )

        return "\n".join(context_parts)

    def get_similar_answers(
        self,
        class_map: ClassAnswerMap,
        question_id: str,
        copy_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get answers similar to a specific copy.

        Args:
            class_map: Analysis results
            question_id: Question to look at
            copy_id: Reference copy ID

        Returns:
            List of similar answer summaries
        """
        if question_id not in class_map.clusters:
            return []

        # Find which cluster this copy belongs to
        clusters = class_map.clusters[question_id]

        for cluster in clusters:
            if copy_id in cluster.get("copy_ids", []):
                # Return other copies in this cluster
                similar = []
                for other_id in cluster["copy_ids"]:
                    if other_id != copy_id:
                        similar.append({
                            "copy_id": other_id,
                            "cluster_id": cluster["cluster_id"],
                            "representative": cluster.get("representative_description", "")
                        })
                return similar

        return []
