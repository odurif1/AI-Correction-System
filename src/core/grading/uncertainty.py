"""
Uncertainty quantification and confidence scoring.

Calculates confidence scores and determines when human input is needed.
"""

import numpy as np
from typing import List, Dict, Optional, Any
from enum import Enum

from core.models import (
    CopyDocument, GradedCopy, ClassAnswerMap, AnswerCluster
)
from config.constants import SIMILARITY_THRESHOLD


class UncertaintySource(str, Enum):
    """Sources of uncertainty in grading."""
    NO_SIMILAR_ANSWERS = "no_similar_answers"
    DISAGREEMENT_IN_CLUSTER = "disagreement_in_cluster"
    OUTLIER_ANSWER = "outlier_answer"
    NO_CRITERIA = "no_criteria"
    UNCLEAR_CONTENT = "unclear_content"
    ALTERNATIVE_METHOD = "alternative_method"
    NO_UNCERTAINTY = "no_uncertainty"


class UncertaintyCalculator:
    """
    Calculates confidence scores for grading decisions.

    Uses multiple factors:
    - Similarity to known patterns
    - Cluster consistency
    - Policy completeness
    - Answer clarity
    """

    def __init__(
        self,
        class_map: ClassAnswerMap = None,
        high_threshold: float = 0.85,
        low_threshold: float = 0.60
    ):
        """
        Initialize the calculator.

        Args:
            class_map: Cross-copy analysis results
            high_threshold: Confidence threshold for auto-grading
            low_threshold: Confidence threshold for asking teacher
        """
        self.class_map = class_map
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def calculate_confidence(
        self,
        copy: CopyDocument,
        question_id: str,
        ai_raw_confidence: float = 0.5
    ) -> float:
        """
        Calculate overall confidence score.

        Args:
            copy: Student's copy
            question_id: Question being graded
            ai_raw_confidence: Raw confidence from AI model

        Returns:
            Final confidence score (0-1)
        """
        factors = self._get_confidence_factors(copy, question_id)

        # Start with AI confidence
        confidence = ai_raw_confidence

        # Apply factors
        for factor_name, factor_value in factors.items():
            if factor_value > 0:
                confidence += factor_value
            elif factor_value < 0:
                confidence -= abs(factor_value)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    def _get_confidence_factors(
        self,
        copy: CopyDocument,
        question_id: str
    ) -> Dict[str, float]:
        """
        Get confidence adjustment factors.

        Returns:
            Dict of factor names to adjustments (-1 to +1)
        """
        factors = {}

        # Factor 1: Is this in a known cluster?
        if copy.cluster_id is not None:
            factors["in_cluster"] = 0.15
        else:
            factors["in_cluster"] = -0.1

        # Factor 2: Cluster consistency (if available)
        if self.class_map and question_id in self.class_map.clusters:
            clusters = self.class_map.clusters[question_id]
            for cluster in clusters:
                if copy.id in cluster.get("copy_ids", []):
                    # Check if cluster has consistent grades
                    if cluster.get("avg_grade") is not None:
                        variance = cluster.get("grade_variance", 0)
                        if variance < 0.5:  # Low variance = consistent
                            factors["cluster_consistency"] = 0.1
                        else:
                            factors["cluster_consistency"] = -0.1
                    break

        # Factor 3: Is this an outlier?
        if self.class_map:
            for outlier in self.class_map.outliers:
                if outlier.copy_id == copy.id and outlier.question_id == question_id:
                    factors["is_outlier"] = -0.25
                    break

        # Factor 4: Answer clarity (length-based heuristic)
        answer = copy.content_summary.get(question_id, "")
        if answer:
            if len(answer) < 10:  # Very short
                factors["answer_clarity"] = -0.15
            elif len(answer) > 100:  # Substantial
                factors["answer_clarity"] = 0.05

        return factors

    def identify_uncertainty_source(
        self,
        copy: CopyDocument,
        question_id: str,
        confidence: float
    ) -> UncertaintySource:
        """
        Identify the primary source of uncertainty.

        Args:
            copy: Student's copy
            question_id: Question being graded
            confidence: Calculated confidence score

        Returns:
            UncertaintySource enum value
        """
        if confidence >= self.high_threshold:
            return UncertaintySource.NO_UNCERTAINTY

        # Check for specific sources
        factors = self._get_confidence_factors(copy, question_id)

        if factors.get("is_outlier", 0) < -0.2:
            return UncertaintySource.OUTLIER_ANSWER

        if factors.get("in_cluster", 0) < 0:
            return UncertaintySource.NO_SIMILAR_ANSWERS

        if factors.get("cluster_consistency", 0) < 0:
            return UncertaintySource.DISAGREEMENT_IN_CLUSTER

        if factors.get("answer_clarity", 0) < -0.1:
            return UncertaintySource.UNCLEAR_CONTENT

        return UncertaintySource.ALTERNATIVE_METHOD

    def needs_teacher_input(
        self,
        confidence: float,
        uncertainty_source: UncertaintySource
    ) -> bool:
        """
        Determine if teacher input is needed.

        Args:
            confidence: Confidence score
            uncertainty_source: Source of uncertainty

        Returns:
            True if teacher should be consulted
        """
        # Low confidence always needs teacher
        if confidence < self.low_threshold:
            return True

        # Certain uncertainty types need teacher even with medium confidence
        if uncertainty_source in [
            UncertaintySource.OUTLIER_ANSWER,
            UncertaintySource.ALTERNATIVE_METHOD
        ]:
            return confidence < self.high_threshold

        return False

    def should_flag_for_review(self, confidence: float) -> bool:
        """
        Check if grading should be flagged for teacher review.

        Args:
            confidence: Confidence score

        Returns:
            True if should be flagged
        """
        return self.low_threshold <= confidence < self.high_threshold

    def explain_confidence(
        self,
        copy: CopyDocument,
        question_id: str
    ) -> str:
        """
        Generate human-readable explanation of confidence.

        Args:
            copy: Student's copy
            question_id: Question being graded

        Returns:
            Explanation string
        """
        factors = self._get_confidence_factors(copy, question_id)

        explanations = []

        for factor, value in factors.items():
            if value > 0:
                explanations.append(f"+{factor} increases confidence")
            elif value < 0:
                explanations.append(f"-{factor} decreases confidence")

        return "; ".join(explanations) if explanations else "Standard confidence"


class ConsistencyChecker:
    """
    Checks grading consistency across similar answers.
    """

    def __init__(self, class_map: ClassAnswerMap = None):
        """Initialize consistency checker."""
        self.class_map = class_map

    def check_cluster_consistency(
        self,
        cluster: AnswerCluster,
        grades: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Check if grades in a cluster are consistent.

        Args:
            cluster: AnswerCluster to check
            grades: Dict of {copy_id: grade}

        Returns:
            Consistency report
        """
        cluster_grades = [
            grades.get(cid, 0) for cid in cluster.copy_ids
            if cid in grades
        ]

        if not cluster_grades:
            return {
                "consistent": True,
                "variance": 0,
                "range": 0
            }

        variance = np.var(cluster_grades)
        grade_range = max(cluster_grades) - min(cluster_grades)

        # Threshold for "consistent"
        is_consistent = variance < 1.0 and grade_range < 2.0

        return {
            "consistent": is_consistent,
            "variance": float(variance),
            "range": float(grade_range),
            "mean": float(np.mean(cluster_grades)),
            "grades": cluster_grades
        }

    def find_inconsistencies(
        self,
        class_map: ClassAnswerMap,
        all_graded: List[GradedCopy]
    ) -> List[Dict[str, Any]]:
        """
        Find grading inconsistencies across the class.

        Args:
            class_map: Class answer map
            all_graded: All graded copies

        Returns:
            List of inconsistency reports
        """
        inconsistencies = []

        # Build grade lookup
        grade_lookup = {}
        for graded in all_graded:
            grade_lookup[graded.copy_id] = graded.grades

        # Check each question's clusters
        for question_id, clusters in class_map.clusters.items():
            for cluster_data in clusters:
                cluster = AnswerCluster(**cluster_data)

                # Get grades for this cluster
                cluster_grades = {}
                for copy_id in cluster.copy_ids:
                    if copy_id in grade_lookup:
                        cluster_grades[copy_id] = grade_lookup[copy_id].get(
                            question_id, 0
                        )

                # Check consistency
                check = self.check_cluster_consistency(cluster, cluster_grades)

                if not check["consistent"]:
                    inconsistencies.append({
                        "question_id": question_id,
                        "cluster_id": cluster.cluster_id,
                        "copy_ids": cluster.copy_ids,
                        "grades": cluster_grades,
                        "variance": check["variance"],
                        "range": check["range"],
                        "mean": check["mean"]
                    })

        return inconsistencies

    def suggest_unified_grade(
        self,
        inconsistency: Dict[str, Any]
    ) -> float:
        """
        Suggest a unified grade for inconsistent grading.

        Args:
            inconsistency: Inconsistency report

        Returns:
            Suggested grade
        """
        # Use mean rounded to nearest half-point
        mean = inconsistency.get("mean", 0)
        return round(mean * 2) / 2
