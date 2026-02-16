"""
Consistency checking and calibration.

Detects grading inconsistencies and proposes corrections.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from core.models import (
    GradingSession, GradedCopy, InconsistencyReport,
    ClassAnswerMap
)
from grading.uncertainty import ConsistencyChecker


class ConsistencyDetector:
    """
    Detects grading inconsistencies across the class.

    Identifies:
    - Same answers with different grades
    - Cluster inconsistencies
    - Outlier grading patterns
    """

    def __init__(self, session: GradingSession):
        """
        Initialize detector.

        Args:
            session: Grading session to analyze
        """
        self.session = session
        self.checker = ConsistencyChecker(session.class_map)

    def detect_all(self) -> List[InconsistencyReport]:
        """
        Detect all inconsistencies in the session.

        Returns:
            List of inconsistency reports
        """
        inconsistencies = []

        # Check cluster-based inconsistencies
        if self.session.class_map:
            cluster_inc = self._check_cluster_inconsistencies()
            inconsistencies.extend(cluster_inc)

        # Check similar answers (cross-cluster)
        similar_inc = self._check_similar_answer_inconsistencies()
        inconsistencies.extend(similar_inc)

        # Check score distribution anomalies
        distribution_inc = self._check_distribution_anomalies()
        inconsistencies.extend(distribution_inc)

        return inconsistencies

    def _check_cluster_inconsistencies(self) -> List[InconsistencyReport]:
        """Check for inconsistencies within answer clusters."""
        reports = []

        if not self.session.class_map:
            return reports

        # Build grade lookup
        grade_lookup = {}
        for graded in self.session.graded_copies:
            grade_lookup[graded.copy_id] = graded.grades

        # Check each question's clusters
        for question_id, clusters in self.session.class_map.clusters.items():
            for cluster_data in clusters:
                cluster_grades = {}

                for copy_id in cluster_data.get("copy_ids", []):
                    if copy_id in grade_lookup and question_id in grade_lookup[copy_id]:
                        cluster_grades[copy_id] = grade_lookup[copy_id][question_id]

                if len(cluster_grades) < 2:
                    continue

                # Check variance
                grades_list = list(cluster_grades.values())
                variance = np.var(grades_list)
                grade_range = max(grades_list) - min(grades_list)

                if variance > 0.5 or grade_range > 1.5:
                    # Inconsistency detected
                    reports.append(InconsistencyReport(
                        question_id=question_id,
                        copy_ids=list(cluster_grades.keys()),
                        grades=cluster_grades,
                        max_difference=grade_range,
                        suggested_grade=round(np.mean(grades_list) * 2) / 2,  # Round to half
                        reasoning=f"Grades in same cluster vary by {grade_range:.1f} points"
                    ))

        return reports

    def _check_similar_answer_inconsistencies(self) -> List[InconsistencyReport]:
        """Check for inconsistencies between similar answers in different clusters."""
        reports = []

        # This is more complex - would require embedding similarity
        # For now, return empty
        return reports

    def _check_distribution_anomalies(self) -> List[InconsistencyReport]:
        """Check for unusual score distributions."""
        reports = []

        # Check per-question score distributions
        for question_id in self._get_all_question_ids():
            scores = self._get_question_scores(question_id)

            if len(scores) < 3:
                continue

            # Check for outliers using IQR
            q1 = np.percentile(scores, 25)
            q3 = np.percentile(scores, 75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = [s for s in scores if s < lower_bound or s > upper_bound]

            if outliers:
                # Find which copies have these scores
                copy_ids = self._get_copies_with_scores(question_id, outliers)

                if copy_ids:
                    reports.append(InconsistencyReport(
                        question_id=question_id,
                        copy_ids=copy_ids,
                        grades={cid: self._get_copy_grade(cid, question_id) for cid in copy_ids},
                        max_difference=max(outliers) - min(outliers),
                        reasoning="Scores are statistical outliers"
                    ))

        return reports

    def _get_all_question_ids(self) -> List[str]:
        """Get all question IDs from graded copies."""
        question_ids = set()

        for graded in self.session.graded_copies:
            question_ids.update(graded.grades.keys())

        return list(question_ids)

    def _get_question_scores(self, question_id: str) -> List[float]:
        """Get all scores for a question."""
        return [
            g.grades.get(question_id, 0)
            for g in self.session.graded_copies
            if question_id in g.grades
        ]

    def _get_copies_with_scores(self, question_id: str, scores: List[float]) -> List[str]:
        """Get copy IDs that have specific scores."""
        copy_ids = []

        for graded in self.session.graded_copies:
            if question_id in graded.grades and graded.grades[question_id] in scores:
                copy_ids.append(graded.copy_id)

        return copy_ids

    def _get_copy_grade(self, copy_id: str, question_id: str) -> float:
        """Get a specific copy's grade for a question."""
        for graded in self.session.graded_copies:
            if graded.copy_id == copy_id and question_id in graded.grades:
                return graded.grades[question_id]
        return 0.0


class CalibrationReport:
    """
    Generates calibration reports for the teacher.
    """

    def __init__(self, session: GradingSession):
        """Initialize report generator."""
        self.session = session
        self.detector = ConsistencyDetector(session)

    def generate(self) -> Dict:
        """
        Generate complete calibration report.

        Returns:
            Report dictionary
        """
        inconsistencies = self.detector.detect_all()

        return {
            "session_id": self.session.session_id,
            "total_copies": len(self.session.graded_copies),
            "inconsistencies_found": len(inconsistencies),
            "inconsistencies": [inc.model_dump() for inc in inconsistencies],
            "score_statistics": self._get_score_statistics(),
            "recommendations": self._generate_recommendations(inconsistencies)
        }

    def _get_score_statistics(self) -> Dict:
        """Get overall score statistics."""
        scores = [g.total_score for g in self.session.graded_copies]

        if not scores:
            return {}

        return {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "min": float(min(scores)),
            "max": float(max(scores)),
            "range": float(max(scores) - min(scores))
        }

    def _generate_recommendations(self, inconsistencies: List[InconsistencyReport]) -> List[str]:
        """Generate recommendations based on inconsistencies."""
        recommendations = []

        if not inconsistencies:
            recommendations.append("No inconsistencies detected. Grading is consistent.")
            return recommendations

        # Categorize by severity
        high_severity = [inc for inc in inconsistencies if inc.max_difference > 2.0]
        medium_severity = [inc for inc in inconsistencies if 1.0 <= inc.max_difference <= 2.0]

        if high_severity:
            recommendations.append(
                f"High severity inconsistencies found ({len(high_severity)} questions). "
                "Review recommended."
            )

        if medium_severity:
            recommendations.append(
                f"Medium severity inconsistencies found ({len(medium_severity)} questions). "
                "Consider calibration."
            )

        # Specific recommendations
        question_counts = defaultdict(int)
        for inc in inconsistencies:
            question_counts[inc.question_id] += 1

        if question_counts:
            top_issue = max(question_counts.items(), key=lambda x: x[1])
            recommendations.append(
                f"Question {top_issue[0]} has the most inconsistencies ({top_issue[1]}). "
                "Review grading criteria."
            )

        return recommendations

    def propose_corrections(self) -> List[Dict]:
        """
        Propose specific corrections for inconsistencies.

        Returns:
            List of correction proposals
        """
        inconsistencies = self.detector.detect_all()
        proposals = []

        for inc in inconsistencies:
            proposals.append({
                "question_id": inc.question_id,
                "affected_copies": inc.copy_ids,
                "current_grades": inc.grades,
                "proposed_grade": inc.suggested_grade,
                "reasoning": inc.reasoning
            })

        return proposals
