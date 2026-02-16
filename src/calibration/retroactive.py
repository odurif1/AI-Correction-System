"""
Retroactive application of grading changes.

When a teacher adjusts a grade, this module finds similar answers
and applies the same principle to already-graded copies.
"""

import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from core.models import (
    GradingSession, GradedCopy, TeacherDecision, SimilarCopies,
    CopyDocument, UncertaintyType
)
from ai.openai_provider import OpenAIProvider
from analysis.clustering import EmbeddingClustering
from storage.session_store import SessionStore


class RetroactiveApplier:
    """
    Applies teacher decisions to similar answers retroactively.

    This is the CRITICAL component for ensuring fairness:
    - When teacher adjusts one copy, find all similar copies
    - Propose applying the same change to teacher
    - Apply to both already-graded and pending copies
    - Maintain audit trail
    """

    def __init__(
        self,
        session: GradingSession,
        ai_provider: OpenAIProvider = None
    ):
        """
        Initialize the retroactive applier.

        Args:
            session: Current grading session
            ai_provider: AI provider (creates default if None)
        """
        self.session = session
        self.ai = ai_provider or OpenAIProvider()
        self.clustering = EmbeddingClustering()

    def find_similar(
        self,
        decision: TeacherDecision
    ) -> SimilarCopies:
        """
        Find copies similar to the source of a teacher decision.

        Args:
            decision: Teacher decision with source copy info

        Returns:
            SimilarCopies with already_graded and pending lists
        """
        source_copy = next(
            (c for c in self.session.copies if c.id == decision.source_copy_id),
            None
        )

        if not source_copy:
            return SimilarCopies()

        # Get all copy IDs
        all_ids = [c.id for c in self.session.copies if c.id != decision.source_copy_id]

        # Method 1: Find by cluster membership
        cluster_ids = set()
        if self.session.clusters:
            for cluster_id, copy_ids in self.session.clusters.items():
                if decision.source_copy_id in copy_ids:
                    cluster_ids.update(copy_ids)

        # Method 2: Find by semantic similarity
        similar_by_embedding = []

        if source_copy.embedding:
            for copy in self.session.copies:
                if copy.id != decision.source_copy_id and copy.embedding:
                    similarity = self.clustering._cosine_similarity(
                        source_copy.embedding,
                        copy.embedding
                    )
                    if similarity >= 0.85:  # High similarity threshold
                        similar_by_embedding.append((copy.id, similarity))

        # Combine and deduplicate
        similar_ids = cluster_ids | {cid for cid, _ in similar_by_embedding}

        # Separate already graded vs pending
        already_graded = [
            cid for cid in similar_ids
            if any(gc.copy_id == cid for gc in self.session.graded_copies)
        ]
        pending = [
            cid for cid in similar_ids
            if cid not in already_graded
        ]

        # Build similarity scores
        similarity_scores = {}
        for cid in similar_ids:
            # For cluster members, assign high similarity
            if cid in cluster_ids:
                similarity_scores[cid] = 0.9
            else:
                # Use embedding similarity
                for copy_id, sim in similar_by_embedding:
                    if copy_id == cid:
                        similarity_scores[cid] = sim

        return SimilarCopies(
            already_graded=already_graded,
            pending=pending,
            similarity_scores=similarity_scores
        )

    async def propose_change(
        self,
        decision: TeacherDecision,
        similar: SimilarCopies
    ) -> bool:
        """
        Propose applying a change to similar copies to the teacher.

        Args:
            decision: Teacher decision to propagate
            similar: Similar copies found

        Returns:
            True if teacher approves
        """
        if not similar.already_graded and not similar.pending:
            return False

        # Build proposal message
        proposal = self._build_proposal(decision, similar)

        # In CLI mode, this would prompt the user
        # For now, we'll assume approval if applies_to_all is True
        # In production, this would be interactive

        return decision.applies_to_all

    def _build_proposal(
        self,
        decision: TeacherDecision,
        similar: SimilarCopies
    ) -> str:
        """Build a proposal message for the teacher."""
        lines = [
            f"Teacher Decision Recorded:",
            f"  Question: {decision.question_id}",
            f"  Copy: {decision.source_copy_id}",
            f"  Guidance: {decision.teacher_guidance}",
            ""
        ]

        if decision.extracted_rule:
            lines.extend([
                f"Extracted Rule:",
                f"  {decision.extracted_rule}",
                ""
            ])

        if similar.already_graded:
            lines.extend([
                f"Apply to {len(similar.already_graded)} already graded copies:",
                f"  {', '.join(similar.already_graded[:5])}",
                ""
            ])

        if similar.pending:
            lines.extend([
                f"Apply to {len(similar.pending)} pending copies:",
                f"  {', '.join(similar.pending[:5])}",
                ""
            ])

        lines.append("Apply to all similar copies?")

        return "\n".join(lines)

    async def apply(
        self,
        decision: TeacherDecision,
        similar: SimilarCopies,
        session_store: SessionStore
    ) -> int:
        """
        Apply a teacher decision to similar copies.

        Args:
            decision: Teacher decision with extracted rule
            similar: Similar copies to apply to
            session_store: Storage for saving updates

        Returns:
            Number of copies updated
        """
        updated_count = 0

        # Re-grade already-graded copies
        for copy_id in similar.already_graded:
            graded = next(
                (g for g in self.session.graded_copies if g.copy_id == copy_id),
                None
            )

            if graded and decision.question_id in graded.grades:
                # Calculate new grade based on the decision
                grade_adjustment = decision.new_score - decision.original_score
                new_grade = graded.grades[decision.question_id] + grade_adjustment

                # Update
                old_grade = graded.grades[decision.question_id]
                graded.grades[decision.question_id] = max(0, new_grade)

                # Update total
                graded.total_score += (new_grade - old_grade)

                # Track adjustment
                from core.models import TeacherAdjustment
                graded.adjustments.append(TeacherAdjustment(
                    question_id=decision.question_id,
                    original_score=old_grade,
                    new_score=graded.grades[decision.question_id],
                    reason=f"Retroactive: {decision.teacher_guidance}"
                ))

                # Save
                session_store.update_copy(graded)
                updated_count += 1

        # For pending copies, add to policy so they're graded correctly
        if similar.pending and decision.extracted_rule:
            self.session.policy.teacher_decisions.append(decision.extracted_rule)
            session_store.save_policy(self.session.session_id, self.session.policy)

        return updated_count

    async def extract_and_apply(
        self,
        question_id: str,
        copy_id: str,
        teacher_guidance: str,
        original_score: float,
        new_score: float,
        session_store: SessionStore
    ) -> Tuple[int, str]:
        """
        Complete workflow: extract rule, find similar, apply.

        Args:
            question_id: Question being adjusted
            copy_id: Source copy ID
            teacher_guidance: Teacher's explanation
            original_score: Original grade
            new_score: New grade from teacher
            session_store: Storage

        Returns:
            (number_updated, extracted_rule)
        """
        # Create decision
        decision = TeacherDecision(
            question_id=question_id,
            source_copy_id=copy_id,
            teacher_guidance=teacher_guidance,
            original_score=original_score,
            new_score=new_score,
            applies_to_all=True
        )

        # Get source copy for context
        source_copy = next(
            (c for c in self.session.copies if c.id == copy_id),
            None
        )

        # Extract rule
        if source_copy:
            question_context = f"Question {question_id}"
            student_answer = source_copy.content_summary.get(question_id, "")

            rule = self.ai.extract_rule(
                teacher_decision=teacher_guidance,
                question_context=question_context,
                original_grade=original_score,
                new_grade=new_score,
                student_answer=student_answer
            )

            decision.extracted_rule = rule
        else:
            decision.extracted_rule = teacher_guidance

        # Find similar copies
        similar = self.find_similar(decision)

        # Propose to teacher (auto-approve for now)
        approved = await self.propose_change(decision, similar)

        if approved:
            # Apply
            updated = await self.apply(decision, similar, session_store)
            return updated, decision.extracted_rule

        return 0, decision.extracted_rule


class RetroactiveBatch:
    """
    Batch processing of retroactive changes.

    Handles multiple teacher decisions efficiently.
    """

    def __init__(self, session: GradingSession):
        """Initialize batch processor."""
        self.session = session
        self.applier = RetroactiveApplier(session)

    async def process_decisions(
        self,
        decisions: List[TeacherDecision],
        session_store: SessionStore
    ) -> Dict[str, int]:
        """
        Process multiple teacher decisions.

        Args:
            decisions: List of teacher decisions
            session_store: Storage

        Returns:
            Dict with {question_id: count_updated}
        """
        results = {}

        for decision in decisions:
            similar = self.applier.find_similar(decision)
            updated = await self.applier.apply(decision, similar, session_store)

            key = f"{decision.question_id}"
            results[key] = results.get(key, 0) + updated

        return results

    async def resolve_inconsistencies(
        self,
        inconsistencies: List[Dict],
        session_store: SessionStore
    ) -> List[Dict]:
        """
        Resolve grading inconsistencies.

        Args:
            inconsistencies: List of inconsistency reports
            session_store: Storage

        Returns:
            List of resolution results
        """
        resolutions = []

        for inc in inconsistencies:
            # Create a unified decision
            question_id = inc["question_id"]
            suggested_grade = inc.get("suggested_grade")

            if suggested_grade is None:
                from grading.uncertainty import ConsistencyChecker
                checker = ConsistencyChecker()
                suggested_grade = checker.suggest_unified_grade(inc)

            # Create decision for adjustment
            # This would normally ask teacher first
            resolutions.append({
                "question_id": question_id,
                "unified_grade": suggested_grade,
                "affected_copies": len(inc["copy_ids"])
            })

        return resolutions
