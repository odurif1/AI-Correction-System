"""
Disagreement Analyzer for conversation-based grading.

Analyzes results from two LLMs and flags questions that need verification.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from difflib import SequenceMatcher


class DisagreementType(Enum):
    """Types of disagreements between LLMs."""
    NONE = "none"
    GRADE_DIFFERENCE = "grade_difference"
    READING_DIFFERENCE = "reading_difference"
    NOT_FOUND_CONFLICT = "not_found_conflict"
    SCALE_DIFFERENCE = "scale_difference"  # Barème détecté différent


@dataclass
class NameDisagreement:
    """Details about a student name disagreement."""
    llm1_name: str
    llm2_name: str
    similarity: float
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "llm1_name": self.llm1_name,
            "llm2_name": self.llm2_name,
            "similarity": round(self.similarity, 2),
            "reason": self.reason
        }


@dataclass
class QuestionDisagreement:
    """Details about a disagreement for a specific question."""
    question_id: str
    disagreement_type: DisagreementType
    llm1_grade: float
    llm2_grade: float
    grade_difference: float
    llm1_reading: str
    llm2_reading: str
    llm1_confidence: float
    llm2_confidence: float
    llm1_max_points: float = 1.0  # Barème détecté par LLM1
    llm2_max_points: float = 1.0  # Barème détecté par LLM2
    reading_difference_type: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "type": self.disagreement_type.value,
            "llm1": {
                "grade": self.llm1_grade,
                "reading": self.llm1_reading,
                "confidence": self.llm1_confidence,
                "max_points": self.llm1_max_points
            },
            "llm2": {
                "grade": self.llm2_grade,
                "reading": self.llm2_reading,
                "confidence": self.llm2_confidence,
                "max_points": self.llm2_max_points
            },
            "grade_difference": self.grade_difference,
            "reason": self.reason
        }


@dataclass
class DisagreementReport:
    """Complete report of disagreements between two LLM results."""
    agreed_questions: List[str]
    flagged_questions: List[QuestionDisagreement]
    total_questions: int
    agreement_rate: float
    # Copy-level disagreements
    name_disagreement: Optional[NameDisagreement] = None
    llm1_name: Optional[str] = None
    llm2_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "total_questions": self.total_questions,
            "agreed": len(self.agreed_questions),
            "flagged": len(self.flagged_questions),
            "agreement_rate": round(self.agreement_rate, 2),
            "agreed_questions": self.agreed_questions,
            "flagged_questions": [d.to_dict() for d in self.flagged_questions]
        }
        if self.name_disagreement:
            result["name_disagreement"] = self.name_disagreement.to_dict()
        return result

    @property
    def has_any_disagreement(self) -> bool:
        """Check if there's any disagreement (name or questions)."""
        return len(self.flagged_questions) > 0 or self.name_disagreement is not None


@dataclass
class PositionSwap:
    """Record of a position swap (flip-flop) between two LLMs."""
    question_id: str
    initial_grades: Tuple[float, float]  # (llm1_initial, llm2_initial)
    after_verification: Tuple[float, float]  # (llm1_after, llm2_after)
    is_swap: bool  # True if positions were swapped
    confidence_dropped: bool  # True if either LLM dropped confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "initial_grades": list(self.initial_grades),
            "after_verification": list(self.after_verification),
            "is_swap": self.is_swap,
            "confidence_dropped": self.confidence_dropped
        }


# Seuils hardcodés (simples, pas de config)
GRADE_DIFF_PERCENTAGE = 0.10  # 10% du barème = différence significative
READING_SIMILARITY_THRESHOLD = 0.3  # Similarité Jaccard minimum
POSITION_SWAP_THRESHOLD = 0.5  # Minimum grade difference to detect swap


def detect_position_swaps(
    initial_results: Dict[str, Dict[str, float]],
    after_verification: Dict[str, Dict[str, float]],
    confidence_initial: Dict[str, Tuple[float, float]] = None,
    confidence_after: Dict[str, Tuple[float, float]] = None
) -> List[PositionSwap]:
    """
    Detect position swaps (flip-flops) after verification.

    A swap occurs when:
    - LLM1 was higher than LLM2 initially
    - After verification, LLM1 is lower than LLM2 (or vice versa)

    This is a red flag suggesting suggestibility rather than independent re-analysis.

    Args:
        initial_results: Dict mapping qid -> {"llm1": grade, "llm2": grade}
        after_verification: Dict mapping qid -> {"llm1": grade, "llm2": grade}
        confidence_initial: Optional dict mapping qid -> (conf1, conf2)
        confidence_after: Optional dict mapping qid -> (conf1, conf2)

    Returns:
        List of PositionSwap objects for questions where a swap was detected
    """
    swaps = []
    confidence_initial = confidence_initial or {}
    confidence_after = confidence_after or {}

    for qid in initial_results:
        init = initial_results[qid]
        after = after_verification.get(qid, init)

        llm1_init = init.get("llm1", 0)
        llm2_init = init.get("llm2", 0)
        llm1_after = after.get("llm1", 0)
        llm2_after = after.get("llm2", 0)

        # Check for swap (positions crossed)
        init_diff = llm1_init - llm2_init
        after_diff = llm1_after - llm2_after

        # Swap detected if signs are opposite and difference is significant
        is_swap = (
            abs(init_diff) >= POSITION_SWAP_THRESHOLD and
            abs(after_diff) >= POSITION_SWAP_THRESHOLD and
            (init_diff > 0 and after_diff < 0) or (init_diff < 0 and after_diff > 0)
        )

        # Check if confidence dropped
        conf_init = confidence_initial.get(qid, (1.0, 1.0))
        conf_after = confidence_after.get(qid, (1.0, 1.0))
        confidence_dropped = (
            conf_after[0] < conf_init[0] - 0.2 or
            conf_after[1] < conf_init[1] - 0.2
        )

        if is_swap:
            swaps.append(PositionSwap(
                question_id=qid,
                initial_grades=(llm1_init, llm2_init),
                after_verification=(llm1_after, llm2_after),
                is_swap=is_swap,
                confidence_dropped=confidence_dropped
            ))

    return swaps


def compute_reading_anchors(
    llm1_results: Dict[str, Any],
    llm2_results: Dict[str, Any],
    disagreement_qids: List[str]
) -> Dict[str, str]:
    """
    Compute reading anchors for questions where both LLMs initially agreed on reading.

    These anchors should be used in ultimatum to prevent hallucination.

    Args:
        llm1_results: Full results from LLM1
        llm2_results: Full results from LLM2
        disagreement_qids: List of question IDs that are in disagreement (about grade)

    Returns:
        Dict mapping question_id -> anchored reading (only for questions with reading agreement)
    """
    anchors = {}
    llm1_questions = llm1_results.get("questions", {})
    llm2_questions = llm2_results.get("questions", {})

    for qid in disagreement_qids:
        q1 = llm1_questions.get(qid, {})
        q2 = llm2_questions.get(qid, {})

        reading1 = (q1.get("student_answer_read") or "").strip().lower()
        reading2 = (q2.get("student_answer_read") or "").strip().lower()

        # Check if readings are similar (agreed)
        if not reading1 or not reading2:
            continue

        # Use SequenceMatcher for fuzzy comparison
        similarity = SequenceMatcher(None, reading1, reading2).ratio()

        if similarity >= 0.8:  # High similarity = reading agreement
            # Use the longer reading as anchor (usually more complete)
            anchor = q1.get("student_answer_read") if len(reading1) >= len(reading2) else q2.get("student_answer_read")
            if anchor:
                anchors[qid] = anchor.strip()

    return anchors


class DisagreementAnalyzer:
    """
    Analyse les désaccords entre deux résultats LLM.

    Flag les questions nécessitant une cross-verification.
    """

    def analyze(
        self,
        llm1_results: Dict[str, Any],
        llm2_results: Dict[str, Any]
    ) -> DisagreementReport:
        """
        Compare results and identify disagreements.
        Includes both copy-level (name) and question-level disagreements.
        """
        # ===== Copy-level: Check student name =====
        name_disagreement = None
        llm1_name = llm1_results.get("student_name")
        llm2_name = llm2_results.get("student_name")

        if llm1_name and llm2_name:
            name1_normalized = llm1_name.strip().lower()
            name2_normalized = llm2_name.strip().lower()

            if name1_normalized != name2_normalized:
                # Any name difference is a disagreement - names must match exactly
                similarity = SequenceMatcher(None, name1_normalized, name2_normalized).ratio()
                name_disagreement = NameDisagreement(
                    llm1_name=llm1_name.strip(),
                    llm2_name=llm2_name.strip(),
                    similarity=similarity,
                    reason=f"Noms différents: '{llm1_name}' vs '{llm2_name}'"
                )

        # ===== Question-level disagreements =====
        llm1_questions = llm1_results.get("questions", {})
        llm2_questions = llm2_results.get("questions", {})

        all_qids = set(llm1_questions.keys()) | set(llm2_questions.keys())

        agreed = []
        flagged = []

        for qid in sorted(all_qids):
            q1 = llm1_questions.get(qid, {})
            q2 = llm2_questions.get(qid, {})

            disagreement = self._analyze_question(qid, q1, q2)

            if disagreement is None:
                agreed.append(qid)
            else:
                flagged.append(disagreement)

        total = len(all_qids)
        agreement_rate = len(agreed) / total if total > 0 else 1.0

        return DisagreementReport(
            agreed_questions=agreed,
            flagged_questions=flagged,
            total_questions=total,
            agreement_rate=agreement_rate,
            name_disagreement=name_disagreement,
            llm1_name=llm1_name,
            llm2_name=llm2_name
        )

    def _analyze_question(
        self,
        qid: str,
        q1: Dict[str, Any],
        q2: Dict[str, Any]
    ) -> Optional[QuestionDisagreement]:
        """Analyse une question. Retourne None si accord, QuestionDisagreement si désaccord."""
        grade1 = float(q1.get("grade", 0))
        grade2 = float(q2.get("grade", 0))
        reading1 = q1.get("student_answer_read", "") or ""
        reading2 = q2.get("student_answer_read", "") or ""
        conf1 = float(q1.get("confidence", 1.0))
        conf2 = float(q2.get("confidence", 1.0))
        max_pts1 = float(q1.get("max_points", 1.0))
        max_pts2 = float(q2.get("max_points", 1.0))

        grade_diff = abs(grade1 - grade2)
        scale_diff = abs(max_pts1 - max_pts2)

        # Check FIRST: barème différent (scale difference)
        # This is important because it affects how we interpret grades
        if scale_diff > 0.1:  # Tolérance de 0.1 pour les arrondis
            return QuestionDisagreement(
                question_id=qid,
                disagreement_type=DisagreementType.SCALE_DIFFERENCE,
                llm1_grade=grade1,
                llm2_grade=grade2,
                grade_difference=grade_diff,
                llm1_reading=reading1,
                llm2_reading=reading2,
                llm1_confidence=conf1,
                llm2_confidence=conf2,
                llm1_max_points=max_pts1,
                llm2_max_points=max_pts2,
                reason=f"Barème détecté différent: {max_pts1} vs {max_pts2} points"
            )

        # Check: un LLM trouve, l'autre non
        not_found_indicators = ['non', 'no', 'pas visible', 'not visible', 'absent', 'not found', 'non trouvé']
        r1_not_found = any(ind in reading1.lower() for ind in not_found_indicators)
        r2_not_found = any(ind in reading2.lower() for ind in not_found_indicators)

        if r1_not_found != r2_not_found:
            return QuestionDisagreement(
                question_id=qid,
                disagreement_type=DisagreementType.NOT_FOUND_CONFLICT,
                llm1_grade=grade1,
                llm2_grade=grade2,
                grade_difference=grade_diff,
                llm1_reading=reading1,
                llm2_reading=reading2,
                llm1_confidence=conf1,
                llm2_confidence=conf2,
                llm1_max_points=max_pts1,
                llm2_max_points=max_pts2,
                reason="Un LLM a trouvé la réponse, l'autre non"
            )

        # Check: lecture différente (AVANT la note car c'est souvent la cause racine)
        reading_diff_type = self._classify_reading_difference(reading1, reading2)
        if reading_diff_type in ("substantial", "partial"):
            return QuestionDisagreement(
                question_id=qid,
                disagreement_type=DisagreementType.READING_DIFFERENCE,
                llm1_grade=grade1,
                llm2_grade=grade2,
                grade_difference=grade_diff,
                llm1_reading=reading1,
                llm2_reading=reading2,
                llm1_confidence=conf1,
                llm2_confidence=conf2,
                llm1_max_points=max_pts1,
                llm2_max_points=max_pts2,
                reading_difference_type=reading_diff_type,
                reason=f"Lectures différentes ({reading_diff_type}): '{reading1}' vs '{reading2}'"
            )

        # Check: différence de note significative (seulement si lectures identiques)
        # Seuil = 10% du barème de la question
        grade_threshold = max_pts1 * GRADE_DIFF_PERCENTAGE
        if grade_diff >= grade_threshold:
            return QuestionDisagreement(
                question_id=qid,
                disagreement_type=DisagreementType.GRADE_DIFFERENCE,
                llm1_grade=grade1,
                llm2_grade=grade2,
                grade_difference=grade_diff,
                llm1_reading=reading1,
                llm2_reading=reading2,
                llm1_confidence=conf1,
                llm2_confidence=conf2,
                llm1_max_points=max_pts1,
                llm2_max_points=max_pts2,
                reason=f"Différence de note: {grade_diff:.2f} pts (seuil: {grade_threshold:.2f} pts = 10% du barème)"
            )

        # Accord
        return None

    def _classify_reading_difference(self, reading1: str, reading2: str) -> Optional[str]:
        """Classifie le type de différence de lecture."""
        if not reading1 and not reading2:
            return None
        if not reading1 or not reading2:
            return "substantial"

        r1 = reading1.lower().strip()
        r2 = reading2.lower().strip()

        if r1 == r2:
            return None

        # Différence d'accents seulement
        import unicodedata
        def strip_accents(s):
            return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

        if strip_accents(r1) == strip_accents(r2):
            return "accent"

        # Chevauchement partiel
        if r1 in r2 or r2 in r1:
            return "partial"

        # Similarité Jaccard
        words1 = set(r1.split())
        words2 = set(r2.split())

        stop_words = {'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'a', 'est', 'et', 'ou',
                      'the', 'a', 'an', 'is', 'are', 'and', 'or', 'to', 'in', 'of'}
        words1 -= stop_words
        words2 -= stop_words

        if not words1 or not words2:
            return "substantial"

        intersection = len(words1 & words2)
        union = len(words1 | words2)
        similarity = intersection / union if union > 0 else 0

        if similarity < READING_SIMILARITY_THRESHOLD:
            return "substantial"

        return None
