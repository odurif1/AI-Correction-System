"""
Disagreement Analyzer for conversation-based grading.

Analyzes results from two LLMs and flags questions that need verification.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class DisagreementType(Enum):
    """Types of disagreements between LLMs."""
    NONE = "none"
    GRADE_DIFFERENCE = "grade_difference"
    READING_DIFFERENCE = "reading_difference"
    NOT_FOUND_CONFLICT = "not_found_conflict"


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
    reading_difference_type: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "type": self.disagreement_type.value,
            "llm1": {
                "grade": self.llm1_grade,
                "reading": self.llm1_reading,
                "confidence": self.llm1_confidence
            },
            "llm2": {
                "grade": self.llm2_grade,
                "reading": self.llm2_reading,
                "confidence": self.llm2_confidence
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_questions": self.total_questions,
            "agreed": len(self.agreed_questions),
            "flagged": len(self.flagged_questions),
            "agreement_rate": round(self.agreement_rate, 2),
            "agreed_questions": self.agreed_questions,
            "flagged_questions": [d.to_dict() for d in self.flagged_questions]
        }


# Seuils hardcodés (simples, pas de config)
GRADE_THRESHOLD = 0.5  # Différence de note pour flagger
READING_SIMILARITY_THRESHOLD = 0.3  # Similarité Jaccard minimum


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
        """
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
            agreement_rate=agreement_rate
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

        grade_diff = abs(grade1 - grade2)

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
                reason="Un LLM a trouvé la réponse, l'autre non"
            )

        # Check: différence de note significative
        if grade_diff >= GRADE_THRESHOLD:
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
                reason=f"Différence de note: {grade_diff:.1f} points"
            )

        # Check: lecture substantiellement différente
        reading_diff_type = self._classify_reading_difference(reading1, reading2)
        if reading_diff_type == "substantial":
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
                reading_difference_type=reading_diff_type,
                reason="Lectures substantiellement différentes"
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
