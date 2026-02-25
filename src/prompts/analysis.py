"""
Analysis prompts for the AI correction system.

Contains:
- Rule extraction
- Cross-copy analysis
"""

from typing import Dict, Any, Optional, List
from prompts.common import detect_language

def build_rule_extraction_prompt(
    teacher_decision: str,
    context: str,
    language: str = 'en'
) -> str:
    """
    Build prompt to extract generalizable rule from teacher decision.

    Args:
        teacher_decision: What the teacher said/did
        context: Context of the decision (question, student answer, etc.)
        language: Language for prompt

    Returns:
        Formatted prompt for rule extraction
    """
    if language == 'fr':
        return f"""Extrais une règle de correction généralisable de cette décision d'enseignant.

DÉCISION DE L'ENSEIGNANT: {teacher_decision}

CONTEXTE: {context}

TÂCHE:
Extraire une règle qui peut être appliquée à d'autres copies similaires.

FORMAT:
SI [condition générale]
ALORS [action de correction]
PARCE QUE [raison pédagogique]

La règle doit être:
- Généralisable (pas spécifique à cette copie)
- Applicable
- Cohérente avec les principes de correction équitable"""
    else:
        return f"""Extract a generalizable grading rule from this teacher decision.

TEACHER DECISION: {teacher_decision}

CONTEXT: {context}

TASK:
Extract a rule that can be applied to other similar copies.

FORMAT:
IF [general condition]
THEN [grading action]
BECAUSE [pedagogical reasoning]

The rule must be:
- Generalizable (not specific to this copy)
- Actionable
- Consistent with fair grading principles"""



def build_cross_copy_analysis_prompt(
    question_id: str,
    question_text: str,
    answer_summaries: List[tuple[str, str]],
    max_points: float,
    language: str = None
) -> str:
    """
    Build prompt for cross-copy analysis.

    Args:
        question_id: Question identifier
        question_text: The question being asked
        answer_summaries: List of (copy_id, answer_summary) tuples
        max_points: Maximum points for this question
        language: Language for prompts (auto-detected if None)

    Returns:
        Formatted prompt for cross-copy analysis
    """
    # Auto-detect language if not provided
    if language is None:
        combined_text = f"{question_text} " + " ".join([ans for _, ans in answer_summaries])
        language = detect_language(combined_text)

    # Build answers section
    answers_section = ""
    for i, (copy_id, answer) in enumerate(answer_summaries, 1):
        answers_section += f"{i}. Student {copy_id}: {answer}\n"

    if language == 'fr':
        return f"""Analyse les réponses des élèves à cette question.

QUESTION: {question_text}
IDENTIFIANT: {question_id}
NOTE MAXIMALE: {max_points} points

RÉPONSES DES ÉLÈVES:
{answers_section}

TÂCHE:
1. Identifier les réponses courantes correctes
2. Identifier les erreurs communes
3. Identifier les approches uniques ou originales
4. Estimer la difficulté de la question (0.0-1.0)

FORMAT DE RÉPONSE:
COMMON_CORRECT: <liste des réponses correctes communes, séparées par virgule, ou "none">
COMMON_ERRORS: <liste des erreurs communes, séparées par virgule, ou "none">
UNIQUE_APPROACHES: <liste des approches uniques, séparées par virgule, ou "none">
DIFFICULTY_ESTIMATE: <estimation 0.0-1.0>"""
    else:
        return f"""Analyze student answers to this question.

QUESTION: {question_text}
IDENTIFIER: {question_id}
MAX POINTS: {max_points} points

STUDENT ANSWERS:
{answers_section}

TASK:
1. Identify common correct answers
2. Identify common errors
3. Identify unique or original approaches
4. Estimate difficulty of the question (0.0-1.0)

RESPONSE FORMAT:
COMMON_CORRECT: <list of common correct answers, comma-separated, or "none">
COMMON_ERRORS: <list of common errors, comma-separated, or "none">
UNIQUE_APPROACHES: <list of unique approaches, comma-separated, or "none">
DIFFICULTY_ESTIMATE: <estimate 0.0-1.0>"""


# ============= UNIFIED VERIFICATION PROMPTS =============


