"""
Batch Grader - Grades all copies in a single API call.

Instead of grading each copy individually, this module sends all copies
to the LLM at once, allowing:
- Consistency across all copies (same answer = same grade)
- Pattern detection (clustering, outliers)
- 90%+ reduction in API calls

Architecture:
- Batches of N copies (configurable via --batch-size)
- Each batch is one API call
- Dual LLM mode: 2 parallel batch calls
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)


def get_agreement_threshold(max_points: float) -> float:
    """
    Calculate the relative agreement threshold based on max_points.

    Args:
        max_points: Maximum points for the question

    Returns:
        Absolute threshold value (e.g., 0.1 for 10% of 1 point)
    """
    return max_points * get_settings().grade_agreement_threshold


def get_flip_flop_threshold(max_points: float) -> float:
    """
    Calculate the relative flip-flop detection threshold based on max_points.

    Args:
        max_points: Maximum points for the question

    Returns:
        Absolute threshold value for detecting significant position swaps
    """
    return max_points * get_settings().flip_flop_threshold


@dataclass
class BatchCopyResult:
    """Result for a single copy within a batch."""
    copy_index: int
    student_name: Optional[str]
    questions: Dict[str, Dict[str, Any]]  # {Q1: {grade, reading, feedback, ...}}
    overall_feedback: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "copy_index": self.copy_index,
            "student_name": self.student_name,
            "questions": self.questions,
            "overall_feedback": self.overall_feedback
        }


@dataclass
class BatchResult:
    """Complete result from batch grading."""
    copies: List[BatchCopyResult]
    patterns: Dict[str, Any]  # Detected patterns across copies
    raw_response: str
    parse_success: bool
    parse_errors: List[str]
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "copies": [c.to_dict() for c in self.copies],
            "patterns": self.patterns,
            "parse_success": self.parse_success,
            "parse_errors": self.parse_errors,
            "duration_ms": self.duration_ms
        }


def build_batch_grading_prompt(
    copies_data: List[Dict[str, Any]],
    questions: Dict[str, Dict[str, Any]],
    language: str = "fr"
) -> str:
    """
    Build a prompt for batch grading all copies at once.

    Args:
        copies_data: List of copy data with images and metadata
        questions: Dict of {question_id: {text, criteria, max_points}}
        language: Language for prompts

    Returns:
        Complete prompt string
    """
    if language == "fr":
        return _build_batch_prompt_fr(copies_data, questions)
    else:
        return _build_batch_prompt_en(copies_data, questions)


def _build_batch_prompt_fr(
    copies_data: List[Dict[str, Any]],
    questions: Dict[str, Dict[str, Any]]
) -> str:
    """Build batch grading prompt in French."""

    # Build questions section
    questions_text = ""
    for qid, qdata in questions.items():
        questions_text += f"""
## {qid}
**Question:** {qdata.get('text', 'Non spécifiée')}
**Critères:** {qdata.get('criteria', 'Non spécifiés')}
**Barème:** {qdata.get('max_points', 1)} point(s)

"""

    prompt = f"""Tu es un correcteur expérimenté. Tu dois corriger {len(copies_data)} copies d'élèves en UNE SEULE analyse.

Cette approche te permet de:
- Garantir la COHÉRENCE: même réponse = même note
- Détecter les PATTERNS: réponses courantes, outliers, copiage potentiel
- Être EFFICACE: tout en un seul passage

═══════════════════════════════════════════════════════════════════

# BARÈME DE NOTATION

{questions_text}

═══════════════════════════════════════════════════════════════════

# COPIES À CORRIGER

Tu vas recevoir {len(copies_data)} copies. Pour CHAQUE copie:
1. Lis le nom de l'élève (si présent)
2. Lis les réponses à chaque question
3. Note selon le barème
4. Donne un feedback sobre et professionnel

═══════════════════════════════════════════════════════════════════

# RÈGLES IMPORTANTES

1. **COHÉRENCE ABSOLUE**: Si deux élèves ont écrit la même réponse, ils doivent avoir la même note
2. **LECTURE ATTENTIVE**: Utilise le CONTEXTE (question, autres copies, cohérence) pour déchiffrer l'écriture manuscrite
3. **FEEDBACK SOBRE**: Commentaire court, constructif, adapté à la difficulté
4. **DÉTECTION PATTERNS**: Note si beaucoup d'élèves ont la même réponse (correcte ou non)
5. **CROISEMENT**: Comparer les réponses entre copies t'aide à lire l'écriture et assurer la cohérence

═══════════════════════════════════════════════════════════════════

# FORMAT DE RÉPONSE (JSON)

```json
{{
  "copies": [
    {{
      "copy_index": 1,
      "student_name": "Nom de l'élève ou null",
      "questions": {{
        "Q1": {{
          "student_answer_read": "Ce que l'élève a écrit",
          "grade": 1.0,
          "max_points": 1.0,
          "confidence": 0.95,
          "reasoning": "Pourquoi cette note",
          "feedback": "Feedback sobre"
        }},
        "Q2": {{ ... }}
      }},
      "overall_feedback": "Commentaire général sur la copie"
    }},
    {{
      "copy_index": 2,
      ...
    }}
  ],
  "patterns": {{
    "Q1": {{
      "common_answer": "fiole jaugée",
      "frequency": "75%",
      "is_correct": true
    }},
    "suspicious_similarities": [
      {{"copies": [3, 7], "question": "Q4", "reason": "Réponses identiques mot pour mot"}}
    ]
  }}
}}
```

═══════════════════════════════════════════════════════════════════

Analyse maintenant les {len(copies_data)} copies fournies et retourne ta correction au format JSON.
"""
    return prompt


def _build_batch_prompt_en(
    copies_data: List[Dict[str, Any]],
    questions: Dict[str, Dict[str, Any]]
) -> str:
    """Build batch grading prompt in English."""

    # Build questions section
    questions_text = ""
    for qid, qdata in questions.items():
        questions_text += f"""
## {qid}
**Question:** {qdata.get('text', 'Not specified')}
**Criteria:** {qdata.get('criteria', 'Not specified')}
**Max Points:** {qdata.get('max_points', 1)}

"""

    prompt = f"""You are an experienced grader. You must grade {len(copies_data)} student copies in ONE analysis.

This approach allows you to:
- Ensure CONSISTENCY: same answer = same grade
- Detect PATTERNS: common answers, outliers, potential cheating
- Be EFFICIENT: everything in one pass

═══════════════════════════════════════════════════════════════════

# GRADING RUBRIC

{questions_text}

═══════════════════════════════════════════════════════════════════

# COPIES TO GRADE

You will receive {len(copies_data)} copies. For EACH copy:
1. Read the student's name (if present)
2. Read the answers to each question
3. Grade according to the rubric
4. Provide concise, professional feedback

═══════════════════════════════════════════════════════════════════

# IMPORTANT RULES

1. **ABSOLUTE CONSISTENCY**: If two students wrote the same answer, they must get the same grade
2. **CAREFUL READING**: Use CONTEXT (question, other copies, consistency) to decipher handwriting
3. **CONCISE FEEDBACK**: Short, constructive comment adapted to difficulty
4. **PATTERN DETECTION**: Note if many students have the same answer (correct or not)
5. **CROSS-REFERENCE**: Comparing answers across copies helps you read handwriting and ensure consistency

═══════════════════════════════════════════════════════════════════

# RESPONSE FORMAT (JSON)

```json
{{
  "copies": [
    {{
      "copy_index": 1,
      "student_name": "Student name or null",
      "questions": {{
        "Q1": {{
          "student_answer_read": "What the student wrote",
          "grade": 1.0,
          "max_points": 1.0,
          "confidence": 0.95,
          "reasoning": "Why this grade",
          "feedback": "Concise feedback"
        }},
        "Q2": {{ ... }}
      }},
      "overall_feedback": "General comment on the copy"
    }},
    {{
      "copy_index": 2,
      ...
    }}
  ],
  "patterns": {{
    "Q1": {{
      "common_answer": "volumetric flask",
      "frequency": "75%",
      "is_correct": true
    }},
    "suspicious_similarities": [
      {{"copies": [3, 7], "question": "Q4", "reason": "Identical word-for-word answers"}}
    ]
  }}
}}
```

═══════════════════════════════════════════════════════════════════

Now analyze the {len(copies_data)} provided copies and return your grading in JSON format.
"""
    return prompt


class BatchGrader:
    """
    Grade all copies in batches using a single API call per batch.

    Usage:
        grader = BatchGrader(provider)
        result = await grader.grade_batch(copies, questions, language="fr")
    """

    def __init__(self, provider):
        """
        Initialize batch grader.

        Args:
            provider: LLM provider (must support multi-image calls)
        """
        self.provider = provider

    async def grade_batch(
        self,
        copies: List[Dict[str, Any]],
        questions: Dict[str, Dict[str, Any]],
        language: str = "fr"
    ) -> BatchResult:
        """
        Grade a batch of copies in a single API call.

        Args:
            copies: List of copy data dicts with:
                - copy_index: 1-based index
                - image_paths: List of image paths for this copy
                - (optional) student_name: Pre-detected name
            questions: Dict of {question_id: {text, criteria, max_points}}
            language: Language for prompts

        Returns:
            BatchResult with all copy grades and detected patterns
        """
        start_time = time.time()

        # Build prompt
        prompt = build_batch_grading_prompt(copies, questions, language)

        # Collect all images from all copies
        all_images = []
        for copy in copies:
            all_images.extend(copy.get('image_paths', []))

        # Call LLM with all images
        try:
            # Use call_vision directly to get raw response text
            raw_response = self.provider.call_vision(prompt, image_path=all_images)
            if not isinstance(raw_response, str):
                raw_response = str(raw_response)

        except Exception as e:
            logger.error(f"Batch grading API call failed: {e}")
            return BatchResult(
                copies=[],
                patterns={},
                raw_response=str(e),
                parse_success=False,
                parse_errors=[f"API call failed: {str(e)}"],
                duration_ms=(time.time() - start_time) * 1000
            )

        # Parse response
        result = self._parse_batch_response(raw_response, copies, start_time)

        return result

    def _parse_batch_response(
        self,
        raw_response: str,
        copies: List[Dict[str, Any]],
        start_time: float
    ) -> BatchResult:
        """Parse the LLM response into structured BatchResult."""

        parse_errors = []
        copies_results = []
        patterns = {}

        # Try to extract JSON from response
        try:
            # Find JSON in response
            json_match = raw_response
            if '```json' in raw_response:
                json_start = raw_response.find('```json') + 7
                json_end = raw_response.find('```', json_start)
                json_match = raw_response[json_start:json_end].strip()
            elif '```' in raw_response:
                json_start = raw_response.find('```') + 3
                json_end = raw_response.find('```', json_start)
                json_match = raw_response[json_start:json_end].strip()

            # Find the main JSON object
            brace_start = json_match.find('{')
            brace_end = json_match.rfind('}') + 1
            if brace_start >= 0 and brace_end > brace_start:
                json_str = json_match[brace_start:brace_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON object found in response")

            # Parse copies
            for copy_data in data.get('copies', []):
                copy_index = copy_data.get('copy_index', 0)
                student_name = copy_data.get('student_name')

                # Parse questions
                questions = {}
                for qid, qdata in copy_data.get('questions', {}).items():
                    questions[qid] = {
                        'student_answer_read': qdata.get('student_answer_read', ''),
                        'grade': float(qdata.get('grade', 0)),
                        'max_points': float(qdata.get('max_points', 1)),
                        'confidence': float(qdata.get('confidence', 0.8)),
                        'reasoning': qdata.get('reasoning', ''),
                        'feedback': qdata.get('feedback', '')
                    }

                copies_results.append(BatchCopyResult(
                    copy_index=copy_index,
                    student_name=student_name,
                    questions=questions,
                    overall_feedback=copy_data.get('overall_feedback', '')
                ))

            # Parse patterns
            patterns = data.get('patterns', {})

        except json.JSONDecodeError as e:
            parse_errors.append(f"JSON parsing error: {str(e)}")
        except Exception as e:
            parse_errors.append(f"Parsing error: {str(e)}")

        # Ensure we have results for all input copies
        if len(copies_results) < len(copies):
            parse_errors.append(
                f"Only {len(copies_results)}/{len(copies)} copies parsed"
            )

        return BatchResult(
            copies=copies_results,
            patterns=patterns,
            raw_response=raw_response[:5000] if raw_response else "",
            parse_success=len(parse_errors) == 0 and len(copies_results) > 0,
            parse_errors=parse_errors,
            duration_ms=(time.time() - start_time) * 1000
        )


async def grade_all_copies_in_batches(
    provider,
    copies: List[Dict[str, Any]],
    questions: Dict[str, Dict[str, Any]],
    max_pages_per_batch: int = 0,
    pages_per_copy: int = 2,
    language: str = "fr",
    progress_callback=None
) -> List[BatchResult]:
    """
    Grade all copies in batches.

    Args:
        provider: LLM provider
        copies: List of all copies to grade
        questions: Question definitions
        max_pages_per_batch: Max pages per batch (0 = no limit, all in one batch)
        pages_per_copy: Number of pages per copy (for calculating batch size)
        language: Language for prompts
        progress_callback: Optional callback for progress updates

    Returns:
        List of BatchResult, one per batch
    """
    grader = BatchGrader(provider)
    results = []

    # If no limit, process all copies in one batch
    if max_pages_per_batch <= 0:
        batches = [copies]
    else:
        # Split into batches based on page count
        batches = []
        current_batch = []
        current_pages = 0

        for copy in copies:
            copy_pages = pages_per_copy  # Each copy has this many pages

            if current_pages + copy_pages > max_pages_per_batch and current_batch:
                # Start new batch
                batches.append(current_batch)
                current_batch = []
                current_pages = 0

            current_batch.append(copy)
            current_pages += copy_pages

        if current_batch:
            batches.append(current_batch)

    total_batches = len(batches)

    for batch_idx, batch_copies in enumerate(batches):
        if progress_callback:
            await progress_callback('batch_start', {
                'batch_index': batch_idx + 1,
                'total_batches': total_batches,
                'copies_in_batch': len(batch_copies)
            })

        result = await grader.grade_batch(batch_copies, questions, language)
        results.append(result)

        if progress_callback:
            await progress_callback('batch_done', {
                'batch_index': batch_idx + 1,
                'total_batches': total_batches,
                'success': result.parse_success
            })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# POST-BATCH VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Disagreement:
    """Represents a disagreement between two LLMs on a specific question."""
    copy_index: int
    question_id: str
    llm1_name: str
    llm1_grade: float
    llm1_reasoning: str
    llm1_reading: str
    llm2_name: str
    llm2_grade: float
    llm2_reasoning: str
    llm2_reading: str
    difference: float
    max_points: float  # For calculating relative thresholds
    image_paths: List[str]  # Paths to the copy images


def detect_disagreements(
    llm1_result: BatchResult,
    llm2_result: BatchResult,
    llm1_name: str,
    llm2_name: str,
    copies_data: List[Dict[str, Any]],
    threshold: float = 0.10
) -> List[Disagreement]:
    """
    Detect disagreements between two LLM batch results.

    Args:
        llm1_result: First LLM's batch result
        llm2_result: Second LLM's batch result
        llm1_name: Name of first LLM
        llm2_name: Name of second LLM
        copies_data: Original copies data with image paths
        threshold: Minimum difference as percentage of max_points (default 10%)

    Returns:
        List of Disagreement objects
    """
    disagreements = []

    # Build lookup for LLM2 results
    llm2_copies = {c.copy_index: c for c in llm2_result.copies}

    for llm1_copy in llm1_result.copies:
        copy_idx = llm1_copy.copy_index
        llm2_copy = llm2_copies.get(copy_idx)

        if not llm2_copy:
            continue

        # Get image paths for this copy
        copy_data = next((c for c in copies_data if c['copy_index'] == copy_idx), None)
        image_paths = copy_data.get('image_paths', []) if copy_data else []

        # Check each question
        for qid, q1_data in llm1_copy.questions.items():
            q2_data = llm2_copy.questions.get(qid)
            if not q2_data:
                continue

            grade1 = float(q1_data.get('grade', 0))
            grade2 = float(q2_data.get('grade', 0))
            max_points = max(
                float(q1_data.get('max_points', 1.0)),
                float(q2_data.get('max_points', 1.0))
            )
            diff = abs(grade1 - grade2)

            # Use relative threshold (percentage of max_points)
            relative_threshold = max_points * threshold

            if diff >= relative_threshold:
                disagreements.append(Disagreement(
                    copy_index=copy_idx,
                    question_id=qid,
                    llm1_name=llm1_name,
                    llm1_grade=grade1,
                    llm1_reasoning=q1_data.get('reasoning', ''),
                    llm1_reading=q1_data.get('student_answer_read', ''),
                    llm2_name=llm2_name,
                    llm2_grade=grade2,
                    llm2_reasoning=q2_data.get('reasoning', ''),
                    llm2_reading=q2_data.get('student_answer_read', ''),
                    difference=diff,
                    max_points=max_points,
                    image_paths=image_paths
                ))

    return disagreements


def build_dual_llm_verification_prompt(
    disagreements: List[Disagreement],
    provider_name: str,
    other_provider_name: str,
    is_own_perspective: bool = True,
    language: str = "fr"
) -> str:
    """
    Build a verification prompt for ONE LLM showing both LLMs' grades.

    Args:
        disagreements: List of disagreements to verify
        provider_name: Name of the LLM receiving this prompt
        other_provider_name: Name of the other LLM
        is_own_perspective: If True, this LLM's grades are shown as "you"
        language: Language for prompts

    Returns:
        Complete prompt string
    """
    if language == "fr":
        return _build_dual_llm_verification_prompt_fr(
            disagreements, provider_name, other_provider_name, is_own_perspective
        )
    else:
        return _build_dual_llm_verification_prompt_en(
            disagreements, provider_name, other_provider_name, is_own_perspective
        )


def _build_dual_llm_verification_prompt_fr(
    disagreements: List[Disagreement],
    provider_name: str,
    other_provider_name: str,
    is_own_perspective: bool
) -> str:
    """Build dual LLM verification prompt in French."""

    # Build disagreements text from this LLM's perspective
    disagreements_text = ""
    for i, d in enumerate(disagreements, 1):
        if is_own_perspective:
            # This LLM's grades shown as "tu"
            your_grade = d.llm1_grade if d.llm1_name == provider_name else d.llm2_grade
            your_reading = d.llm1_reading if d.llm1_name == provider_name else d.llm2_reading
            your_reasoning = d.llm1_reasoning if d.llm1_name == provider_name else d.llm2_reasoning
            other_grade = d.llm2_grade if d.llm1_name == provider_name else d.llm1_grade
            other_reading = d.llm2_reading if d.llm1_name == provider_name else d.llm1_reading
            other_reasoning = d.llm2_reasoning if d.llm1_name == provider_name else d.llm1_reasoning
        else:
            your_grade = d.llm2_grade
            your_reading = d.llm2_reading
            your_reasoning = d.llm2_reasoning
            other_grade = d.llm1_grade
            other_reading = d.llm1_reading
            other_reasoning = d.llm1_reasoning

        disagreements_text += f"""
## Désaccord {i}: Copie {d.copy_index}, {d.question_id}

**TOI ({provider_name})** as donné: **{your_grade}** pts
- Ta lecture: "{your_reading}"
- Ton raisonnement: {your_reasoning}

**L'AUTRE IA ({other_provider_name})** a donné: **{other_grade}** pts
- Sa lecture: "{other_reading}"
- Son raisonnement: {other_reasoning}

Écart: {abs(your_grade - other_grade)} points

"""

    prompt = f"""Tu as corrigé des copies et un DÉSACCORD a été détecté avec un autre correcteur IA.

Tu dois maintenant RÉEXAMINER ta correction en tenant compte de l'avis de l'autre IA.

═══════════════════════════════════════════════════════════════════

# DÉSACCORDS À RÉEXAMINER

{disagreements_text}

═══════════════════════════════════════════════════════════════════

# TA MISSION

Pour chaque désaccord:
1. RELIS l'image de la copie attentivement
2. COMPARE ta lecture avec celle de l'autre IA
3. DÉCIDE si tu maintiens ta note ou si tu l'ajustes
4. JUSTIFIE ta décision

**IMPORTANT**: Tu peux:
- Maintenir ta note si tu es sûr de toi
- Ajuster ta note si l'autre IA t'a fait voir quelque chose que tu as manqué
- Changer complètement si tu realizes une erreur

═══════════════════════════════════════════════════════════════════

# FORMAT DE RÉPONSE (JSON)

```json
{{
  "verifications": [
    {{
      "copy_index": 1,
      "question_id": "Q1",
      "my_initial_grade": 1.0,
      "my_new_grade": 0.5,
      "changed": true,
      "reasoning": "Pourquoi j'ai changé/maintenu ma note",
      "confidence": 0.9
    }},
    ...
  ]
}}
```

═══════════════════════════════════════════════════════════════════

Réexamine les copies et retourne ta décision au format JSON.
"""
    return prompt


def _build_dual_llm_verification_prompt_en(
    disagreements: List[Disagreement],
    provider_name: str,
    other_provider_name: str,
    is_own_perspective: bool
) -> str:
    """Build dual LLM verification prompt in English."""

    disagreements_text = ""
    for i, d in enumerate(disagreements, 1):
        if is_own_perspective:
            your_grade = d.llm1_grade if d.llm1_name == provider_name else d.llm2_grade
            your_reading = d.llm1_reading if d.llm1_name == provider_name else d.llm2_reading
            your_reasoning = d.llm1_reasoning if d.llm1_name == provider_name else d.llm2_reasoning
            other_grade = d.llm2_grade if d.llm1_name == provider_name else d.llm1_grade
            other_reading = d.llm2_reading if d.llm1_name == provider_name else d.llm1_reading
            other_reasoning = d.llm2_reasoning if d.llm1_name == provider_name else d.llm1_reasoning
        else:
            your_grade = d.llm2_grade
            your_reading = d.llm2_reading
            your_reasoning = d.llm2_reasoning
            other_grade = d.llm1_grade
            other_reading = d.llm1_reading
            other_reasoning = d.llm1_reasoning

        disagreements_text += f"""
## Disagreement {i}: Copy {d.copy_index}, {d.question_id}

**YOU ({provider_name})** gave: **{your_grade}** pts
- Your reading: "{your_reading}"
- Your reasoning: {your_reasoning}

**THE OTHER AI ({other_provider_name})** gave: **{other_grade}** pts
- Their reading: "{other_reading}"
- Their reasoning: {other_reasoning}

Difference: {abs(your_grade - other_grade)} points

"""

    prompt = f"""You graded copies and a DISAGREEMENT was detected with another AI grader.

You must now REEXAMINE your grading considering the other AI's opinion.

═══════════════════════════════════════════════════════════════════

# DISAGREEMENTS TO REEXAMINE

{disagreements_text}

═══════════════════════════════════════════════════════════════════

# YOUR MISSION

For each disagreement:
1. REREAD the copy image carefully
2. COMPARE your reading with the other AI's
3. DECIDE whether to maintain or adjust your grade
4. JUSTIFY your decision

**IMPORTANT**: You can:
- Maintain your grade if you're confident
- Adjust your grade if the other AI pointed out something you missed
- Change completely if you realize an error

═══════════════════════════════════════════════════════════════════

# RESPONSE FORMAT (JSON)

```json
{{
  "verifications": [
    {{
      "copy_index": 1,
      "question_id": "Q1",
      "my_initial_grade": 1.0,
      "my_new_grade": 0.5,
      "changed": true,
      "reasoning": "Why I changed/maintained my grade",
      "confidence": 0.9
    }},
    ...
  ]
}}
```

═══════════════════════════════════════════════════════════════════

Reexamine the copies and return your decision in JSON format.
"""
    return prompt


async def run_dual_llm_verification(
    providers: List[tuple],
    disagreements: List[Disagreement],
    language: str = "fr"
) -> Dict[str, Dict[str, Any]]:
    """
    Run verification with BOTH LLMs seeing each other's work.

    Args:
        providers: List of (name, provider) tuples
        disagreements: List of disagreements to verify
        language: Language for prompts

    Returns:
        Dict mapping "copy_{idx}_{qid}" -> {final_grade, llm1_grade, llm2_grade, method, ...}
    """
    if not disagreements:
        return {}

    llm1_name, llm1_provider = providers[0]
    llm2_name, llm2_provider = providers[1]

    # Collect all unique image paths
    all_images = []
    seen = set()
    for d in disagreements:
        for img in d.image_paths:
            if img not in seen:
                all_images.append(img)
                seen.add(img)

    # Build prompts for each LLM
    llm1_prompt = build_dual_llm_verification_prompt(
        disagreements, llm1_name, llm2_name, is_own_perspective=True, language=language
    )
    llm2_prompt = build_dual_llm_verification_prompt(
        disagreements, llm2_name, llm1_name, is_own_perspective=True, language=language
    )

    # Call both LLMs in parallel
    async def call_provider(provider, prompt):
        try:
            raw_response = provider.call_vision(prompt, image_path=all_images)
            if not isinstance(raw_response, str):
                raw_response = str(raw_response)
            return raw_response
        except Exception as e:
            logger.error(f"Verification call failed: {e}")
            return None

    llm1_response, llm2_response = await asyncio.gather(
        call_provider(llm1_provider, llm1_prompt),
        call_provider(llm2_provider, llm2_prompt)
    )

    # Parse responses
    llm1_results = _parse_verification_response(llm1_response)
    llm2_results = _parse_verification_response(llm2_response)

    # Merge results
    results = {}
    for d in disagreements:
        key = f"copy_{d.copy_index}_{d.question_id}"

        # Get new grades from each LLM
        llm1_new = llm1_results.get(key, {}).get('my_new_grade', d.llm1_grade)
        llm2_new = llm2_results.get(key, {}).get('my_new_grade', d.llm2_grade)

        # Resolve: if both agree now, use that; otherwise average
        if abs(llm1_new - llm2_new) < get_agreement_threshold(d.max_points):
            final_grade = (llm1_new + llm2_new) / 2
            method = "verification_consensus"
        else:
            final_grade = (llm1_new + llm2_new) / 2
            method = "verification_average"

        results[key] = {
            'final_grade': final_grade,
            'llm1_new_grade': llm1_new,
            'llm2_new_grade': llm2_new,
            'llm1_reasoning': llm1_results.get(key, {}).get('reasoning', ''),
            'llm2_reasoning': llm2_results.get(key, {}).get('reasoning', ''),
            'method': method,
            'max_points': d.max_points,
            'confidence': max(
                llm1_results.get(key, {}).get('confidence', 0.8),
                llm2_results.get(key, {}).get('confidence', 0.8)
            )
        }

    return results


async def run_per_question_dual_verification(
    providers: List[tuple],
    disagreements: List[Disagreement],
    language: str = "fr"
) -> Dict[str, Dict[str, Any]]:
    """
    Run verification for each disagreement separately with BOTH LLMs.

    Unlike run_dual_llm_verification (which groups all disagreements),
    this makes one call per disagreement per LLM (more granular).

    Args:
        providers: List of (name, provider) tuples
        disagreements: List of disagreements to verify
        language: Language for prompts

    Returns:
        Dict mapping "copy_{idx}_{qid}" -> {final_grade, llm1_grade, llm2_grade, method, ...}
    """
    if not disagreements:
        return {}

    llm1_name, llm1_provider = providers[0]
    llm2_name, llm2_provider = providers[1]

    results = {}

    # Process each disagreement separately
    for d in disagreements:
        key = f"copy_{d.copy_index}_{d.question_id}"

        # Build prompts for this single disagreement
        llm1_prompt = build_dual_llm_verification_prompt(
            [d], llm1_name, llm2_name, is_own_perspective=True, language=language
        )
        llm2_prompt = build_dual_llm_verification_prompt(
            [d], llm2_name, llm1_name, is_own_perspective=True, language=language
        )

        # Call both LLMs in parallel for this disagreement
        async def call_provider(provider, prompt, images):
            try:
                raw_response = provider.call_vision(prompt, image_path=images)
                if not isinstance(raw_response, str):
                    raw_response = str(raw_response)
                return raw_response
            except Exception as e:
                logger.error(f"Per-question verification call failed: {e}")
                return None

        llm1_response, llm2_response = await asyncio.gather(
            call_provider(llm1_provider, llm1_prompt, d.image_paths),
            call_provider(llm2_provider, llm2_prompt, d.image_paths)
        )

        # Parse responses
        llm1_results = _parse_verification_response(llm1_response)
        llm2_results = _parse_verification_response(llm2_response)

        # Get new grades from each LLM
        llm1_new = llm1_results.get(key, {}).get('my_new_grade', d.llm1_grade)
        llm2_new = llm2_results.get(key, {}).get('my_new_grade', d.llm2_grade)

        # Resolve: if both agree now, use that; otherwise average
        if abs(llm1_new - llm2_new) < get_agreement_threshold(d.max_points):
            final_grade = (llm1_new + llm2_new) / 2
            method = "verification_consensus"
        else:
            final_grade = (llm1_new + llm2_new) / 2
            method = "verification_average"

        results[key] = {
            'final_grade': final_grade,
            'llm1_new_grade': llm1_new,
            'llm2_new_grade': llm2_new,
            'llm1_reasoning': llm1_results.get(key, {}).get('reasoning', ''),
            'llm2_reasoning': llm2_results.get(key, {}).get('reasoning', ''),
            'method': method,
            'max_points': d.max_points,
            'confidence': max(
                llm1_results.get(key, {}).get('confidence', 0.8),
                llm2_results.get(key, {}).get('confidence', 0.8)
            ),
            'image_paths': d.image_paths  # Keep for potential ultimatum
        }

    return results


def _parse_verification_response(raw_response: str) -> Dict[str, Dict]:
    """Parse a verification response from an LLM."""
    results = {}
    if not raw_response:
        return results

    try:
        json_match = raw_response
        if '```json' in raw_response:
            json_start = raw_response.find('```json') + 7
            json_end = raw_response.find('```', json_start)
            json_match = raw_response[json_start:json_end].strip()
        elif '```' in raw_response:
            json_start = raw_response.find('```') + 3
            json_end = raw_response.find('```', json_start)
            json_match = raw_response[json_start:json_end].strip()

        brace_start = json_match.find('{')
        brace_end = json_match.rfind('}') + 1
        if brace_start >= 0 and brace_end > brace_start:
            json_str = json_match[brace_start:brace_end]
            data = json.loads(json_str)

            for v in data.get('verifications', []):
                key = f"copy_{v.get('copy_index')}_{v.get('question_id')}"
                results[key] = {
                    'my_new_grade': float(v.get('my_new_grade', v.get('my_initial_grade', 0))),
                    'changed': v.get('changed', False),
                    'reasoning': v.get('reasoning', ''),
                    'confidence': float(v.get('confidence', 0.8))
                }
    except Exception as e:
        logger.error(f"Failed to parse verification response: {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ULTIMATUM ROUND
# ═══════════════════════════════════════════════════════════════════════════════

def build_ultimatum_prompt(
    disagreements: List[Dict[str, Any]],
    provider_name: str,
    other_provider_name: str,
    language: str = "fr"
) -> str:
    """
    Build an ultimatum prompt for ONE LLM showing persistent disagreements.

    Args:
        disagreements: List of disagreements after verification
        provider_name: Name of the LLM receiving this prompt
        other_provider_name: Name of the other LLM
        language: Language for prompts

    Returns:
        Complete prompt string
    """
    if language == "fr":
        return _build_ultimatum_prompt_fr(disagreements, provider_name, other_provider_name)
    else:
        return _build_ultimatum_prompt_en(disagreements, provider_name, other_provider_name)


def _build_ultimatum_prompt_fr(
    disagreements: List[Dict[str, Any]],
    provider_name: str,
    other_provider_name: str
) -> str:
    """Build ultimatum prompt in French."""

    disagreements_text = ""
    for i, d in enumerate(disagreements, 1):
        disagreements_text += f"""
## ULTIMATUM {i}: Copie {d['copy_index']}, {d['question_id']}

**TOI ({provider_name})** après vérification: **{d['llm1_grade']}** pts
- Ton raisonnement: {d.get('llm1_reasoning', '')}

**L'AUTRE IA ({other_provider_name})** après vérification: **{d['llm2_grade']}** pts
- Son raisonnement: {d.get('llm2_reasoning', '')}

Écart persistant: {abs(d['llm1_grade'] - d['llm2_grade'])} points

"""

    prompt = f"""═══════════════════════════════════════════════════════════════════
ULTIMATUM - DÉCISION FINALE
═══════════════════════════════════════════════════════════════════

Malgré la vérification croisée, le désaccord PERSISTE avec l'autre IA.
Tu dois maintenant prendre une DÉCISION FINALE pour chaque cas.

{disagreements_text}

═══════════════════════════════════════════════════════════════════

# RÈGLES DE L'ULTIMATUM

1. **DÉCISION OBLIGATOIRE**: Tu DOIS choisir ta note finale
2. **OPTION A - Maintenir**: Si tu es sûr de toi, garde ta note
3. **OPTION B - Céder**: Si l'autre IA t'a convaincu, accepte sa note
4. **OPTION C - Compromis**: Propose une note intermédiaire justifiée

**ATTENTION**:
- Si tu es INCERTAIN, abaisse ta confiance (< 0.5)
- INTERDICTION de choisir au hasard
- Tu DOIS justifier ta décision finale

═══════════════════════════════════════════════════════════════════

# FORMAT DE RÉPONSE (JSON)

```json
{{
  "ultimatum_decisions": [
    {{
      "copy_index": 1,
      "question_id": "Q1",
      "my_final_grade": 0.5,
      "decision": "maintained" ou "yielded" ou "compromise",
      "reasoning": "Pourquoi j'ai pris cette décision finale",
      "confidence": 0.9
    }},
    ...
  ]
}}
```

═══════════════════════════════════════════════════════════════════

Relis les copies et prends ta DÉCISION FINALE au format JSON.
"""
    return prompt


def _build_ultimatum_prompt_en(
    disagreements: List[Dict[str, Any]],
    provider_name: str,
    other_provider_name: str
) -> str:
    """Build ultimatum prompt in English."""

    disagreements_text = ""
    for i, d in enumerate(disagreements, 1):
        disagreements_text += f"""
## ULTIMATUM {i}: Copy {d['copy_index']}, {d['question_id']}

**YOU ({provider_name})** after verification: **{d['llm1_grade']}** pts
- Your reasoning: {d.get('llm1_reasoning', '')}

**THE OTHER AI ({other_provider_name})** after verification: **{d['llm2_grade']}** pts
- Their reasoning: {d.get('llm2_reasoning', '')}

Persistent difference: {abs(d['llm1_grade'] - d['llm2_grade'])} points

"""

    prompt = f"""═══════════════════════════════════════════════════════════════════
ULTIMATUM - FINAL DECISION
═══════════════════════════════════════════════════════════════════

Despite cross-verification, the disagreement PERSISTS with the other AI.
You must now make a FINAL DECISION for each case.

{disagreements_text}

═══════════════════════════════════════════════════════════════════

# ULTIMATUM RULES

1. **MANDATORY DECISION**: You MUST choose your final grade
2. **OPTION A - Maintain**: If you're confident, keep your grade
3. **OPTION B - Yield**: If the other AI convinced you, accept their grade
4. **OPTION C - Compromise**: Propose an intermediate justified grade

**WARNING**:
- If UNCERTAIN, lower your confidence (< 0.5)
- FORBIDDEN to choose randomly
- You MUST justify your final decision

═══════════════════════════════════════════════════════════════════

# RESPONSE FORMAT (JSON)

```json
{{
  "ultimatum_decisions": [
    {{
      "copy_index": 1,
      "question_id": "Q1",
      "my_final_grade": 0.5,
      "decision": "maintained" or "yielded" or "compromise",
      "reasoning": "Why I made this final decision",
      "confidence": 0.9
    }},
    ...
  ]
}}
```

═══════════════════════════════════════════════════════════════════

Reread the copies and make your FINAL DECISION in JSON format.
"""
    return prompt


async def run_dual_llm_ultimatum(
    providers: List[tuple],
    persistent_disagreements: List[Dict[str, Any]],
    language: str = "fr"
) -> Dict[str, Dict[str, Any]]:
    """
    Run ultimatum round with BOTH LLMs for persistent disagreements.

    Args:
        providers: List of (name, provider) tuples
        persistent_disagreements: List of disagreements that persist after verification
        language: Language for prompts

    Returns:
        Dict mapping "copy_{idx}_{qid}" -> {final_grade, llm1_grade, llm2_grade, method, ...}
    """
    if not persistent_disagreements:
        return {}

    llm1_name, llm1_provider = providers[0]
    llm2_name, llm2_provider = providers[1]

    # Collect all unique image paths
    all_images = []
    seen = set()
    for d in persistent_disagreements:
        for img in d.get('image_paths', []):
            if img not in seen:
                all_images.append(img)
                seen.add(img)

    # Build ultimatum prompts for each LLM
    llm1_prompt = build_ultimatum_prompt(
        persistent_disagreements, llm1_name, llm2_name, language=language
    )
    llm2_prompt = build_ultimatum_prompt(
        persistent_disagreements, llm2_name, llm1_name, language=language
    )

    # Call both LLMs in parallel
    async def call_provider(provider, prompt):
        try:
            raw_response = provider.call_vision(prompt, image_path=all_images)
            if not isinstance(raw_response, str):
                raw_response = str(raw_response)
            return raw_response
        except Exception as e:
            logger.error(f"Ultimatum call failed: {e}")
            return None

    llm1_response, llm2_response = await asyncio.gather(
        call_provider(llm1_provider, llm1_prompt),
        call_provider(llm2_provider, llm2_prompt)
    )

    # Parse responses
    llm1_results = _parse_ultimatum_response(llm1_response)
    llm2_results = _parse_ultimatum_response(llm2_response)

    # Merge results
    results = {}
    for d in persistent_disagreements:
        key = f"copy_{d['copy_index']}_{d['question_id']}"

        # Get final grades from each LLM
        llm1_final = llm1_results.get(key, {}).get('my_final_grade', d['llm1_grade'])
        llm2_final = llm2_results.get(key, {}).get('my_final_grade', d['llm2_grade'])

        # Get max_points for relative threshold calculation
        max_pts = d.get('max_points', 1.0)

        # Resolve: if both agree now, use that; otherwise average
        if abs(llm1_final - llm2_final) < get_agreement_threshold(max_pts):
            final_grade = (llm1_final + llm2_final) / 2
            method = "ultimatum_consensus"
        else:
            final_grade = (llm1_final + llm2_final) / 2
            method = "ultimatum_average"

        # Detect flip-flop (grades swapped sides)
        initial_diff = d['llm1_grade'] - d['llm2_grade']
        ultimatum_diff = llm1_final - llm2_final
        is_swap = (
            (initial_diff > 0 and ultimatum_diff < 0) or
            (initial_diff < 0 and ultimatum_diff > 0)
        )
        # Use configurable threshold (0 = detect any swap)
        significant_diff = get_flip_flop_threshold(max_pts)
        flip_flop = (
            is_swap and
            abs(initial_diff) >= significant_diff and
            abs(ultimatum_diff) >= significant_diff
        )

        results[key] = {
            'final_grade': final_grade,
            'llm1_final_grade': llm1_final,
            'llm2_final_grade': llm2_final,
            'llm1_decision': llm1_results.get(key, {}).get('decision', 'unknown'),
            'llm2_decision': llm2_results.get(key, {}).get('decision', 'unknown'),
            'llm1_reasoning': llm1_results.get(key, {}).get('reasoning', ''),
            'llm2_reasoning': llm2_results.get(key, {}).get('reasoning', ''),
            'method': method,
            'flip_flop_detected': flip_flop,
            'confidence': max(
                llm1_results.get(key, {}).get('confidence', 0.8),
                llm2_results.get(key, {}).get('confidence', 0.8)
            )
        }

    return results


def _parse_ultimatum_response(raw_response: str) -> Dict[str, Dict]:
    """Parse an ultimatum response from an LLM."""
    results = {}
    if not raw_response:
        return results

    try:
        json_match = raw_response
        if '```json' in raw_response:
            json_start = raw_response.find('```json') + 7
            json_end = raw_response.find('```', json_start)
            json_match = raw_response[json_start:json_end].strip()
        elif '```' in raw_response:
            json_start = raw_response.find('```') + 3
            json_end = raw_response.find('```', json_start)
            json_match = raw_response[json_start:json_end].strip()

        brace_start = json_match.find('{')
        brace_end = json_match.rfind('}') + 1
        if brace_start >= 0 and brace_end > brace_start:
            json_str = json_match[brace_start:brace_end]
            data = json.loads(json_str)

            for u in data.get('ultimatum_decisions', []):
                key = f"copy_{u.get('copy_index')}_{u.get('question_id')}"
                results[key] = {
                    'my_final_grade': float(u.get('my_final_grade', 0)),
                    'decision': u.get('decision', 'unknown'),
                    'reasoning': u.get('reasoning', ''),
                    'confidence': float(u.get('confidence', 0.8))
                }
    except Exception as e:
        logger.error(f"Failed to parse ultimatum response: {e}")

    return results


# Keep the old functions for backward compatibility but mark as deprecated
def build_grouped_verification_prompt(
    disagreements: List[Disagreement],
    language: str = "fr"
) -> str:
    """
    Build a prompt for verifying ALL disagreements in one call.

    Args:
        disagreements: List of disagreements to verify
        language: Language for prompts

    Returns:
        Complete prompt string
    """
    if language == "fr":
        return _build_grouped_verification_prompt_fr(disagreements)
    else:
        return _build_grouped_verification_prompt_en(disagreements)


def _build_grouped_verification_prompt_fr(disagreements: List[Disagreement]) -> str:
    """Build grouped verification prompt in French."""

    # Build disagreements summary
    disagreements_text = ""
    for i, d in enumerate(disagreements, 1):
        disagreements_text += f"""
## Désaccord {i}: Copie {d.copy_index}, {d.question_id}

**{d.llm1_name}** a donné: **{d.llm1_grade}** pts
- Lecture: "{d.llm1_reading}"
- Raisonnement: {d.llm1_reasoning}

**{d.llm2_name}** a donné: **{d.llm2_grade}** pts
- Lecture: "{d.llm2_reading}"
- Raisonnement: {d.llm2_reasoning}

Écart: {d.difference} points

"""

    prompt = f"""Tu es un arbitre chargé de résoudre des désaccords entre deux correcteurs IA.

Deux IA ont corrigé les mêmes copies mais ont des notes différentes sur certains points.
Tu dois examiner chaque désaccord et proposer une note finale justifiée.

═══════════════════════════════════════════════════════════════════

# DÉSACCORDS À RÉSOUDRE

{disagreements_text}

═══════════════════════════════════════════════════════════════════

# TA MISSION

Pour chaque désaccord:
1. Analyse les deux lectures et raisonnements
2. Vérifie la cohérence avec les autres copies
3. Propose une note finale avec justification

═══════════════════════════════════════════════════════════════════

# FORMAT DE RÉPONSE (JSON)

```json
{{
  "resolutions": [
    {{
      "copy_index": 1,
      "question_id": "Q1",
      "final_grade": 0.5,
      "reasoning": "Explication de ta décision",
      "confidence": 0.9
    }},
    ...
  ]
}}
```

═══════════════════════════════════════════════════════════════════

Tu as accès aux images des copies. Analyse-les et retourne ta décision au format JSON.
"""
    return prompt


def _build_grouped_verification_prompt_en(disagreements: List[Disagreement]) -> str:
    """Build grouped verification prompt in English."""

    disagreements_text = ""
    for i, d in enumerate(disagreements, 1):
        disagreements_text += f"""
## Disagreement {i}: Copy {d.copy_index}, {d.question_id}

**{d.llm1_name}** gave: **{d.llm1_grade}** pts
- Reading: "{d.llm1_reading}"
- Reasoning: {d.llm1_reasoning}

**{d.llm2_name}** gave: **{d.llm2_grade}** pts
- Reading: "{d.llm2_reading}"
- Reasoning: {d.llm2_reasoning}

Difference: {d.difference} points

"""

    prompt = f"""You are an arbitrator resolving disagreements between two AI graders.

Two AIs graded the same copies but have different scores on some points.
You must examine each disagreement and propose a final justified score.

═══════════════════════════════════════════════════════════════════

# DISAGREEMENTS TO RESOLVE

{disagreements_text}

═══════════════════════════════════════════════════════════════════

# YOUR MISSION

For each disagreement:
1. Analyze both readings and reasonings
2. Check consistency with other copies
3. Propose a final score with justification

═══════════════════════════════════════════════════════════════════

# RESPONSE FORMAT (JSON)

```json
{{
  "resolutions": [
    {{
      "copy_index": 1,
      "question_id": "Q1",
      "final_grade": 0.5,
      "reasoning": "Explanation of your decision",
      "confidence": 0.9
    }},
    ...
  ]
}}
```

═══════════════════════════════════════════════════════════════════

You have access to the copy images. Analyze them and return your decision in JSON format.
"""
    return prompt


def build_per_question_verification_prompt(
    disagreement: Disagreement,
    language: str = "fr"
) -> str:
    """
    Build a prompt for verifying a SINGLE disagreement.

    Args:
        disagreement: The disagreement to verify
        language: Language for prompts

    Returns:
        Complete prompt string
    """
    if language == "fr":
        return _build_per_question_verification_prompt_fr(disagreement)
    else:
        return _build_per_question_verification_prompt_en(disagreement)


def _build_per_question_verification_prompt_fr(disagreement: Disagreement) -> str:
    """Build per-question verification prompt in French."""

    prompt = f"""Tu es un arbitre chargé de résoudre un désaccord entre deux correcteurs IA.

═══════════════════════════════════════════════════════════════════

# DÉSACCORD

**Copie:** {disagreement.copy_index}
**Question:** {disagreement.question_id}

**{disagreement.llm1_name}** a donné: **{disagreement.llm1_grade}** pts
- Lecture: "{disagreement.llm1_reading}"
- Raisonnement: {disagreement.llm1_reasoning}

**{disagreement.llm2_name}** a donné: **{disagreement.llm2_grade}** pts
- Lecture: "{disagreement.llm2_reading}"
- Raisonnement: {disagreement.llm2_reasoning}

Écart: {disagreement.difference} points

═══════════════════════════════════════════════════════════════════

# TA MISSION

1. Regarde l'image de la copie
2. Relis ce que l'élève a écrit
3. Détermine quelle IA a la meilleure lecture et notation
4. Propose une note finale

═══════════════════════════════════════════════════════════════════

# FORMAT DE RÉPONSE (JSON)

```json
{{
  "copy_index": {disagreement.copy_index},
  "question_id": "{disagreement.question_id}",
  "final_grade": 0.5,
  "student_answer_read": "Ce que tu as lu",
  "reasoning": "Explication de ta décision",
  "confidence": 0.9,
  "preferred_llm": "{disagreement.llm1_name}" ou "{disagreement.llm2_name}" ou "neither"
}}
```

Analyse l'image et retourne ta décision au format JSON.
"""
    return prompt


def _build_per_question_verification_prompt_en(disagreement: Disagreement) -> str:
    """Build per-question verification prompt in English."""

    prompt = f"""You are an arbitrator resolving a disagreement between two AI graders.

═══════════════════════════════════════════════════════════════════

# DISAGREEMENT

**Copy:** {disagreement.copy_index}
**Question:** {disagreement.question_id}

**{disagreement.llm1_name}** gave: **{disagreement.llm1_grade}** pts
- Reading: "{disagreement.llm1_reading}"
- Reasoning: {disagreement.llm1_reasoning}

**{disagreement.llm2_name}** gave: **{disagreement.llm2_grade}** pts
- Reading: "{disagreement.llm2_reading}"
- Reasoning: {disagreement.llm2_reasoning}

Difference: {disagreement.difference} points

═══════════════════════════════════════════════════════════════════

# YOUR MISSION

1. Look at the copy image
2. Re-read what the student wrote
3. Determine which AI has the better reading and grading
4. Propose a final score

═══════════════════════════════════════════════════════════════════

# RESPONSE FORMAT (JSON)

```json
{{
  "copy_index": {disagreement.copy_index},
  "question_id": "{disagreement.question_id}",
  "final_grade": 0.5,
  "student_answer_read": "What you read",
  "reasoning": "Explanation of your decision",
  "confidence": 0.9,
  "preferred_llm": "{disagreement.llm1_name}" or "{disagreement.llm2_name}" or "neither"
}}
```

Analyze the image and return your decision in JSON format.
"""
    return prompt


async def run_grouped_verification(
    provider,
    disagreements: List[Disagreement],
    language: str = "fr"
) -> Dict[str, Dict[str, Any]]:
    """
    Run verification for all disagreements in a single API call.

    Args:
        provider: LLM provider
        disagreements: List of disagreements to verify
        language: Language for prompts

    Returns:
        Dict mapping "copy_{idx}_{qid}" -> {final_grade, reasoning, confidence}
    """
    if not disagreements:
        return {}

    # Collect all unique image paths
    all_images = []
    seen = set()
    for d in disagreements:
        for img in d.image_paths:
            if img not in seen:
                all_images.append(img)
                seen.add(img)

    # Build prompt
    prompt = build_grouped_verification_prompt(disagreements, language)

    # Call LLM
    try:
        raw_response = provider.call_vision(prompt, image_path=all_images)
        if not isinstance(raw_response, str):
            raw_response = str(raw_response)
    except Exception as e:
        logger.error(f"Grouped verification failed: {e}")
        return {}

    # Parse response
    results = {}
    try:
        # Extract JSON
        json_match = raw_response
        if '```json' in raw_response:
            json_start = raw_response.find('```json') + 7
            json_end = raw_response.find('```', json_start)
            json_match = raw_response[json_start:json_end].strip()
        elif '```' in raw_response:
            json_start = raw_response.find('```') + 3
            json_end = raw_response.find('```', json_start)
            json_match = raw_response[json_start:json_end].strip()

        brace_start = json_match.find('{')
        brace_end = json_match.rfind('}') + 1
        if brace_start >= 0 and brace_end > brace_start:
            json_str = json_match[brace_start:brace_end]
            data = json.loads(json_str)

            for resolution in data.get('resolutions', []):
                key = f"copy_{resolution.get('copy_index')}_{resolution.get('question_id')}"
                results[key] = {
                    'final_grade': float(resolution.get('final_grade', 0)),
                    'reasoning': resolution.get('reasoning', ''),
                    'confidence': float(resolution.get('confidence', 0.8))
                }
    except Exception as e:
        logger.error(f"Failed to parse grouped verification response: {e}")

    return results


async def run_per_question_verification(
    provider,
    disagreement: Disagreement,
    language: str = "fr"
) -> Dict[str, Any]:
    """
    Run verification for a single disagreement.

    Args:
        provider: LLM provider
        disagreement: The disagreement to verify
        language: Language for prompts

    Returns:
        Dict with final_grade, reasoning, confidence, student_answer_read
    """
    # Build prompt
    prompt = build_per_question_verification_prompt(disagreement, language)

    # Call LLM with only this copy's images
    try:
        raw_response = provider.call_vision(prompt, image_path=disagreement.image_paths)
        if not isinstance(raw_response, str):
            raw_response = str(raw_response)
    except Exception as e:
        logger.error(f"Per-question verification failed for copy {disagreement.copy_index}, {disagreement.question_id}: {e}")
        return {
            'final_grade': (disagreement.llm1_grade + disagreement.llm2_grade) / 2,
            'reasoning': 'Verification failed, using average',
            'confidence': 0.5
        }

    # Parse response
    try:
        json_match = raw_response
        if '```json' in raw_response:
            json_start = raw_response.find('```json') + 7
            json_end = raw_response.find('```', json_start)
            json_match = raw_response[json_start:json_end].strip()
        elif '```' in raw_response:
            json_start = raw_response.find('```') + 3
            json_end = raw_response.find('```', json_start)
            json_match = raw_response[json_start:json_end].strip()

        brace_start = json_match.find('{')
        brace_end = json_match.rfind('}') + 1
        if brace_start >= 0 and brace_end > brace_start:
            json_str = json_match[brace_start:brace_end]
            data = json.loads(json_str)
            return {
                'final_grade': float(data.get('final_grade', 0)),
                'reasoning': data.get('reasoning', ''),
                'confidence': float(data.get('confidence', 0.8)),
                'student_answer_read': data.get('student_answer_read', ''),
                'preferred_llm': data.get('preferred_llm', 'neither')
            }
    except Exception as e:
        logger.error(f"Failed to parse per-question verification response: {e}")

    return {
        'final_grade': (disagreement.llm1_grade + disagreement.llm2_grade) / 2,
        'reasoning': 'Parse failed, using average',
        'confidence': 0.5
    }
