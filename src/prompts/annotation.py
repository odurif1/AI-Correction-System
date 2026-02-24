"""
Prompts for PDF annotation coordinate detection.

These prompts are used in a post-processing step after grading
to determine optimal placement for student feedback on the PDF.
"""

from typing import Dict, Any

from utils.json_extractor import extract_json_from_response


# Combined prompt for direct coordinate assignment (single-pass)
_DIRECT_ANNOTATION_PROMPT = {
    'fr': """Tu dois placer des annotations de feedback sur cette copie d'élève.

TÂCHE:
Pour chaque question notée, indique les coordonnées optimales pour placer le feedback.

RÈGLES:
1. Le feedback doit être PROCHE de la réponse de l'élève
2. Le feedback ne doit PAS chevaucher le texte existant
3. Privilégier l'espace en dessous ou à droite de la réponse
4. Utiliser des coordonnées en pourcentage de la page
5. Les numéros de page commencent à 1 (page 1 = première page)

FEEDBACKS À PLACER:
{feedback_list}

FORMAT DE RÉPONSE (JSON):
```json
{{
  "annotations": [
    {{
      "question_id": "Q1",
      "page": 1,
      "feedback_text": "Exact.",
      "x_percent": 15.0,
      "y_percent": 25.0,
      "placement": "below_answer",
      "confidence": 0.95
    }}
  ]
}}
```

Coordonnées:
- page: Numéro de page (1 = première page)
- x_percent: Position horizontale gauche du feedback (0-100%)
- y_percent: Position verticale haute du feedback (0-100%)
- placement: "below_answer", "above_answer", "right_of_answer", "left_of_answer"
- confidence: Ta confiance (0.0-1.0)
```""",

    'en': """You must place feedback annotations on this student copy.

TASK:
For each graded question, indicate optimal coordinates to place the feedback.

RULES:
1. Feedback must be CLOSE to the student's answer
2. Feedback must NOT overlap existing text
3. Prefer space below or to the right of the answer
4. Use coordinates as percentage of page
5. Page numbers start at 1 (page 1 = first page)

FEEDBACKS TO PLACE:
{feedback_list}

RESPONSE FORMAT (JSON):
```json
{{
  "annotations": [
    {{
      "question_id": "Q1",
      "page": 1,
      "feedback_text": "Correct.",
      "x_percent": 15.0,
      "y_percent": 25.0,
      "placement": "below_answer",
      "confidence": 0.95
    }}
  ]
}}
```

Coordinates:
- page: Page number (1 = first page)
- x_percent: Horizontal left position of feedback (0-100%)
- y_percent: Vertical top position of feedback (0-100%)
- placement: "below_answer", "above_answer", "right_of_answer", "left_of_answer"
- confidence: Your confidence (0.0-1.0)
```"""
}


def build_direct_annotation_prompt(
    feedback_by_question: Dict[str, str],
    language: str = 'en'
) -> str:
    """
    Build prompt for direct annotation placement (single-pass).

    This is the recommended approach - simpler and more direct.
    """
    base_prompt = _DIRECT_ANNOTATION_PROMPT.get(
        language,
        _DIRECT_ANNOTATION_PROMPT['en']
    )

    feedback_list = "\n".join(
        f"- {q_id}: \"{feedback}\""
        for q_id, feedback in feedback_by_question.items()
    )

    return base_prompt.format(feedback_list=feedback_list)


def parse_annotation_response(response: str) -> Dict[str, Any]:
    """
    Parse LLM response for annotation coordinates.

    Args:
        response: Raw LLM response

    Returns:
        Dict with annotation data
    """
    # Try to extract JSON from response
    data = extract_json_from_response(response)
    if data is not None:
        return data

    # Fallback: return error with raw response
    return {"error": "Failed to parse JSON", "raw_response": response}
