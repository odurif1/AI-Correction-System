"""
Prompts for PDF annotation coordinate detection.

These prompts are used in a post-processing step after grading
to determine optimal placement for student feedback on the PDF.
"""

from typing import Dict, Any, List, Optional

from utils.json_extractor import extract_json_from_response


# Combined prompt for direct coordinate assignment (single-pass)
_DIRECT_ANNOTATION_PROMPT = {
    'fr': """Tu dois placer des annotations de feedback sur cette copie d'élève.

TÂCHE:
Pour chaque question notée, indique les coordonnées optimales pour placer le feedback.

RÈGLES:
1. Le feedback doit être PROCHE de la réponse de l'élève
2. Le feedback ne doit PAS chevaucher le texte existant
3. Privilégier d'abord la marge libre la plus proche de la réponse
4. S'il n'y a pas de marge proche, utiliser un espace blanc réel à droite, à gauche ou sous la réponse
5. Éviter le centre de la page s'il existe une marge libre exploitable
6. Choisir une boîte assez grande pour contenir un feedback court, sans occuper inutilement toute la page
7. Utiliser des coordonnées en pourcentage de la page
8. Les numéros de page commencent à 1 (page 1 = première page)
9. Réponds avec du JSON strict uniquement, sans commentaire

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
      "width_percent": 22.0,
      "height_percent": 7.0,
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
- width_percent: Largeur estimée du bloc (12-35%)
- height_percent: Hauteur estimée du bloc (4-18%)
- placement: "right_margin", "left_margin", "below_answer", "above_answer", "right_of_answer", "left_of_answer", "near_blank_space"
- confidence: Ta confiance (0.0-1.0)
```""",

    'en': """You must place feedback annotations on this student copy.

TASK:
For each graded question, indicate optimal coordinates to place the feedback.

RULES:
1. Feedback must be CLOSE to the student's answer
2. Feedback must NOT overlap existing text
3. Prefer the closest free margin near the answer
4. If no margin is available, use real blank space next to or below the answer
5. Avoid the center of the page when a usable margin exists
6. Choose a box large enough for a short feedback, without occupying the whole page
7. Use coordinates as percentage of page
8. Page numbers start at 1 (page 1 = first page)
9. Return strict JSON only, with no extra text

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
      "width_percent": 22.0,
      "height_percent": 7.0,
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
- width_percent: Estimated box width (12-35%)
- height_percent: Estimated box height (4-18%)
- placement: "right_margin", "left_margin", "below_answer", "above_answer", "right_of_answer", "left_of_answer", "near_blank_space"
- confidence: Your confidence (0.0-1.0)
```"""
}


def build_direct_annotation_prompt(
    feedback_by_question: Dict[str, str],
    grades_by_question: Optional[Dict[str, float]] = None,
    max_points_by_question: Optional[Dict[str, float]] = None,
    language: str = 'en',
    expected_pages_by_question: Optional[Dict[str, List[int]]] = None,
) -> str:
    """
    Build prompt for direct annotation placement (single-pass).

    This is the recommended approach - simpler and more direct.
    """
    base_prompt = _DIRECT_ANNOTATION_PROMPT.get(
        language,
        _DIRECT_ANNOTATION_PROMPT['en']
    )

    lines = []
    for q_id, feedback in feedback_by_question.items():
        page_hint = ""
        if expected_pages_by_question and expected_pages_by_question.get(q_id):
            pages = ", ".join(str(page) for page in expected_pages_by_question[q_id])
            page_hint = f" (pages probables: {pages})"
        grade_hint = ""
        if grades_by_question and q_id in grades_by_question:
            grade_hint = f", note: {grades_by_question[q_id]:.1f}"
            if max_points_by_question and q_id in max_points_by_question:
                grade_hint = f", note: {grades_by_question[q_id]:.1f}/{max_points_by_question[q_id]:.1f}"
        lines.append(f"- {q_id}{page_hint}{grade_hint}: \"{feedback}\"")

    feedback_list = "\n".join(lines)

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
