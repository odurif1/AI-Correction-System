"""
Prompts for PDF annotation coordinate detection.

These prompts are used in a post-processing step after grading
to determine optimal placement for student feedback on the PDF.
"""

from typing import Dict, List, Any, Optional
import json

from utils.json_extractor import extract_json_from_response


_ANNOTATION_COORDINATE_PROMPT = {
    'fr': """Tu dois déterminer les coordonnées optimales pour placer des annotations de feedback sur une copie d'élève.

CONTEXTE:
- Tu recevras une image de la copie annotée avec des zones numérotées (Zone 1, Zone 2, etc.)
- Pour chaque question, tu dois indiquer dans quelle zone placer le feedback
- Le feedback doit être placé PROCHE de la réponse de l'élève mais SANS la chevaucher

RÈGLES DE PLACEMENT:
1. Choisir une zone VIDE (sans texte de l'élève)
2. Privilégier les zones juste après la réponse (en dessous ou à droite)
3. Éviter les zones qui contiennent déjà du texte écrit
4. Si plusieurs zones possibles, choisir la plus proche de la réponse

FORMAT DE RÉPONSE (JSON):
```json
{
  "annotations": {
    "Q1": {
      "zone_id": "Zone 3",
      "feedback_text": "Exact.",
      "confidence": 0.95
    },
    "Q2": {
      "zone_id": "Zone 7",
      "feedback_text": "Attention aux unités.",
      "confidence": 0.90
    }
  }
}
```

Pour chaque question, indique:
- zone_id: L'identifiant de la zone choisie (ex: "Zone 3")
- feedback_text: Le feedback à afficher (déjà généré)
- confidence: Ta confiance dans le placement (0.0-1.0)
```""",

    'en': """You must determine optimal coordinates for placing feedback annotations on a student copy.

CONTEXT:
- You will receive an image of the copy annotated with numbered zones (Zone 1, Zone 2, etc.)
- For each question, you must indicate in which zone to place the feedback
- Feedback must be placed CLOSE to the student's answer but WITHOUT overlapping it

PLACEMENT RULES:
1. Choose an EMPTY zone (no student text)
2. Prefer zones just after the answer (below or to the right)
3. Avoid zones that already contain written text
4. If multiple zones possible, choose the closest to the answer

RESPONSE FORMAT (JSON):
```json
{
  "annotations": {
    "Q1": {
      "zone_id": "Zone 3",
      "feedback_text": "Correct.",
      "confidence": 0.95
    },
    "Q2": {
      "zone_id": "Zone 7",
      "feedback_text": "Check your units.",
      "confidence": 0.90
    }
  }
}
```

For each question, indicate:
- zone_id: The chosen zone identifier (e.g., "Zone 3")
- feedback_text: The feedback to display (already generated)
- confidence: Your confidence in the placement (0.0-1.0)
```"""
}


def build_annotation_coordinate_prompt(
    feedback_by_question: Dict[str, str],
    language: str = 'en'
) -> str:
    """
    Build prompt for annotation coordinate detection.

    Args:
        feedback_by_question: Dict mapping question_id -> feedback text
        language: 'fr' or 'en'

    Returns:
        Prompt string
    """
    base_prompt = _ANNOTATION_COORDINATE_PROMPT.get(
        language,
        _ANNOTATION_COORDINATE_PROMPT['en']
    )

    # Add the feedback texts to place
    feedback_list = "\n".join(
        f"- {q_id}: {feedback}"
        for q_id, feedback in feedback_by_question.items()
    )

    return f"""{base_prompt}

FEEDBACKS À PLACER:
{feedback_list}
"""


# Zone detection prompt (first pass - identify zones on page)
_ZONE_DETECTION_PROMPT = {
    'fr': """Analyse cette image de copie d'élève et identifie les ZONES VIDES où des annotations pourraient être placées.

TÂCHE:
1. Identifie les zones vides/blanches sur la page
2. Attribue un identifiant à chaque zone (Zone 1, Zone 2, etc.)
3. Pour chaque zone, donne les coordonnées approximatives en pourcentage de la page

FORMAT DE RÉPONSE (JSON):
```json
{
  "zones": [
    {
      "id": "Zone 1",
      "x_percent": 10.0,
      "y_percent": 5.0,
      "width_percent": 80.0,
      "height_percent": 10.0,
      "description": "Espace vide en haut de page"
    },
    {
      "id": "Zone 2",
      "x_percent": 85.0,
      "y_percent": 20.0,
      "width_percent": 12.0,
      "height_percent": 60.0,
      "description": "Marge droite"
    }
  ]
}
```

Coordonnées:
- x_percent: Position horizontale gauche (0-100% de la largeur page)
- y_percent: Position verticale haute (0-100% de la hauteur page)
- width_percent: Largeur de la zone (pourcentage)
- height_percent: Hauteur de la zone (pourcentage)
```""",

    'en': """Analyze this student copy image and identify EMPTY ZONES where annotations could be placed.

TASK:
1. Identify empty/white zones on the page
2. Assign an identifier to each zone (Zone 1, Zone 2, etc.)
3. For each zone, give approximate coordinates as percentage of page

RESPONSE FORMAT (JSON):
```json
{
  "zones": [
    {
      "id": "Zone 1",
      "x_percent": 10.0,
      "y_percent": 5.0,
      "width_percent": 80.0,
      "height_percent": 10.0,
      "description": "Empty space at top of page"
    },
    {
      "id": "Zone 2",
      "x_percent": 85.0,
      "y_percent": 20.0,
      "width_percent": 12.0,
      "height_percent": 60.0,
      "description": "Right margin"
    }
  ]
}
```

Coordinates:
- x_percent: Horizontal left position (0-100% of page width)
- y_percent: Vertical top position (0-100% of page height)
- width_percent: Zone width (percentage)
- height_percent: Zone height (percentage)
```"""
}


def build_zone_detection_prompt(language: str = 'en') -> str:
    """Build prompt for detecting empty annotation zones on a page."""
    return _ZONE_DETECTION_PROMPT.get(
        language,
        _ZONE_DETECTION_PROMPT['en']
    )


# Combined prompt for direct coordinate assignment (single-pass alternative)
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
