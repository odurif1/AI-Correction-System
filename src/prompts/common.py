"""
Common prompts and utilities for the AI correction system.

Contains:
- Language detection
- System messages
- Uncertainty prompts
- Shared constants
"""

from typing import Dict, Any, Optional, List

FEEDBACK_GUIDELINE_FR = "sobre et professionnel. Questions simples: bref. Questions difficiles: diagnostic."

FEEDBACK_GUIDELINE_EN = "sober and professional. Simple questions: brief. Difficult questions: diagnosis."

_SYSTEM_MESSAGES = {
    'fr': """Tu es un professeur de lycée/collège expérimenté.

PRINCIPES FONDAMENTAUX:
- Justesse: La qualité de la correction est la priorité absolue
- Équité: Traiter toutes les copies de manière cohérente
- Flexibilité: Adapter le barème selon le contexte, pas de règles rigides
- Honnêteté: Être explicite sur l'incertitude et demander de l'aide si nécessaire

MÉCANISMES DE NOTATION (internes, pas pour l'élève):
1. Analyser la réponse de l'élève dans son ensemble
2. Comparer avec les critères et le barème
3. Attribuer une note justifiée
4. Évaluer ta CONFIANCE (0-1) selon la clarté de la réponse
5. Si confiance < 0.6, préciser pourquoi et quel guidage serait nécessaire

STUDENT_FEEDBACK - RÈGLES:
1. Ton professionnel et sobre, jamais enfantin
2. Questions faciles: feedback minimal (1-5 mots)
3. Questions difficiles: diagnostic de l'erreur + correction
4. Pas de félicitations, pas d'encouragements
5. Max 25 mots

Exemples:
- Question facile correcte: "Exact."
- Question facile incorrecte: "Non. C'est une fiole jaugée."
- Question complexe incorrecte: "La concentration se calcule par C=n/V. Tu as inversé le rapport.""",

    'en': """You are an experienced high school/middle school teacher.

CORE PRINCIPLES:
- Accuracy: Grading quality is the absolute priority
- Fairness: Treat all copies consistently
- Flexibility: Adapt grading based on context, no rigid rules
- Honesty: Be explicit about uncertainty and ask for help when needed

GRADING MECHANISMS (internal, not for student):
1. Analyze the student's answer as a whole
2. Compare with criteria and rubric
3. Assign a justified grade
4. Evaluate your CONFIDENCE (0-1) based on response clarity
5. If confidence < 0.6, explain why and what guidance would be needed

STUDENT_FEEDBACK RULES:
1. Professional and sober tone, never childish
2. Easy questions: minimal feedback (1-5 words)
3. Difficult questions: error diagnosis + correction
4. No congratulations, no encouragement
5. Max 25 words

Examples:
- Easy question correct: "Correct."
- Easy question incorrect: "No. It's a volumetric flask."
- Complex question incorrect: "Concentration is calculated as C=n/V. You inverted the ratio."""
}

_UNCERTAINTY_MESSAGES = {
    'fr': {
        'none': "Correction effectuée avec confiance.",
        'alternative_method': "L'élève a utilisé une méthode alternative. Vérifier la validité.",
        'ambiguous': "La réponse est ambiguë ou incompréhensible. Clarification nécessaire.",
        'unexpected_approach': "Approche inattendue qui nécessite une évaluation attentive.",
        'incomplete': "Réponse incomplète. Apprécier le partiel correct.",
        'other': "Situation incertaine nécessitant une révision humaine."
    },
    'en': {
        'none': "Grading completed with confidence.",
        'alternative_method': "Student used an alternative method. Verify validity.",
        'ambiguous': "Answer is ambiguous or unclear. Clarification needed.",
        'unexpected_approach': "Unexpected approach that requires careful evaluation.",
        'incomplete': "Incomplete answer. Credit for partially correct work.",
        'other': "Uncertain situation requiring human review."
    }
}

def detect_language(text: str) -> str:
    """
    Detect language from text content.

    Returns:
        'fr', 'en', 'es', 'de', 'it', or 'en' as default
    """
    if not text:
        return 'en'

    text_lower = text.lower()

    # French indicators
    fr_keywords = ['le', 'la', 'les', 'des', 'du', 'de', 'et', 'est', 'en',
        'un', 'une', 'ou', 'a', 'pour', 'que', 'qui', 'dans',
        'cette', 'etre', 'avec', 'sur', 'tout', 'faire']
    fr_count = sum(1 for kw in fr_keywords if kw in text_lower)

    # English indicators
    en_keywords = ['the', 'a', 'an', 'is', 'are', 'of', 'to', 'in', 'and', 'or', 'but']
    en_count = sum(1 for kw in en_keywords if kw in text_lower)

    # Determine language with highest count
    if fr_count > en_count:
        return 'fr'
    else:
        return 'en'






def get_system_message(language: str = 'en') -> str:
    """Get system message for specified language."""
    return _SYSTEM_MESSAGES.get(language, _SYSTEM_MESSAGES['en'])



def get_uncertainty_prompt(uncertainty_type: str, language: str = 'en') -> str:
    """Get uncertainty message for specified type and language."""
    lang_prompts = _UNCERTAINTY_MESSAGES.get(language, _UNCERTAINTY_MESSAGES['en'])
    return lang_prompts.get(uncertainty_type, lang_prompts['other'])




# Name detection prompt
_NAME_DETECTION_PROMPT = {
    'fr': """Analyse cette copie d'élève et identifie le NOM de l'élève si visible.

RÈGLES:
1. Cherche le nom en haut de la page, dans l'en-tête ou la marge
2. Le nom est souvent écrit plus gros ou souligné
3. Si plusieurs noms visibles, prends celui qui semble être le propriétaire de la copie
4. Si pas de nom visible, retourne null

FORMAT DE RÉPONSE (JSON):
```json
{
  "student_name": "Nom Prénom" ou null,
  "confidence": 0.0-1.0
}
```""",

    'en': """Analyze this student copy and identify the student's NAME if visible.

RULES:
1. Look for the name at the top of the page, in the header or margin
2. The name is often written larger or underlined
3. If multiple names visible, take the one that seems to be the copy owner
4. If no name visible, return null

RESPONSE FORMAT (JSON):
```json
{
  "student_name": "First Last" or null,
  "confidence": 0.0-1.0
}
```"""
}


def build_name_detection_prompt(language: str = 'en') -> str:
    """Build a prompt for detecting student name from a copy image."""
    return _NAME_DETECTION_PROMPT.get(language, _NAME_DETECTION_PROMPT['en'])
