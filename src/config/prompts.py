"""
AI Prompt templates for intelligent correction system.

These prompts are designed to elicit nuanced, confident responses
from the AI while being explicit about uncertainty.

Supports automatic language detection based on copy content.
"""

from typing import Dict, Any, Optional, List


# Feedback guidelines - used in all grading prompts
FEEDBACK_GUIDELINE_FR = "sobre et professionnel. Questions simples: bref. Questions difficiles: diagnostic."
FEEDBACK_GUIDELINE_EN = "sober and professional. Simple questions: brief. Difficult questions: diagnosis."


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


# ============= PROMPT TEMPLATES =============

# System messages per language
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


def get_system_message(language: str = 'en') -> str:
    """Get system message for specified language."""
    return _SYSTEM_MESSAGES.get(language, _SYSTEM_MESSAGES['en'])


def get_uncertainty_prompt(uncertainty_type: str, language: str = 'en') -> str:
    """Get uncertainty message for specified type and language."""
    lang_prompts = _UNCERTAINTY_MESSAGES.get(language, _UNCERTAINTY_MESSAGES['en'])
    return lang_prompts.get(uncertainty_type, lang_prompts['other'])


def build_grading_prompt(
    question_text: str,
    criteria: str,
    student_answer: str,
    max_points: float = 5.0,
    similar_answers: List[Dict[str, Any]] = None,
    class_context: str = "",
    language: str = None
) -> str:
    """
    Build a grading prompt with all necessary context.

    Args:
        question_text: The question being asked
        criteria: Grading criteria/rubric
        student_answer: Student's response
        max_points: Maximum points for this question
        similar_answers: List of similar answers from other students
        class_context: Context about patterns across all students
        language: Language for prompts (auto-detected if None)

    Returns:
        Formatted prompt string
    """
    # Auto-detect language if not provided
    if language is None:
        combined_text = f"{question_text} {student_answer} {criteria}"
        language = detect_language(combined_text)

    if language == 'fr':
        return _build_french_grading_prompt(
            question_text, criteria, student_answer, max_points,
            similar_answers, class_context
        )
    else:
        return _build_english_grading_prompt(
            question_text, criteria, student_answer, max_points,
            similar_answers, class_context
        )


def _build_french_grading_prompt(
    question_text: str,
    criteria: str,
    student_answer: str,
    max_points: float,
    similar_answers: List[Dict[str, Any]],
    class_context: str
) -> str:
    """Build French grading prompt."""
    prompt = f"""QUESTION: {question_text}

BARÈME ET CRITÈRES: {criteria}

NOTE MAXIMALE: {max_points} points

"""
    if similar_answers:
        prompt += "RÉPONSES SIMILAIRES D'AUTRES ÉLÈVES:\n"
        for i, ans in enumerate(similar_answers[:3], 1):
            prompt += f"{i}. {ans.get('answer', 'N/A')} (Note: {ans.get('grade', 'N/A')})\n"
        prompt += "\n"

    if class_context:
        prompt += f"CONTEXTE DE CLASSE:\n{class_context}\n\n"

    prompt += f"""RÉPONSE DE L'ÉLÈVE:
{student_answer}

INSTRUCTIONS:
1. Analyser la réponse de l'élève avec attention
2. Attribuer une note sur {max_points} points justifiée par la réponse
3. Évaluer ta CERTITUDE sur cette note (0.0-1.0)
4. Fournir une brève justification

DÉFINITION DE LA CERTITUDE (CONFIDENCE):
- 1.0: Réponse claire et sans ambiguïté, note évidente
- 0.8-0.9: Réponse lisible, interprétation fiable
- 0.6-0.7: Légère incertitude sur l'interprétation ou l'attribution
- 0.4-0.5: Réponse difficile à lire ou ambiguë
- < 0.4: Forte incertitude, révision humaine nécessaire

La certitude mesure TA fiabilité sur cette correction, PAS la qualité de la réponse.

IMPORTANT: Sois OBJECTIF et HONNÊTE.
- N'influe pas la note vers le "gentil" ou le "sévère"
- Si tu n'es pas sûr, dis-le franchement (confidence bas)
- Mieux vaut signaler un doute que de donner une note fausse

FORMAT DE RÉPONSE (respecter exactement):
GRADE: <note>/{max_points}
CONFIDENCE: <0.0-1.0>
UNCERTAINTY_TYPE: <none|unreadable|ambiguous|unexpected|incomplete|other>
IF_UNCERTAIN: <pourquoi tu doutes>
INTERNAL_REASONING: Note: <X/{max_points}>. <analyse technique: justification de la note, critères utilisés, éléments corrects/incorrects>
STUDENT_FEEDBACK: <retour sobre et professionnel. Questions faciles: 1-5 mots. Questions difficiles: diagnostic + correction. Pas de félicitations. Max 25 mots.>"""
    return prompt


def _build_english_grading_prompt(
    question_text: str,
    criteria: str,
    student_answer: str,
    max_points: float,
    similar_answers: List[Dict[str, Any]],
    class_context: str
) -> str:
    """Build English grading prompt."""
    prompt = f"""QUESTION: {question_text}

GRADING CRITERIA: {criteria}

MAX POINTS: {max_points} points

"""
    if similar_answers:
        prompt += "SIMILAR ANSWERS FROM OTHER STUDENTS:\n"
        for i, ans in enumerate(similar_answers[:3], 1):
            prompt += f"{i}. {ans.get('answer', 'N/A')} (Grade: {ans.get('grade', 'N/A')})\n"
        prompt += "\n"

    if class_context:
        prompt += f"CLASS CONTEXT:\n{class_context}\n\n"

    prompt += f"""STUDENT ANSWER:
{student_answer}

INSTRUCTIONS:
1. Analyze the student's answer carefully
2. Assign a grade out of {max_points} points justified by the answer
3. Evaluate your CERTAINTY on this grade (0.0-1.0)
4. Provide a brief justification

CERTAINTY (CONFIDENCE) DEFINITION:
- 1.0: Clear, unambiguous answer, obvious grade
- 0.8-0.9: Readable answer, reliable interpretation
- 0.6-0.7: Slight uncertainty on interpretation or grading
- 0.4-0.5: Difficult to read or ambiguous answer
- < 0.4: High uncertainty, human review needed

Certainty measures YOUR reliability on this grading, NOT the quality of the answer.

IMPORTANT: Be OBJECTIVE and HONEST.
- Do not bias the grade towards "nice" or "harsh"
- If unsure, say so frankly (low confidence)
- Better to signal doubt than give a wrong grade

RESPONSE FORMAT (must follow exactly):
GRADE: <score>/{max_points}
CONFIDENCE: <0.0-1.0>
UNCERTAINTY_TYPE: <none|unreadable|ambiguous|unexpected|incomplete|other>
IF_UNCERTAIN: <why you are uncertain>
INTERNAL_REASONING: Score: <X/{max_points}>. <technical analysis: justification of grade, criteria used, correct/incorrect elements>
STUDENT_FEEDBACK: <sober professional feedback. Easy questions: 1-5 words. Difficult questions: diagnosis + correction. No congratulations. Max 25 words.>"""
    return prompt


def build_vision_grading_prompt(
    question_text: str,
    criteria: str,
    max_points: float = 5.0,
    class_context: str = "",
    similar_answers: List[Dict[str, Any]] = None,
    language: str = 'en'
) -> str:
    """
    Build prompt for vision-based grading.

    Args:
        question_text: The question being asked
        criteria: Grading criteria
        max_points: Maximum points
        class_context: Context about patterns across all students
        similar_answers: List of similar answers from other students
        language: Language for prompt

    Returns:
        Formatted prompt for vision AI
    """
    # Build similar answers section if provided
    similar_section = ""
    if similar_answers:
        similar_section = "RÉPONSES SIMILAIRES D'AUTRES ÉLÈVES:\n" if language == 'fr' else "SIMILAR ANSWERS FROM OTHER STUDENTS:\n"
        for i, ans in enumerate(similar_answers[:3], 1):
            similar_section += f"{i}. {ans.get('answer', 'N/A')} ({ans.get('grade', 'N/A')})\n"
        similar_section += "\n"

    # Build class context section if provided
    context_section = ""
    if class_context:
        context_section = f"CONTEXTE DE CLASSE:\n{class_context}\n\n" if language == 'fr' else f"CLASS CONTEXT:\n{class_context}\n\n"

    if language == 'fr':
        return f"""Analyse cette copie d'élève et corrige-la.

QUESTION: {question_text}

{context_section}{similar_section}CRITÈRES DE CORRECTION: {criteria}

NOTE MAXIMALE: {max_points} points

TÂCHES:
1. Identifier et comprendre la réponse de l'élève
2. Corriger selon les critères
3. Attribuer une note sur {max_points}
4. Évaluer ta CERTITUDE (0-1)

DÉFINITION DE LA CERTITUDE:
- 1.0: Réponse claire et sans ambiguïté
- 0.8-0.9: Réponse lisible, interprétation fiable
- 0.6-0.7: Légère incertitude
- 0.4-0.5: Difficile à lire ou ambiguë
- < 0.4: Forte incertitude, révision humaine nécessaire

IMPORTANT: Sois OBJECTIF et HONNÊTE dans ta note et ton score de confiance.

FORMAT DE RÉPONSE (respecter exactement):
GRADE: <note>/{max_points}
CONFIDENCE: <0.0-1.0>
UNCERTAINTY_TYPE: <none|unreadable|ambiguous|unexpected|incomplete|other>
IF_UNCERTAIN: <pourquoi tu doutes>
STUDENT_ANSWER_READ: <texte exact écrit par l'élève, sans phrase introductive>
INTERNAL_REASONING: Note: <X/{max_points}>. <analyse technique: justification de la note, critères utilisés, éléments corrects/incorrects>
STUDENT_FEEDBACK: <retour sobre et professionnel. Questions faciles: 1-5 mots. Questions difficiles: diagnostic + correction. Pas de félicitations. Max 25 mots.>"""
    else:
        return f"""Analyze this student copy and grade it.

QUESTION: {question_text}

{context_section}{similar_section}GRADING CRITERIA: {criteria}

MAX POINTS: {max_points} points

TASKS:
1. Identify and understand the student's answer
2. Grade according to criteria
3. Assign score out of {max_points}
4. Evaluate your CERTAINTY (0-1)

CERTAINTY DEFINITION:
- 1.0: Clear, unambiguous answer
- 0.8-0.9: Readable, reliable interpretation
- 0.6-0.7: Slight uncertainty
- 0.4-0.5: Difficult to read or ambiguous
- < 0.4: High uncertainty, human review needed

IMPORTANT: Be OBJECTIVE and HONEST in your grade and confidence score.

RESPONSE FORMAT (must follow exactly):
GRADE: <score>/{max_points}
CONFIDENCE: <0.0-1.0>
UNCERTAINTY_TYPE: <none|unreadable|ambiguous|unexpected|incomplete|other>
IF_UNCERTAIN: <why you are uncertain>
STUDENT_ANSWER_READ: <exact text written by the student, no introductory phrase>
INTERNAL_REASONING: Score: <X/{max_points}>. <technical analysis: justification of grade, criteria used, correct/incorrect elements>
STUDENT_FEEDBACK: <sober professional feedback. Easy questions: 1-5 words. Difficult questions: diagnosis + correction. No congratulations. Max 25 words.>"""


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


def build_feedback_prompt(
    graded_copy: Any,
    language: str = 'en'
) -> str:
    """
    Build prompt for generating personalized feedback.

    Args:
        graded_copy: GradedCopy object
        language: Language for feedback

    Returns:
        Formatted prompt for feedback generation
    """
    if language == 'fr':
        return f"""Génère un feedback personnalisé pour cet élève.

NOTE: {graded_copy.total_score}/{graded_copy.max_score}

DÉTAILS PAR QUESTION:
{_format_questions_for_feedback(graded_copy, 'fr')}

INSTRUCTIONS:
1. Feedback encourageant et constructif
2. Mettre en valeur les points forts
3. Identifier les axes d'amélioration
4. 2-3 phrases maximum"""
    else:
        return f"""Generate personalized feedback for this student.

SCORE: {graded_copy.total_score}/{graded_copy.max_score}

QUESTION DETAILS:
{_format_questions_for_feedback(graded_copy, 'en')}

INSTRUCTIONS:
1. Encouraging and constructive feedback
2. Highlight strengths
3. Identify areas for improvement
4. 2-3 sentences maximum"""


def _format_questions_for_feedback(graded_copy: Any, language: str) -> str:
    """Format question details for feedback prompt."""
    lines = []
    for q_id, grade in graded_copy.grades.items():
        feedback = graded_copy.student_feedback.get(q_id, "")
        lines.append(f"- {q_id}: {grade} points ({feedback})")
    return "\n".join(lines)


def build_multi_question_grading_prompt(
    questions: List[Dict[str, Any]],
    language: str = "fr",
    second_reading: bool = False
) -> str:
    """
    Build prompt for grading multiple questions in a single API call.

    Args:
        questions: List of question dicts with:
            - id: Question identifier (e.g., "Q1")
            - text: The question text
            - criteria: Grading criteria
            - max_points: Maximum points
        language: Language for prompt
        second_reading: If True, add second reading instruction in prompt

    Returns:
        Formatted prompt for multi-question grading
    """
    # Build questions section
    questions_section = ""
    for i, q in enumerate(questions, 1):
        # Don't show max_points since it needs to be detected by the LLM
        questions_section += f"""
━━━ QUESTION {q['id']} ━━━
TEXTE: {q['text']}
CRITÈRES: {q['criteria']}

"""

    if language == "fr":
        base_prompt = f"""Tu es un correcteur expérimenté. Analyse cette copie et corrige TOUTES les questions.

━━━ QUESTIONS À CORRIGER ━━━
{questions_section}

━━━ INSTRUCTIONS IMPORTANTES ━━━
1. CHERCHE D'ABORD LE BARÈME sur le document:
   - Regarde en haut de la page, à côté de chaque question
   - Le barème est souvent indiqué comme "(1pt)", "(2 pts)", "/1", "/2", etc.
   - Si tu ne trouves pas le barème, utilise 1 point par défaut

2. Pour CHAQUE question:
   - Localise la réponse de l'élève
   - Lis EXACTEMENT ce qu'il a écrit
   - Note sur le barème détecté (0 à max)
   - Évalue ta certitude (0-1)

3. Génère un feedback sobre

━━━ NOTATION RELATIVE AU BARÈME ━━━
- La note DOIT être comprise entre 0 et le barème détecté
- Exemple: si barème = 2 points, la note peut être 0, 0.5, 1, 1.5 ou 2
- 0 = réponse fausse ou absente
- Max = réponse complète et correcte

⚠ LANGUE: Répondez IMPÉRATIVEMENT EN FRANÇAIS dans tous les champs texte (reasoning, feedback).

━━━ FORMAT DE RÉPONSE (JSON) ━━━
Réponds UNIQUEMENT avec un JSON valide. Tous les textes doivent être EN FRANÇAIS:

{{
  "student_name": "<nom détecté ou null>",
  "questions": {{
    "Q1": {{
      "location": "<page X, zone Y>",
      "student_answer_read": "<texte exact écrit par l'élève>",
      "max_points": <barème détecté sur la copie>,
      "grade": <note sur le barème>,
      "confidence": <0.0-1.0>,
      "reasoning": "<analyse technique EN FRANÇAIS>",
      "feedback": f"<{FEEDBACK_GUIDELINE_FR}>"
    }},
    "Q2": {{ ... }}
  }}
}}"""
    else:
        base_prompt = f"""You are an experienced grader. Analyze this copy and grade ALL questions.

━━━ QUESTIONS TO GRADE ━━━
{questions_section}

━━━ IMPORTANT INSTRUCTIONS ━━━
1. FIRST FIND THE SCALE on the document:
   - Look at the top of the page, next to each question
   - Scale is often indicated as "(1pt)", "(2 pts)", "/1", "/2", etc.
   - If you can't find the scale, use 1 point as default

2. For EACH question:
   - Locate the student's answer
   - Read EXACTLY what they wrote
   - Grade on the detected scale (0 to max)
   - Evaluate your certainty (0-1)

3. Generate sober feedback

━━━ GRADING RELATIVE TO SCALE ━━━
- Grade MUST be between 0 and the detected scale
- Example: if scale = 2 points, grade can be 0, 0.5, 1, 1.5 or 2
- 0 = wrong or missing answer
- Max = complete and correct answer

━━━ RESPONSE FORMAT (JSON) ━━━
Respond ONLY with valid JSON:

{{
  "student_name": "<detected name or null>",
  "questions": {{
    "Q1": {{
      "location": "<page X, zone Y>",
      "student_answer_read": "<exact text written by student>",
      "max_points": <scale detected on copy>,
      "grade": <score on scale>,
      "confidence": <0.0-1.0>,
      "reasoning": "<technical analysis>",
      "feedback": f"<{FEEDBACK_GUIDELINE_EN}>"
    }},
    "Q2": {{ ... }}
  }}
}}"""

    # Add second reading instruction if enabled
    if second_reading:
        if language == "fr":
            base_prompt += """

━━━ DEUXIÈME LECTURE (IMPORTANT) ━━━
Après ta première correction:
1. RELIS ta correction en entier
2. Vérifie que chaque note correspond au barème détecté
3. Vérifie que tes lectures des réponses sont exactes
4. Ajuste si nécessaire

Tu dois faire ce travail de vérification DANS CETTE MÊME RÉPONSE."""
        else:
            base_prompt += """

━━━ SECOND READING (IMPORTANT) ━━━
After your first grading pass:
1. RE-READ your corrections entirely
2. Verify each grade matches the detected scale
3. Verify your readings of answers are accurate
4. Adjust if necessary

You must do this verification IN THIS SAME RESPONSE."""

    return base_prompt


def build_auto_detect_grading_prompt(
    language: str = "fr",
    second_reading: bool = False
) -> str:
    """
    Build prompt for detecting and grading questions when none are pre-defined.

    Used in INDIVIDUAL mode where questions are not known ahead of time.

    Args:
        language: Language for prompt
        second_reading: If True, add second reading instruction

    Returns:
        Formatted prompt for auto-detect grading
    """
    if language == "fr":
        base_prompt = """Tu es un correcteur expérimenté. Analyse cette copie d'élève.

━━━ TA MISSION ━━━
1. IDENTIFIE le nom de l'élève (si visible)
2. DÉTECTE toutes les questions présentes sur la copie
3. CORRIGE chaque question détectée

━━━ COMMENT DÉTECTER LES QUESTIONS ━━━
- Cherche les numéros de questions: Q1, Q2, 1., 2., Question 1, etc.
- Chaque question a généralement une zone de réponse associée
- Numérote-les Q1, Q2, Q3... dans l'ordre d'apparition

━━━ BARÈME ━━━
- CHERCHE le barème sur le document (souvent indiqué comme "(1pt)", "/2", etc.)
- Si tu ne trouves pas le barème, utilise 1 point par défaut

━━━ POUR CHAQUE QUESTION DÉTECTÉE ━━━
- Localise la réponse de l'élève
- Lis EXACTEMENT ce qu'il a écrit
- Note sur le barème détecté (0 à max)
- Évalue ta certitude (0-1)
- Génère un feedback sobre et professionnel

━━━ NOTATION NUANCÉE ━━━
Accorde des POINTS PARTIELS pour les réponses partiellement correctes:
- Formule/méthode correcte mais exécution erronée: 50% des points
- Démarche correcte avec erreurs mineures: 75% des points
- Seules les erreurs conceptuelles majeures méritent 0 point
NE JAMAIS mettre 0 si une partie de la démarche est correcte.

━━━ FORMAT DE RÉPONSE (JSON) ━━━
Réponds UNIQUEMENT avec un JSON valide:

{
  "student_name": "<nom détecté ou null>",
  "questions": {
    "Q1": {
      "question_text": "<texte de la question détectée>",
      "location": "<page X, zone Y>",
      "student_answer_read": "<texte exact écrit par l'élève>",
      "max_points": <barème détecté>,
      "grade": <note sur le barème>,
      "confidence": <0.0-1.0>,
      "reasoning": "<analyse technique>",
      "feedback": f"<{FEEDBACK_GUIDELINE_FR}>"
    },
    "Q2": { ... }
  }
}"""
    else:
        base_prompt = """You are an experienced grader. Analyze this student copy.

━━━ YOUR MISSION ━━━
1. IDENTIFY the student name (if visible)
2. DETECT all questions present on the copy
3. GRADE each detected question

━━━ HOW TO DETECT QUESTIONS ━━━
- Look for question numbers: Q1, Q2, 1., 2., Question 1, etc.
- Each question usually has an associated answer area
- Number them Q1, Q2, Q3... in order of appearance

━━━ SCALE ━━━
- FIND the scale on the document (often indicated as "(1pt)", "/2", etc.)
- If you can't find the scale, use 1 point as default

━━━ FOR EACH DETECTED QUESTION ━━━
- Locate the student's answer
- Read EXACTLY what they wrote
- Grade on the detected scale (0 to max)
- Evaluate your certainty (0-1)
- Generate sober, professional feedback

━━━ NUANCED GRADING ━━━
Give PARTIAL CREDIT for partially correct answers:
- Correct formula/method but wrong execution: 50% of points
- Correct approach with minor errors: 75% of points
- Only major conceptual errors deserve 0 points
NEVER give 0 if part of the approach is correct.

━━━ RESPONSE FORMAT (JSON) ━━━
Respond ONLY with valid JSON:

{
  "student_name": "<detected name or null>",
  "questions": {
    "Q1": {
      "question_text": "<detected question text>",
      "location": "<page X, zone Y>",
      "student_answer_read": "<exact text written by student>",
      "max_points": <detected scale>,
      "grade": <grade on scale>,
      "confidence": <0.0-1.0>,
      "reasoning": "<technical analysis>",
      "feedback": f"<{FEEDBACK_GUIDELINE_EN}>"
    },
    "Q2": { ... }
  }
}"""

    # Add second reading instruction if enabled
    if second_reading:
        if language == "fr":
            base_prompt += """

━━━ DEUXIÈME LECTURE (IMPORTANT) ━━━
Après ta première correction:
1. RELIS ta correction en entier
2. Vérifie que chaque note correspond au barème détecté
3. Vérifie que tes lectures des réponses sont exactes
4. Ajuste si nécessaire

Tu dois faire ce travail de vérification DANS CETTE MÊME RÉPONSE."""
        else:
            base_prompt += """

━━━ SECOND READING (IMPORTANT) ━━━
After your first grading pass:
1. RE-READ your corrections entirely
2. Verify each grade matches the detected scale
3. Verify your readings of answers are accurate
4. Adjust if necessary

You must do this verification IN THIS SAME RESPONSE."""

    return base_prompt


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

def build_unified_verification_prompt(
    questions: List[Dict[str, Any]],
    disagreements: List[Dict[str, Any]],
    name_disagreement: Optional[Dict[str, Any]] = None,
    language: str = "fr"
) -> str:
    """
    Build unified verification prompt for ALL disagreements.

    Args:
        questions: List of all question dicts with id, text, criteria, max_points
        disagreements: List of disagreement dicts, each with:
            - question_id: str
            - llm1: {grade, reading, confidence, max_points}
            - llm2: {grade, reading, confidence, max_points}
            - type: disagreement type
            - reason: str
        name_disagreement: Optional dict with llm1_name, llm2_name
        language: Language for prompt

    Returns:
        Formatted prompt for unified verification
    """
    # Build question lookup
    question_lookup = {q["id"]: q for q in questions}

    # Build name section if there's a name disagreement
    name_section = ""
    if name_disagreement:
        if language == "fr":
            name_section = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── NOM DE L'ÉLÈVE ───
- Vous avez lu: "{name_disagreement.get('llm1_name', '')}"
- L'autre correcteur a lu: "{name_disagreement.get('llm2_name', '')}"
→ Réexaminez le nom sur la copie.
"""
        else:
            name_section = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── STUDENT NAME ───
- You read: "{name_disagreement.get('llm1_name', '')}"
- The other grader read: "{name_disagreement.get('llm2_name', '')}"
→ Re-examine the name on the copy.
"""

    # Build questions section
    questions_section = ""
    auto_detect_warning = False
    for d in disagreements:
        qid = d["question_id"]
        q = question_lookup.get(qid, {})
        llm1 = d.get("llm1", {})
        llm2 = d.get("llm2", {})

        # Build question text section (only if we have the text)
        q_text_section = ""
        q_text = q.get('text', '')
        q_criteria = q.get('criteria', '')

        if language == "fr":
            if q_text:
                q_text_section = f"Texte: {q_text}\n"
            if q_criteria:
                q_text_section += f"Critères: {q_criteria}\n"
            if not q_text and not q_criteria:
                q_text_section = "⚠ ANOMALIE: Question détectée automatiquement - texte non disponible\n"
                auto_detect_warning = True

            questions_section += f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── QUESTION {qid} ───
{q_text_section}Barème: {llm1.get('max_points', 1)} point(s)

- Votre note initiale: {llm1.get('grade', 0)}/{llm1.get('max_points', 1)}
- Votre lecture initiale: "{llm1.get('reading', '')}"
- L'autre note: {llm2.get('grade', 0)}/{llm2.get('max_points', 1)}
- Lecture de l'autre: "{llm2.get('reading', '')}"
"""
        else:
            if q_text:
                q_text_section = f"Text: {q_text}\n"
            if q_criteria:
                q_text_section += f"Criteria: {q_criteria}\n"
            if not q_text and not q_criteria:
                q_text_section = "⚠ ANOMALY: Auto-detected question - text not available\n"
                auto_detect_warning = True

            questions_section += f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── QUESTION {qid} ───
{q_text_section}Scale: {llm1.get('max_points', 1)} point(s)

- Your initial grade: {llm1.get('grade', 0)}/{llm1.get('max_points', 1)}
- Your initial reading: "{llm1.get('reading', '')}"
- Other's grade: {llm2.get('grade', 0)}/{llm2.get('max_points', 1)}
- Other's reading: "{llm2.get('reading', '')}"
"""

    # Build questions JSON format with reading anchors
    questions_json = ""
    for d in disagreements:
        qid = d["question_id"]
        llm1 = d.get("llm1", {})
        original_reading = llm1.get('reading', '').replace('"', "'")  # Escape quotes
        # Use language-specific placeholders
        if language == "fr":
            questions_json += f'''
    "{qid}": {{
      "student_answer_read": "<votre lecture de la copie>",
      "original_reading": "{original_reading}",
      "grade": <note>,
      "max_points": {llm1.get('max_points', 1)},
      "confidence": <0.0-1.0>,
      "reasoning": "<analysez les deux lectures, identifiez la correcte>",
      "feedback": f"<{FEEDBACK_GUIDELINE_FR}>"
    }},'''
        else:
            questions_json += f'''
    "{qid}": {{
      "student_answer_read": "<your reading of the student's copy>",
      "original_reading": "{original_reading}",
      "grade": <grade>,
      "max_points": {llm1.get('max_points', 1)},
      "confidence": <0.0-1.0>,
      "reasoning": "<analyze both readings, identify the correct one>",
      "feedback": f"<{FEEDBACK_GUIDELINE_EN}>"
    }},'''

    # Remove trailing comma
    if questions_json.endswith(','):
        questions_json = questions_json[:-1]

    if language == "fr":
        return f"""─── VÉRIFICATION UNIFIÉE ───

Vous avez corrigé cette copie avec un autre correcteur.
Certains points sont en désaccord. Veuillez réexaminer TOUS ces éléments.
{name_section}{questions_section}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── RÈGLE FONDAMENTALE ───

⚠ LISEZ LA COPIE VOUS-MÊME. NE COPIEZ PAS LA LECTURE DE L'AUTRE.

- Regardez la copie de l'élève. Lisez la réponse avec vos propres yeux.
- Identifiez d'abord la bonne réponse sur l'image, puis comparez avec celle de l'élève.
- Votre "student_answer_read" = votre lecture personnelle de la copie de l'élève.
- Dans votre raisonnement, considérez les deux lectures: la vôtre et celle de l'autre correcteur. Identifiez laquelle correspond à la copie de l'élève.
- Ne changez pas votre note juste pour "être d'accord".

⚠ LANGUE: Répondez IMPÉRATIVEMENT EN FRANÇAIS.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── FORMAT DE RÉPONSE (JSON) ───
Réponds UNIQUEMENT avec un JSON valide:
{{
  "student_name": "<nom final ou null si inchangé>",
  "questions": {{{questions_json}
  }}
}}"""
    else:
        return f"""─── UNIFIED VERIFICATION ───

You graded this copy with another grader.
Some points are in disagreement. Please re-examine ALL these elements.
{name_section}{questions_section}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── FUNDAMENTAL RULE ───

⚠ READ THE STUDENT'S COPY YOURSELF. DO NOT COPY THE OTHER'S READING.

- Look at the student's copy. Read the answer with your own eyes.
- First identify the correct answer from the image, then compare with the student's.
- Your "student_answer_read" = your personal reading of the student's copy.
- In your reasoning, consider both readings: yours and the other grader's. Identify which one matches the student's copy.
- Do not change your grade just to "agree".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── RESPONSE FORMAT (JSON) ───
Respond ONLY with valid JSON:
{{
  "student_name": "<final name or null if unchanged>",
  "questions": {{{questions_json}
  }}
}}"""


def build_unified_ultimatum_prompt(
    questions: List[Dict[str, Any]],
    disagreements: List[Dict[str, Any]],
    evolution: Dict[str, List[float]],
    name_disagreement: Optional[Dict[str, Any]] = None,
    name_evolution: Optional[List[str]] = None,
    language: str = "fr",
    reading_anchors: Optional[Dict[str, str]] = None
) -> str:
    """
    Build unified ultimatum prompt for remaining disagreements after verification.

    Args:
        questions: List of all question dicts
        disagreements: List of disagreement dicts (only unresolved ones)
        evolution: Dict mapping question_id -> list of grade tuples [(initial1, initial2), (after_v1, after_v2)]
        name_disagreement: Optional dict with original name disagreement
        name_evolution: Optional list of name tuples [(llm1_initial, llm2_initial), (llm1_after, llm2_after)]
        language: Language for prompt
        reading_anchors: Dict mapping question_id -> agreed reading to anchor (only if initial reading agreed)

    Returns:
        Formatted prompt for unified ultimatum
    """
    # Build question lookup
    question_lookup = {q["id"]: q for q in questions}
    reading_anchors = reading_anchors or {}

    # Build name section if there's a name disagreement
    name_section = ""
    if name_disagreement:
        if language == "fr":
            name_section = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── NOM DE L'ÉLÈVE (TOUJOURS EN DÉSACCORD) ───
- Votre lecture initiale: "{name_disagreement.get('llm1_name', '')}"
- Lecture de l'autre: "{name_disagreement.get('llm2_name', '')}"
→ DÉCISION FINALE requise.
"""
        else:
            name_section = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── STUDENT NAME (STILL IN DISAGREEMENT) ───
- Your initial reading: "{name_disagreement.get('llm1_name', '')}"
- Other's reading: "{name_disagreement.get('llm2_name', '')}"
→ FINAL DECISION required.
"""

    # Build questions section with evolution and reading anchors
    questions_section = ""
    for d in disagreements:
        qid = d["question_id"]
        q = question_lookup.get(qid, {})
        q_evolution = evolution.get(qid, [])
        llm1 = d.get("llm1", {})
        llm2 = d.get("llm2", {})

        # Check if reading is anchored (agreed initially)
        anchored_reading = reading_anchors.get(qid)
        anchor_warning = ""
        if anchored_reading:
            if language == "fr":
                anchor_warning = f'''
⚠ LECTURE FIGÉE: "{anchored_reading}"
Cette lecture était en ACCORD INITIAL entre les deux correcteurs.
Vous DEVEZ utiliser cette lecture. N'en inventez pas une autre.'''
            else:
                anchor_warning = f'''
⚠ ANCHORED READING: "{anchored_reading}"
This reading was in INITIAL AGREEMENT between both graders.
You MUST use this reading. Do not invent another one.'''

        # Format evolution
        if len(q_evolution) >= 2:
            initial = q_evolution[0]
            after_v = q_evolution[1]
            if language == "fr":
                evolution_text = f"Évolution: {initial[0]} → {after_v[0]} (vous) | {initial[1]} → {after_v[1]} (autre)"
            else:
                evolution_text = f"Evolution: {initial[0]} → {after_v[0]} (you) | {initial[1]} → {after_v[1]} (other)"
        elif len(q_evolution) == 1:
            initial = q_evolution[0]
            if language == "fr":
                evolution_text = f"Notes initiales: {initial[0]} (vous) | {initial[1]} (autre)"
            else:
                evolution_text = f"Initial grades: {initial[0]} (you) | {initial[1]} (other)"
        else:
            evolution_text = ""

        if language == "fr":
            questions_section += f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── QUESTION {qid} (TOUJOURS EN DÉSACCORD) ───
Texte: {q.get('text', 'N/A')}
Barème: {llm1.get('max_points', 1)} point(s)

- Votre note actuelle: {llm1.get('grade', 0)}/{llm1.get('max_points', 1)}
- Note de l'autre: {llm2.get('grade', 0)}/{llm2.get('max_points', 1)}
- {evolution_text}
{anchor_warning}
"""
        else:
            questions_section += f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── QUESTION {qid} (STILL IN DISAGREEMENT) ───
Text: {q.get('text', 'N/A')}
Scale: {llm1.get('max_points', 1)} point(s)

- Your current grade: {llm1.get('grade', 0)}/{llm1.get('max_points', 1)}
- Other's grade: {llm2.get('grade', 0)}/{llm2.get('max_points', 1)}
- {evolution_text}
{anchor_warning}
"""

    # Build questions JSON format with reading constraints
    questions_json = ""
    for d in disagreements:
        qid = d["question_id"]
        llm1 = d.get("llm1", {})
        anchored_reading = reading_anchors.get(qid)

        if anchored_reading:
            # Reading is anchored - must use it
            reading_field = f'"student_answer_read": "{anchored_reading}",'
        else:
            # Reading not anchored - can re-read but must justify
            if language == "fr":
                reading_field = '''"student_answer_read": "<votre lecture - RELISEZ sur l'image>",'''
            else:
                reading_field = '''"student_answer_read": "<your reading - RE-READ from image>",'''

        # Language-specific feedback placeholder
        feedback_placeholder = FEEDBACK_GUIDELINE_FR if language == "fr" else FEEDBACK_GUIDELINE_EN

        questions_json += f'''
    "{qid}": {{
      {reading_field}
      "grade": <note finale>,
      "max_points": {llm1.get('max_points', 1)},
      "confidence": <0.0-1.0>,
      "reasoning": "<justification finale>",
      "feedback": "{feedback_placeholder}"
    }},'''

    # Remove trailing comma
    if questions_json.endswith(','):
        questions_json = questions_json[:-1]

    if language == "fr":
        return f"""─── ULTIMATUM UNIFIÉ - DÉCISION FINALE ───

Le désaccord PERSISTE après vérification. Vous devez prendre une DÉCISION FINALE.
{name_section}{questions_section}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── RÈGLES CRITIQUES ───
1. Ne changez votre note QUE si vous êtes CONVAINCU d'une erreur
2. ⚠ INTERDICTION D'INVENTER une nouvelle lecture
3. Si une lecture est figée (en accord initial), vous DEVEZ l'utiliser
4. Si vous changez de position, justifiez pourquoi votre analyse initiale était erronée

─── VOS OPTIONS ───
- Option A: Accepter l'autre note → expliquez pourquoi leur analyse est meilleure
- Option B: Maintenir votre note → arguments précis justifiant votre position

⚠ LANGUE: Répondez IMPÉRATIVEMENT EN FRANÇAIS dans tous les champs texte (reasoning, feedback).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── FORMAT DE RÉPONSE (JSON) ───
Réponds UNIQUEMENT avec un JSON valide:
{{
  "student_name": "<nom final ou null si inchangé>",
  "questions": {{{questions_json}
  }}
}}"""
    else:
        return f"""─── UNIFIED ULTIMATUM - FINAL DECISION ───

Disagreement PERSISTS after verification. You must make a FINAL DECISION.
{name_section}{questions_section}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── CRITICAL RULES ───
1. Only change your grade if you are CONVINCED of an error
2. ⚠ FORBIDDEN TO INVENT a new reading
3. If a reading is anchored (initially agreed), you MUST use it
4. If you change position, justify why your initial analysis was wrong

─── YOUR OPTIONS ───
- Option A: Accept the other grade → explain why their analysis is better
- Option B: Maintain your grade → precise arguments supporting your position

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── RESPONSE FORMAT (JSON) ───
Respond ONLY with valid JSON:
{{
  "student_name": "<final name or null if unchanged>",
  "questions": {{{questions_json}
  }}
}}"""


# Test
if __name__ == '__main__':
    tests = [
        ('Le chat est gris', 'fr'),
        ('The cat is gray', 'en'),
    ]
    for text, expected in tests:
        result = detect_language(text)
        status = 'OK' if result == expected else 'KO'
        print(f'{status}: {text} -> {result} ({expected})')
