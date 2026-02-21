"""
Grading prompts for the AI correction system.

Contains:
- Single question grading
- Multi-question grading  
- Vision-based grading
- Auto-detect grading
- Feedback generation
"""

from typing import Dict, Any, Optional, List
from prompts.common import detect_language

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



