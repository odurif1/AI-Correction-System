"""
Verification and ultimatum prompts for the AI correction system.

Contains:
- Unified verification prompt
- Unified ultimatum prompt
"""

from typing import Dict, Any, Optional, List

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

        # Check for scale disagreement
        llm1_max = llm1.get('max_points', 1)
        llm2_max = llm2.get('max_points', 1)
        scale_disagreement = abs(llm1_max - llm2_max) > 0.1

        if language == "fr":
            if q_text:
                q_text_section = f"Texte: {q_text}\n"
            if q_criteria:
                q_text_section += f"Critères: {q_criteria}\n"
            if not q_text and not q_criteria:
                q_text_section = "⚠ ANOMALIE: Question détectée automatiquement - texte non disponible\n"
                auto_detect_warning = True

            # Add scale disagreement warning if applicable
            scale_warning = ""
            if scale_disagreement:
                scale_warning = f"""
⚠ DÉSACCORD SUR LE BARÈME: Vous avez détecté {llm1_max} pts, l'autre a détecté {llm2_max} pts.
→ CHERCHEZ le barème sur la copie et METTEZ-VOUS D'ACCORD sur le barème correct.
→ Utilisez ce barème convenu dans votre "max_points" final.
"""

            questions_section += f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── QUESTION {qid} ───
{q_text_section}Barème que vous avez détecté: {llm1_max} point(s)
{scale_warning}
- Votre note initiale: {llm1.get('grade', 0)}/{llm1_max}
- Votre lecture initiale: "{llm1.get('reading', '')}"
- L'autre note: {llm2.get('grade', 0)}/{llm2_max}
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

            # Add scale disagreement warning if applicable
            scale_warning = ""
            if scale_disagreement:
                scale_warning = f"""
⚠ SCALE DISAGREEMENT: You detected {llm1_max} pts, the other detected {llm2_max} pts.
→ SEARCH for the scale on the copy and AGREE on the correct scale.
→ Use this agreed scale in your final "max_points".
"""

            questions_section += f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── QUESTION {qid} ───
{q_text_section}Scale you detected: {llm1_max} point(s)
{scale_warning}
- Your initial grade: {llm1.get('grade', 0)}/{llm1_max}
- Your initial reading: "{llm1.get('reading', '')}"
- Other's grade: {llm2.get('grade', 0)}/{llm2_max}
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


