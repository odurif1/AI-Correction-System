"""
Batch grading prompts for the AI correction system.

This module contains all prompt templates for batch grading operations:
- Initial batch grading
- Dual LLM verification
- Ultimatum round
- Grouped and per-question verification
"""

from typing import List, Dict, Any


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH GRADING PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

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

    questions_text = ""
    for qid, qdata in questions.items():
        questions_text += f"""
## {qid}
**Question:** {qdata.get('text', 'Non spécifiée')}
**Critères:** {qdata.get('criteria', 'Non spécifiés')}
**Barème:** {qdata.get('max_points', 1)} point(s)

"""

    return f"""Tu es un correcteur expérimenté. Tu dois corriger {len(copies_data)} copies d'élèves en UNE SEULE analyse.

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


def _build_batch_prompt_en(
    copies_data: List[Dict[str, Any]],
    questions: Dict[str, Dict[str, Any]]
) -> str:
    """Build batch grading prompt in English."""

    questions_text = ""
    for qid, qdata in questions.items():
        questions_text += f"""
## {qid}
**Question:** {qdata.get('text', 'Not specified')}
**Criteria:** {qdata.get('criteria', 'Not specified')}
**Max Points:** {qdata.get('max_points', 1)}

"""

    return f"""You are an experienced grader. You must grade {len(copies_data)} student copies in ONE analysis.

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
      {{"copies": [3, 7], "question": "Q4", "reason": "Word-for-word identical answers"}}
    ]
  }}
}}
```

═══════════════════════════════════════════════════════════════════

Now analyze the {len(copies_data)} provided copies and return your grading in JSON format.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# DUAL LLM VERIFICATION PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

def build_dual_llm_verification_prompt(
    disagreements: List[Any],
    provider_name: str,
    other_provider_name: str,
    is_own_perspective: bool = True,
    language: str = "fr"
) -> str:
    """
    Build a verification prompt for ONE LLM showing both LLMs' grades.

    Args:
        disagreements: List of Disagreement objects to verify
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
    disagreements: List[Any],
    provider_name: str,
    other_provider_name: str,
    is_own_perspective: bool
) -> str:
    """Build dual LLM verification prompt in French."""

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
## Désaccord {i}: Copie {d.copy_index}, {d.question_id}

**TOI ({provider_name})** as donné: **{your_grade}** pts
- Ta lecture: "{your_reading}"
- Ton raisonnement: {your_reasoning}

**L'AUTRE IA ({other_provider_name})** a donné: **{other_grade}** pts
- Sa lecture: "{other_reading}"
- Son raisonnement: {other_reasoning}

Écart: {abs(your_grade - other_grade)} points

"""

    return f"""Tu as corrigé des copies et un DÉSACCORD a été détecté avec un autre correcteur IA.

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


def _build_dual_llm_verification_prompt_en(
    disagreements: List[Any],
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

    return f"""You graded copies and a DISAGREEMENT was detected with another AI grader.

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


# ═══════════════════════════════════════════════════════════════════════════════
# ULTIMATUM PROMPTS
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

    return f"""═══════════════════════════════════════════════════════════════════
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

    return f"""═══════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════════
# GROUPED VERIFICATION PROMPTS (deprecated - kept for compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

def build_grouped_verification_prompt(
    disagreements: List[Any],
    language: str = "fr"
) -> str:
    """Build a grouped verification prompt (deprecated)."""
    if language == "fr":
        return _build_grouped_verification_prompt_fr(disagreements)
    else:
        return _build_grouped_verification_prompt_en(disagreements)


def _build_grouped_verification_prompt_fr(disagreements: List[Any]) -> str:
    """Build grouped verification prompt in French (deprecated)."""
    # Use dual LLM verification instead
    return build_dual_llm_verification_prompt(
        disagreements, "LLM1", "LLM2", is_own_perspective=True, language="fr"
    )


def _build_grouped_verification_prompt_en(disagreements: List[Any]) -> str:
    """Build grouped verification prompt in English (deprecated)."""
    # Use dual LLM verification instead
    return build_dual_llm_verification_prompt(
        disagreements, "LLM1", "LLM2", is_own_perspective=True, language="en"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PER-QUESTION VERIFICATION PROMPTS (deprecated - kept for compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

def build_per_question_verification_prompt(
    disagreement: Any,
    language: str = "fr"
) -> str:
    """Build a per-question verification prompt (deprecated)."""
    return build_dual_llm_verification_prompt(
        [disagreement], "LLM1", "LLM2", is_own_perspective=True, language=language
    )
