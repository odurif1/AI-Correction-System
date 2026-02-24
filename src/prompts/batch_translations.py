"""
Translation strings for batch grading prompts.

Add new languages by creating a new dictionary with the same structure.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FRENCH TRANSLATIONS
# ═══════════════════════════════════════════════════════════════════════════════

TRANSLATIONS_FR = {
    # Batch grading prompts
    "batch": {
        "role": "Tu es un correcteur expérimenté. Tu dois corriger {copies_count} en UNE SEULE analyse.",
        "approach_intro": """Cette approche te permet de:
- Garantir la COHÉRENCE: même réponse = même note
- Détecter les PATTERNS: réponses courantes, outliers, copiage potentiel
- Être EFFICACE: tout en un seul passage""",
        "rubric_title": "BARÈME DE NOTATION",
        "question_text": "Question",
        "criteria_text": "Critères",
        "max_points_text": "Barème",
        "not_specified": "Non spécifié",
        "points_suffix": "point(s)",

        # Student detection mode
        "detection_title": "DÉTECTION DES ÉLÈVES",
        "detection_intro": "Ce PDF contient POTENTIELLEMENT PLUSIEURS ÉLÈVES. Tu dois:",
        "detection_steps": [
            "IDENTIFIER combien d'élèves différents sont dans ce document",
            "DÉTERMINER quelles pages appartiennent à chaque élève",
            "LIRE le nom de chaque élève (si visible)",
            "CORRIGER chaque élève séparément"
        ],
        "detection_clues": "Indices pour identifier les élèves:",
        "detection_clue_list": [
            "Changement de nom en haut de page",
            "Style d'écriture différent",
            "Mêmes questions répondues différemment (indique un nouvel élève)",
            "Saut de page ou séparateur visuel"
        ],

        # Copies section
        "copies_title": "COPIES À CORRIGER",
        "copies_instruction_detected": "Pour CHAQUE ÉLÈVE détecté",
        "copies_instruction_batch": "Pour CHAQUE copie",
        "copies_steps": [
            "Lis le nom de l'élève (si présent)",
            "Lis les réponses à chaque question",
            "Note selon le barème",
            "Donne un feedback sobre et professionnel"
        ],

        # Rules
        "rules_title": "RÈGLES IMPORTANTES",
        "rules": [
            ("COHÉRENCE ABSOLUE", "Si deux élèves ont écrit la même réponse, ils doivent avoir la même note"),
            ("LECTURE ATTENTIVE", "Utilise le CONTEXTE (question, autres copies, cohérence) pour déchiffrer l'écriture manuscrite"),
            ("FEEDBACK SOBRE", "Commentaire court, constructif, adapté à la difficulté"),
            ("DÉTECTION PATTERNS", "Note si beaucoup d'élèves ont la même réponse (correcte ou non)"),
            ("CROISEMENT", "Comparer les réponses entre copies t'aide à lire l'écriture et assurer la cohérence")
        ],

        # JSON format
        "response_format_title": "FORMAT DE RÉPONSE (JSON)",
        "json_copy_index": "1",
        "json_student_name": "Nom de l'élève ou null",
        "json_student_answer": "Ce que l'élève a écrit",
        "json_feedback": "Feedback sobre",
        "json_overall_feedback": "Commentaire général sur la copie",
        "json_common_answer": "fiole jaugée",
        "json_reason_similarity": "Réponses identiques mot pour mot",
        "json_reasoning": "Pourquoi cette note",

        # Final
        "final_instruction": "Analyse maintenant le document et retourne ta correction au format JSON.",
    },

    # Dual LLM verification prompts
    "verification": {
        "intro": "Tu as corrigé des copies et un DÉSACCORD a été détecté avec un autre correcteur.",
        "mission_intro": "Tu dois maintenant RÉEXAMINER ta correction en tenant compte de l'avis de l'autre correcteur.",
        "disagreements_title": "DÉSACCORDS À RÉEXAMINER",
        "disagreement_header": "Désaccord",
        "you_gave": "TOI ({provider}) as donné",
        "other_gave": "L'AUTRE IA a donné",
        "your_reading": "Ta lecture",
        "your_reasoning": "Ton raisonnement",
        "their_reading": "Sa lecture",
        "their_reasoning": "Son raisonnement",
        "difference": "Écart",

        # Warnings
        "rubric_warning": "ATTENTION: Désaccord sur le barème!",
        "reading_warning": "⚠️ DÉSACCORD DE LECTURE",

        # Name section
        "name_section_title": "DÉSACCORDS DE NOMS D'ÉTUDIANTS",
        "name_disagreement_header": "Désaccord de nom",
        "you_read": "TOI as lu",
        "other_read": "L'autre correcteur a lu",

        # Mission
        "mission_title": "TA MISSION",
        "mission_steps": [
            "RELIS l'image de la copie attentivement",
            "COMPARE ta lecture avec celle de l'autre correcteur",
            "DÉCIDE si tu maintiens ta note ou si tu l'ajustes",
            "JUSTIFIE ta décision"
        ],
        "mission_options_title": "IMPORTANT",
        "mission_options": "Tu peux:",
        "mission_option_maintain": "Maintenir ta note si tu es sûr de toi",
        "mission_option_adjust": "Ajuster ta note si l'autre correcteur t'a fait voir quelque chose que tu as manqué",
        "mission_option_change": "Changer complètement si tu réalises une erreur",

        # JSON format
        "response_format_title": "FORMAT DE RÉPONSE (JSON)",
        "json_new_reading": "Réponse corrigée de l'élève",
        "json_reasoning": "Pourquoi j'ai changé/maintenu ma note",
        "json_feedback": "Feedback final pour l'élève (concis et constructif)",

        # Final
        "final_instruction": "Réexamine les copies et retourne ta décision au format JSON.",
        "max_points_note": "Si tu changes ton barème (max_points), indique le nouveau barème dans `my_new_max_points`. Si tu corriges ta lecture, indique la nouvelle lecture dans `my_new_reading`.",
    },

    # Ultimatum prompts
    "ultimatum": {
        "header": "ULTIMATUM - DÉCISION FINALE",
        "intro": "Malgré la vérification croisée, le désaccord PERSISTE avec l'autre correcteur.",
        "must_decide": "Tu dois maintenant prendre une DÉCISION FINALE pour chaque cas.",
        "ultimatum_header": "ULTIMATUM",
        "you_after": "Toi après vérification",
        "other_after": "L'autre correcteur après vérification",
        "persistent_diff": "Écart persistant",
        "rubric_still_warning": "Barème toujours en désaccord",
        "your_reasoning": "Ton raisonnement",
        "their_reasoning": "Son raisonnement",

        # Rules
        "rules_title": "RÈGLES DE L'ULTIMATUM",
        "rules": [
            ("DÉCISION OBLIGATOIRE", "Tu DOIS choisir ta note finale"),
            ("OPTION A - Maintenir", "Si tu es sûr de toi, garde ta note"),
            ("OPTION B - Céder", "Si l'autre correcteur t'a convaincu, accepte sa note"),
            ("OPTION C - Compromis", "Propose une note intermédiaire justifiée")
        ],
        "warning_title": "ATTENTION",
        "warnings": [
            "Si tu es INCERTAIN, abaisse ta confiance (< 0.5)",
            "INTERDICTION de choisir au hasard",
            "Tu DOIS justifier ta décision finale"
        ],

        # JSON
        "json_reasoning": "Pourquoi j'ai pris cette décision finale",
        "final_instruction": "Relis les copies et prends ta DÉCISION FINALE au format JSON.",
    },

    # Student name verification
    "name_verification": {
        "header": "VÉRIFICATION DES NOMS D'ÉTUDIANTS",
        "intro": "Il y a un désaccord sur le nom de l'étudiant entre les deux IA.",
        "instruction": "Relis le nom et décide si tu maintiens ou modifies ta lecture.",
        "final_instruction": "Relis les copies et donne ta réponse au format JSON.",
    },

    # Student name ultimatum
    "name_ultimatum": {
        "header": "ULTIMATUM NOMS D'ÉTUDIANTS - DÉCISION FINALE",
        "intro": "Malgré la vérification, le désaccord sur le nom PERSISTE.",
        "must_decide": "Tu dois maintenant prendre une DÉCISION FINALE.",
        "final_instruction": "Relis les copies et prends ta DÉCISION FINALE au format JSON.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# ENGLISH TRANSLATIONS
# ═══════════════════════════════════════════════════════════════════════════════

TRANSLATIONS_EN = {
    # Batch grading prompts
    "batch": {
        "role": "You are an experienced grader. You must grade {copies_count} in ONE analysis.",
        "approach_intro": """This approach allows you to:
- Ensure CONSISTENCY: same answer = same grade
- Detect PATTERNS: common answers, outliers, potential cheating
- Be EFFICIENT: everything in one pass""",
        "rubric_title": "GRADING RUBRIC",
        "question_text": "Question",
        "criteria_text": "Criteria",
        "max_points_text": "Max Points",
        "not_specified": "Not specified",
        "points_suffix": "",

        # Student detection mode
        "detection_title": "STUDENT DETECTION",
        "detection_intro": "This PDF POTENTIALLY CONTAINS MULTIPLE STUDENTS. You must:",
        "detection_steps": [
            "IDENTIFY how many different students are in this document",
            "DETERMINE which pages belong to each student",
            "READ each student's name (if visible)",
            "GRADE each student separately"
        ],
        "detection_clues": "Clues to identify students:",
        "detection_clue_list": [
            "Name change at the top of a page",
            "Different handwriting style",
            "Same questions answered differently (indicates a new student)",
            "Page break or visual separator"
        ],

        # Copies section
        "copies_title": "COPIES TO GRADE",
        "copies_instruction_detected": "For EACH STUDENT detected",
        "copies_instruction_batch": "For EACH copy",
        "copies_steps": [
            "Read the student's name (if present)",
            "Read the answers to each question",
            "Grade according to the rubric",
            "Provide concise, professional feedback"
        ],

        # Rules
        "rules_title": "IMPORTANT RULES",
        "rules": [
            ("ABSOLUTE CONSISTENCY", "If two students wrote the same answer, they must get the same grade"),
            ("CAREFUL READING", "Use CONTEXT (question, other copies, consistency) to decipher handwriting"),
            ("CONCISE FEEDBACK", "Short, constructive comment adapted to difficulty"),
            ("PATTERN DETECTION", "Note if many students have the same answer (correct or not)"),
            ("CROSS-REFERENCE", "Comparing answers across copies helps you read handwriting and ensure consistency")
        ],

        # JSON format
        "response_format_title": "RESPONSE FORMAT (JSON)",
        "json_copy_index": "1",
        "json_student_name": "Student name or null",
        "json_student_answer": "What the student wrote",
        "json_feedback": "Concise feedback",
        "json_overall_feedback": "General comment on the copy",
        "json_common_answer": "volumetric flask",
        "json_reason_similarity": "Word-for-word identical answers",
        "json_reasoning": "Why this grade",

        # Final
        "final_instruction": "Now analyze the document and return your grading in JSON format.",
    },

    # Dual LLM verification prompts
    "verification": {
        "intro": "You graded copies and a DISAGREEMENT was detected with another grader.",
        "mission_intro": "You must now REEXAMINE your grading considering the other grader's opinion.",
        "disagreements_title": "DISAGREEMENTS TO REEXAMINE",
        "disagreement_header": "Disagreement",
        "you_gave": "YOU gave",
        "other_gave": "THE OTHER GRADER gave",
        "your_reading": "Your reading",
        "your_reasoning": "Your reasoning",
        "their_reading": "Their reading",
        "their_reasoning": "Their reasoning",
        "difference": "Difference",

        # Warnings
        "rubric_warning": "WARNING: Rubric disagreement!",
        "reading_warning": "⚠️ READING DISAGREEMENT",

        # Name section
        "name_section_title": "STUDENT NAME DISAGREEMENTS",
        "name_disagreement_header": "Name Disagreement",
        "you_read": "YOU read",
        "other_read": "THE OTHER GRADER read",

        # Mission
        "mission_title": "YOUR MISSION",
        "mission_steps": [
            "REREAD the copy image carefully",
            "COMPARE your reading with the other AI's",
            "DECIDE whether to maintain or adjust your grade",
            "JUSTIFY your decision"
        ],
        "mission_options_title": "IMPORTANT",
        "mission_options": "You can:",
        "mission_option_maintain": "Maintain your grade if you're confident",
        "mission_option_adjust": "Adjust your grade if the other AI pointed out something you missed",
        "mission_option_change": "Change completely if you realize an error",

        # JSON format
        "response_format_title": "RESPONSE FORMAT (JSON)",
        "json_new_reading": "Corrected student answer",
        "json_reasoning": "Why I changed/maintained my grade",
        "json_feedback": "Final feedback for the student (concise and constructive)",

        # Final
        "final_instruction": "Reexamine the copies and return your decision in JSON format.",
        "max_points_note": "If you change your rubric (max_points), indicate the new rubric in `my_new_max_points`. If you correct your reading, indicate the new reading in `my_new_reading`.",
    },

    # Ultimatum prompts
    "ultimatum": {
        "header": "ULTIMATUM - FINAL DECISION",
        "intro": "Despite cross-verification, the disagreement PERSISTS with the other grader.",
        "must_decide": "You must now make a FINAL DECISION for each case.",
        "ultimatum_header": "ULTIMATUM",
        "you_after": "YOU after verification",
        "other_after": "THE OTHER GRADER after verification",
        "persistent_diff": "Persistent difference",
        "rubric_still_warning": "Rubric still in disagreement",
        "your_reasoning": "Your reasoning",
        "their_reasoning": "Their reasoning",

        # Rules
        "rules_title": "ULTIMATUM RULES",
        "rules": [
            ("MANDATORY DECISION", "You MUST choose your final grade"),
            ("OPTION A - Maintain", "If you're confident, keep your grade"),
            ("OPTION B - Yield", "If the other grader convinced you, accept their grade"),
            ("OPTION C - Compromise", "Propose an intermediate justified grade")
        ],
        "warning_title": "WARNING",
        "warnings": [
            "If UNCERTAIN, lower your confidence (< 0.5)",
            "FORBIDDEN to choose randomly",
            "You MUST justify your final decision"
        ],

        # JSON
        "json_reasoning": "Why I made this final decision",
        "final_instruction": "Reread the copies and make your FINAL DECISION in JSON format.",
    },

    # Student name verification
    "name_verification": {
        "header": "STUDENT NAME VERIFICATION",
        "intro": "There is a disagreement on the student name between the two graders.",
        "instruction": "Re-read the name and decide whether to maintain or change your reading.",
        "final_instruction": "Re-read the copies and provide your response in JSON format.",
    },

    # Student name ultimatum
    "name_ultimatum": {
        "header": "STUDENT NAME ULTIMATUM - FINAL DECISION",
        "intro": "Despite verification, the name disagreement PERSISTS.",
        "must_decide": "You must now make a FINAL DECISION.",
        "final_instruction": "Re-read the copies and make your FINAL DECISION in JSON format.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSLATION REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

TRANSLATIONS = {
    "fr": TRANSLATIONS_FR,
    "en": TRANSLATIONS_EN,
    # Add new languages here:
    # "es": TRANSLATIONS_ES,
    # "de": TRANSLATIONS_DE,
    # "it": TRANSLATIONS_IT,
}


def get_translations(language: str) -> dict:
    """
    Get translations for a language.

    Args:
        language: Language code (fr, en, etc.)

    Returns:
        Translation dictionary

    Raises:
        ValueError: If language is not supported
    """
    if language not in TRANSLATIONS:
        available = ", ".join(TRANSLATIONS.keys())
        raise ValueError(f"Unsupported language: {language}. Available: {available}")
    return TRANSLATIONS[language]
