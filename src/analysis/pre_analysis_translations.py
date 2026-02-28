"""
Translation strings for PDF pre-analysis prompts.

Add new languages by creating a new dictionary with the same structure.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FRENCH TRANSLATIONS
# ═══════════════════════════════════════════════════════════════════════════════

TRANSLATIONS_FR = {
    "pre_analysis": {
        "role": "Tu es un assistant d'analyse de documents PDF pour la correction scolaire.",

        # Mission
        "mission_title": "TA MISSION",
        "mission_intro": "Analyse ce document PDF pour déterminer sa structure et son contenu. Tu dois détecter:",
        "detection_steps": [
            "TYPE DE DOCUMENT: S'agit-il de copies d'élèves, d'un sujet, ou d'un document sans rapport ?",
            "STRUCTURE: Un seul élève par PDF ou plusieurs élèves dans ce PDF ? ATTENTION: analyse TOUTES les pages!",
            "SUJET: Le sujet est-il inclus dans le PDF ou séparé ?",
            "ÉLÈVES: Combien d'élèves ? Pour chaque élève: nom, page de début, page de fin",
            "BARÈME: Quels sont les points attribués à chaque question ?",
            "LANGUE: Quelle est la langue principale du document ?",
            "NOM DE L'EXAMEN: Génère un nom court pour cet examen (format: 'Matière - Type', max 50 caractères, ex: 'Mathématiques - Contrôle')"
        ],

        # Critical instructions
        "critical_title": "⚠️ INSTRUCTIONS CRITIQUES",
        "critical_instructions": [
            "Analyse TOUTES les pages du PDF, pas seulement les premières!",
            "Pour détecter plusieurs élèves: cherche les CHANGEMENTS DE NOM entre les pages",
            "Chaque nouvelle copie commence généralement par un nom d'élève différent",
            "Si tu vois des noms différents sur des pages différentes = plusieurs élèves",
            "Structure 'one_pdf_all_students' = plusieurs élèves dans UN SEUL fichier PDF",
            "Structure 'one_pdf_one_student' = un seul élève dans tout le PDF"
        ],

        # Blocking criteria
        "blocking_title": "PROBLÈMES BLOQUANTS",
        "blocking_intro": "Détecte ces problèmes qui empêchent la correction:",
        "blocking_criteria": [
            "Ce n'est PAS des copies d'élèves (document aléatoire, facture, etc.)",
            "Le PDF est corrompu ou illisible",
            "La qualité est trop faible pour lire les réponses",
            "Impossible de déterminer la structure du document",
            "Aucun sujet détecté (ni intégré, ni référence claire)"
        ],

        # Quality issues
        "quality_title": "PROBLÈMES DE QUALITÉ (non-bloquants)",
        "quality_intro": "Détecte ces problèmes qui peuvent affecter la qualité:",
        "quality_issues": [
            "Écriture difficile à lire",
            "Pages tournées ou mal alignées",
            "Taches ou marques sur le document",
            "Qualité de scan faible",
            "Noms d'élèves illisibles"
        ],

        # Response format
        "response_format_title": "FORMAT DE RÉPONSE (JSON)",

        # JSON field descriptions
        "field_descriptions_title": "Descriptions des champs",
        "field_document_type": "Type de document détecté",
        "field_structure": "Structure du PDF (un élève ou plusieurs)",
        "field_subject_integration": "Comment le sujet est intégré",
        "field_grading_scale": "Barème détecté: {question_id: points}",
        "field_blocking_issues": "Liste des problèmes bloquants (vide = OK)",
        "field_warnings": "Liste des avertissements (non-bloquants)",
        "field_exam_name": "Nom court de l'examen (ex: 'Mathématiques - Contrôle')",

        # Document types
        "document_types_title": "Types de documents",
        "doc_type_student_copies": "Copies d'élèves à corriger",
        "doc_type_subject_only": "Uniquement le sujet (pas de copies)",
        "doc_type_random": "Document sans rapport (facture, lettre, etc.)",
        "doc_type_unclear": "Impossible à déterminer",

        # Structure types
        "structure_types_title": "Types de structure",
        "structure_one_student": "Un seul élève dans ce PDF",
        "structure_all_students": "Plusieurs élèves dans ce PDF",
        "structure_ambiguous": "Structure ambiguë",

        # JSON examples
        "json_student_name": "Jean Dupont",
        "json_other_student": "Marie Martin",

        # Final
        "final_instruction": "Analyse TOUTES les pages du document et retourne ton analyse au format JSON.",
    },

    "quick_analysis": {
        "role": "Tu es un assistant d'analyse rapide de documents PDF.",
        "mission": "Détermine rapidement la structure de ce document PDF.",
        "page_count_label": "Nombre de pages",
        "final_instruction": "Retourne uniquement le JSON.",
    },

    # UI labels (for backend messages)
    "ui": {
        "analysis_in_progress": "Analyse du PDF en cours...",
        "analysis_complete": "Analyse terminée",
        "blocking_issue_detected": "Problème bloquant détecté",
        "no_blocking_issues": "Aucun problème bloquant",
        "students_detected": "élèves détectés",
        "grading_scale_detected": "Barème détecté",
        "subject_separate": "Sujet séparé détecté",
        "quality_warning": "Avertissement qualité",
    },

    # Quality issue translations (English -> French)
    "quality_issue_translations": {
        "Pages are not perfectly aligned": "Les pages ne sont pas parfaitement alignées",
        "Some handwriting is difficult to read": "Certaines écritures sont difficiles à lire",
        "Hard to read handwriting": "Écriture difficile à lire",
        "Rotated or misaligned pages": "Pages tournées ou mal alignées",
        "Stains or marks on the document": "Taches ou marques sur le document",
        "Low scan quality": "Qualité de scan faible",
        "Illegible student names": "Noms d'élèves illisibles",
        "Low quality scan": "Scan de basse qualité",
        "Blurry images": "Images floues",
        "Incomplete pages": "Pages incomplètes",
        "Missing pages": "Pages manquantes",
        "Watermarks interfering": "Filigranes gênants",
        "Dark or light spots": "Taches sombres ou claires",
        "Text cut off": "Texte coupé",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# ENGLISH TRANSLATIONS
# ═══════════════════════════════════════════════════════════════════════════════

TRANSLATIONS_EN = {
    "pre_analysis": {
        "role": "You are a PDF document analysis assistant for school grading.",

        # Mission
        "mission_title": "YOUR MISSION",
        "mission_intro": "Analyze this PDF document to determine its structure and content. You must detect:",
        "detection_steps": [
            "DOCUMENT TYPE: Is this student copies, a subject, or an unrelated document?",
            "STRUCTURE: One student per PDF or multiple students in this PDF?",
            "SUBJECT: Is the subject included in the PDF or separate?",
            "STUDENTS: How many students and which pages for each?",
            "GRADING SCALE: What are the points assigned to each question?",
            "LANGUAGE: What is the main language of the document?",
            "EXAM NAME: Generate a short name for this exam (format: 'Subject - Type', max 50 chars, ex: 'Mathematics - Quiz')"
        ],

        # Blocking criteria
        "blocking_title": "BLOCKING ISSUES",
        "blocking_intro": "Detect these issues that prevent grading:",
        "blocking_criteria": [
            "This is NOT student copies (random document, invoice, etc.)",
            "The PDF is corrupted or unreadable",
            "Quality is too low to read answers",
            "Cannot determine document structure",
            "No subject detected (neither integrated nor clearly referenced)"
        ],

        # Quality issues
        "quality_title": "QUALITY ISSUES (non-blocking)",
        "quality_intro": "Detect these issues that may affect quality:",
        "quality_issues": [
            "Hard to read handwriting",
            "Rotated or misaligned pages",
            "Stains or marks on the document",
            "Low scan quality",
            "Illegible student names"
        ],

        # Response format
        "response_format_title": "RESPONSE FORMAT (JSON)",

        # JSON field descriptions
        "field_descriptions_title": "Field descriptions",
        "field_document_type": "Detected document type",
        "field_structure": "PDF structure (one or multiple students)",
        "field_subject_integration": "How the subject is integrated",
        "field_grading_scale": "Detected grading scale: {question_id: points}",
        "field_blocking_issues": "List of blocking issues (empty = OK)",
        "field_warnings": "List of warnings (non-blocking)",
        "field_exam_name": "Short exam name (ex: 'Mathematics - Quiz')",

        # Document types
        "document_types_title": "Document types",
        "doc_type_student_copies": "Student copies to grade",
        "doc_type_subject_only": "Subject only (no copies)",
        "doc_type_random": "Unrelated document (invoice, letter, etc.)",
        "doc_type_unclear": "Cannot determine",

        # Structure types
        "structure_types_title": "Structure types",
        "structure_one_student": "One student in this PDF",
        "structure_all_students": "Multiple students in this PDF",
        "structure_ambiguous": "Ambiguous structure",

        # JSON examples
        "json_student_name": "John Smith",
        "json_other_student": "Jane Doe",

        # Final
        "final_instruction": "Analyze the document and return your analysis in JSON format.",
    },

    "quick_analysis": {
        "role": "You are a quick PDF document analysis assistant.",
        "mission": "Quickly determine the structure of this PDF document.",
        "page_count_label": "Number of pages",
        "final_instruction": "Return only the JSON.",
    },

    # UI labels (for backend messages)
    "ui": {
        "analysis_in_progress": "Analyzing PDF...",
        "analysis_complete": "Analysis complete",
        "blocking_issue_detected": "Blocking issue detected",
        "no_blocking_issues": "No blocking issues",
        "students_detected": "students detected",
        "grading_scale_detected": "Grading scale detected",
        "subject_separate": "Separate subject detected",
        "quality_warning": "Quality warning",
    },

    # Quality issue translations (English -> English, for consistency)
    "quality_issue_translations": {
        "Pages are not perfectly aligned": "Pages are not perfectly aligned",
        "Some handwriting is difficult to read": "Some handwriting is difficult to read",
        "Hard to read handwriting": "Hard to read handwriting",
        "Rotated or misaligned pages": "Rotated or misaligned pages",
        "Stains or marks on the document": "Stains or marks on the document",
        "Low scan quality": "Low scan quality",
        "Illegible student names": "Illegible student names",
        "Low quality scan": "Low quality scan",
        "Blurry images": "Blurry images",
        "Incomplete pages": "Incomplete pages",
        "Missing pages": "Missing pages",
        "Watermarks interfering": "Watermarks interfering",
        "Dark or light spots": "Dark or light spots",
        "Text cut off": "Text cut off",
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


def translate_quality_issue(issue: str, language: str = "fr") -> str:
    """
    Translate a quality issue string to the target language.

    Args:
        issue: The quality issue string (usually in English from LLM)
        language: Target language code (fr, en)

    Returns:
        Translated quality issue string
    """
    translations = get_translations(language)
    quality_translations = translations.get("quality_issue_translations", {})

    # Try exact match first
    if issue in quality_translations:
        return quality_translations[issue]

    # Try case-insensitive match
    issue_lower = issue.lower()
    for key, value in quality_translations.items():
        if key.lower() == issue_lower:
            return value

    # Try partial match (if issue contains key words)
    for key, value in quality_translations.items():
        if key.lower() in issue_lower or issue_lower in key.lower():
            return value

    # Return original if no translation found
    return issue
