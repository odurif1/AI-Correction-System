"""
Batch grading prompts for the AI correction system.

This module contains all prompt templates for batch grading operations:
- Initial batch grading
- Dual LLM verification
- Ultimatum round
- Grouped and per-question verification

To add a new language:
1. Create translations in batch_translations.py
2. Register in TRANSLATIONS dictionary
"""

from typing import List, Dict, Any

from prompts.batch_translations import get_translations


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH GRADING PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

def build_batch_grading_prompt(
    copies_data: List[Dict[str, Any]],
    questions: Dict[str, Dict[str, Any]],
    language: str = "fr",
    detect_students: bool = False
) -> str:
    """
    Build a prompt for batch grading all copies at once.

    Args:
        copies_data: List of copy data with images and metadata
        questions: Dict of {question_id: {text, criteria, max_points}}
        language: Language for prompts
        detect_students: If True, ask LLM to detect multiple students in the PDF

    Returns:
        Complete prompt string
    """
    t = get_translations(language)["batch"]

    # Build questions section
    questions_text = _build_questions_section(questions, t)

    # Build student detection section (if needed)
    student_detection = _build_detection_section(detect_students, t)

    # Determine copies instruction
    if detect_students:
        copies_instruction = t["copies_instruction_detected"]
    else:
        copies_instruction = t["copies_instruction_batch"]

    # Build copies steps
    copies_steps = "\n".join(f"{i+1}. {step}" for i, step in enumerate(t["copies_steps"]))

    # Build rules section
    rules = "\n".join(
        f'{i+1}. **{title}**: {desc}'
        for i, (title, desc) in enumerate(t["rules"])
    )

    # Build JSON example
    json_example = _build_batch_json_example(t)

    # Build rubric section only if questions are defined
    rubric_section = ""
    if questions_text.strip():
        rubric_section = f"""
# {t['rubric_title']}

{questions_text}

═══════════════════════════════════════════════════════════════════
"""

    return f"""{t['role']}

═══════════════════════════════════════════════════════════════════
{rubric_section}{student_detection}
# {t['copies_title']}

{copies_instruction}:
{copies_steps}

═══════════════════════════════════════════════════════════════════

# {t['rules_title']}

{rules}

═══════════════════════════════════════════════════════════════════

# {t['response_format_title']}

{json_example}

═══════════════════════════════════════════════════════════════════

{t['final_instruction']}
"""


def _build_questions_section(questions: Dict[str, Dict[str, Any]], t: dict) -> str:
    """Build the questions/rubric section."""
    lines = []
    for qid, qdata in questions.items():
        text = qdata.get('text', t['not_specified'])
        criteria = qdata.get('criteria', t['not_specified'])
        max_points = qdata.get('max_points', 1)
        points_suffix = t.get('points_suffix', '')
        if points_suffix:
            points_text = f"{max_points} {points_suffix}"
        else:
            points_text = str(max_points)

        lines.append(f"""## {qid}
**{t['question_text']}:** {text}
**{t['criteria_text']}:** {criteria}
**{t['max_points_text']}:** {points_text}

""")
    return "".join(lines)


def _build_detection_section(detect_students: bool, t: dict) -> str:
    """Build the student detection section if needed."""
    if not detect_students:
        return ""

    steps = "\n".join(f"{i+1}. **{step}**" for i, step in enumerate(t["detection_steps"]))
    clues = "\n".join(f"- {clue}" for clue in t["detection_clue_list"])

    return f"""
# {t['detection_title']}

{t['detection_intro']}

{steps}

{t['detection_clues']}
{clues}
"""


def _build_batch_json_example(t: dict) -> str:
    """Build the JSON example for batch grading."""
    return f"""```json
{{
  "copies": [
    {{
      "copy_index": {t['json_copy_index']},
      "student_name": "{t['json_student_name']}",
      "pages": [1, 2],
      "questions": {{
        "Q1": {{
          "student_answer_read": "{t['json_student_answer']}",
          "grade": "{t['json_grade_placeholder']}",
          "max_points": "{t['json_max_points_placeholder']}",
          "confidence": "<0.0 à 1.0>",
          "reasoning": "{t['json_reasoning']}",
          "feedback": "{t['json_feedback']}"
        }},
        "Q2": {{ ... }}
      }},
      "overall_feedback": "{t['json_overall_feedback']}"
    }},
    {{
      "copy_index": 2,
      "student_name": "{t['json_other_student']}",
      "pages": [3, 4],
      ...
    }}
  ],
  "patterns": {{
    "Q1": {{
      "common_answer": "{t['json_common_answer']}",
      "frequency": "75%",
      "is_correct": true
    }},
    "suspicious_similarities": [
      {{"copies": [1, 2], "question": "Q4", "reason": "{t['json_reason_similarity']}"}}
    ]
  }}
}}
```"""


# ═══════════════════════════════════════════════════════════════════════════════
# DUAL LLM VERIFICATION PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

def build_dual_llm_verification_prompt(
    disagreements: List[Any],
    provider_name: str,
    other_provider_name: str,
    is_own_perspective: bool = True,
    language: str = "fr",
    name_disagreements: List[Dict[str, Any]] = None
) -> str:
    """
    Build a verification prompt for ONE LLM showing both LLMs' grades.

    Args:
        disagreements: List of Disagreement objects to verify
        provider_name: Name of the LLM receiving this prompt
        other_provider_name: Name of the other LLM
        is_own_perspective: If True, this LLM's grades are shown as "you"
        language: Language for prompts
        name_disagreements: Optional list of student name disagreements (for grouped mode)

    Returns:
        Complete prompt string
    """
    t = get_translations(language)["verification"]

    # Build disagreements text
    disagreements_text = _build_disagreements_section(
        disagreements, provider_name, other_provider_name, is_own_perspective, t
    )

    # Build name disagreements section if present
    name_section = _build_name_disagreements_section(
        name_disagreements, provider_name, t
    )

    # Build mission steps
    mission_steps = "\n".join(f"{i+1}. {step}" for i, step in enumerate(t["mission_steps"]))

    # Build JSON example
    json_example = _build_verification_json_example(t, name_disagreements is not None and len(name_disagreements) > 0)

    return f"""{t['intro']}

{t['mission_intro']}

═══════════════════════════════════════════════════════════════════

# {t['disagreements_title']}

{disagreements_text}{name_section}
═══════════════════════════════════════════════════════════════════

# {t['mission_title']}

{mission_steps}

**{t['mission_options_title']}**: {t['mission_options']}
- {t['mission_option_maintain']}
- {t['mission_option_adjust']}
- {t['mission_option_change']}

═══════════════════════════════════════════════════════════════════

# {t['response_format_title']}

{json_example}

**{t['mission_options_title']}**: {t['max_points_note']}

═══════════════════════════════════════════════════════════════════

{t['final_instruction']}
"""


def _build_disagreements_section(
    disagreements: List[Any],
    provider_name: str,
    other_provider_name: str,
    is_own_perspective: bool,
    t: dict
) -> str:
    """Build the disagreements listing section."""
    import logging
    logger = logging.getLogger(__name__)

    lines = []
    for i, d in enumerate(disagreements, 1):
        # Determine grades based on perspective
        your_grade, your_max_pts, your_reading, your_reasoning, \
        other_grade, other_max_pts, other_reading, other_reasoning = \
            _extract_disagreement_data(d, provider_name, is_own_perspective)

        # Get disagreement type
        disp_type = getattr(d, 'disagreement_type', 'grade')

        # Build warnings (only reading warning - no rubric warning since barème is frozen)
        warnings = []
        if disp_type in ("reading", "both"):
            warnings.append(f"\n**{t['reading_warning']}**: {t['your_reading']}: \"{your_reading}\" | {t['their_reading']}: \"{other_reading}\"")

        warnings_text = "".join(warnings)

        # Use "L'AUTRE IA" without provider name (not relevant for grading)
        # Include max_points in grade display for context
        lines.append(f"""## {t['disagreement_header']} {i}: Copie {d.copy_index}, {d.question_id}

**{t['you_gave'].format(provider=provider_name)}**: **{your_grade}/{your_max_pts}** pts
- {t['your_reading']}: "{your_reading}"
- {t['your_reasoning']}: {your_reasoning}

**{t['other_gave']}**: **{other_grade}/{other_max_pts}** pts
- {t['their_reading']}: "{other_reading}"
- {t['their_reasoning']}: {other_reasoning}

{t['difference']}: {abs(your_grade - other_grade)} points{warnings_text}

""")

    return "".join(lines)


def _extract_disagreement_data(d: Any, provider_name: str, is_own_perspective: bool) -> tuple:
    """Extract disagreement data based on provider perspective.

    Note: max_points is frozen (from pre-analysis), not detected by LLMs.
    """
    # Get frozen max_points (same for both LLMs)
    frozen_max_pts = d.max_points

    if is_own_perspective:
        is_llm1 = (d.llm1_name == provider_name)
        your_grade = d.llm1_grade if is_llm1 else d.llm2_grade
        your_max_pts = frozen_max_pts  # Use frozen barème
        your_reading = d.llm1_reading if is_llm1 else d.llm2_reading
        your_reasoning = d.llm1_reasoning if is_llm1 else d.llm2_reasoning
        other_grade = d.llm2_grade if is_llm1 else d.llm1_grade
        other_max_pts = frozen_max_pts  # Use frozen barème
        other_reading = d.llm2_reading if is_llm1 else d.llm1_reading
        other_reasoning = d.llm2_reasoning if is_llm1 else d.llm1_reasoning
    else:
        your_grade = d.llm2_grade
        your_max_pts = frozen_max_pts  # Use frozen barème
        your_reading = d.llm2_reading
        your_reasoning = d.llm2_reasoning
        other_grade = d.llm1_grade
        other_max_pts = frozen_max_pts  # Use frozen barème
        other_reading = d.llm1_reading
        other_reasoning = d.llm1_reasoning

    return (your_grade, your_max_pts, your_reading, your_reasoning,
            other_grade, other_max_pts, other_reading, other_reasoning)


def _build_name_disagreements_section(
    name_disagreements: List[Dict[str, Any]],
    provider_name: str,
    t: dict
) -> str:
    """Build the name disagreements section if present."""
    if not name_disagreements:
        return ""

    lines = []
    for i, d in enumerate(name_disagreements, 1):
        is_llm1 = (d.get('llm1_provider') == provider_name)
        your_name = d.get('llm1_name') if is_llm1 else d.get('llm2_name')
        other_name = d.get('llm2_name') if is_llm1 else d.get('llm1_name')

        other_provider_display = t['other_grader_label']
        lines.append(f"""## {t['name_disagreement_header']} {i}: Copie {d['copy_index']}

**{t['you_read'].format(provider=provider_name)}**: **"{your_name}"**
**{t['other_read'].format(provider=other_provider_display)}**: **"{other_name}"**

""")

    return f"""
═══════════════════════════════════════════════════════════════════

# {t['name_section_title']}

{"".join(lines)}"""


def _build_verification_json_example(t: dict, has_name_disagreements: bool) -> str:
    """Build the JSON example for verification."""
    name_verification_format = ""
    if has_name_disagreements:
        name_verification_format = """,
  "name_verifications": [
    {
      "copy_index": 1,
      "my_new_name": "Jean Du Pont",
      "confidence": 0.9
    },
    ...
  ]"""

    return """```json
{
  "verifications": [
    {
      "copy_index": 1,
      "question_id": "Q1",
      "my_new_grade": 0.5,
      "my_new_reading": "%s",
      "reasoning": "%s",
      "feedback": "%s",
      "confidence": 0.9
    },
    ...
  ]%s
}
```""" % (t['json_new_reading'], t['json_reasoning'], t['json_feedback'], name_verification_format)


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
    t = get_translations(language)["ultimatum"]

    # Build disagreements text
    disagreements_text = _build_ultimatum_disagreements_section(
        disagreements, provider_name, t
    )

    # Build rules
    rules = "\n".join(
        f'{i+1}. **{title}**: {desc}'
        for i, (title, desc) in enumerate(t["rules"])
    )

    # Build warnings
    warnings = "\n".join(f"- {w}" for w in t["warnings"])

    # Build JSON example with translatable strings
    json_example = f"""```json
{{
  "ultimatum_decisions": [
    {{
      "copy_index": 1,
      "question_id": "Q1",
      "my_final_grade": 0.5,
      "my_final_reading": "{t['json_final_reading']}",
      "decision": {t['json_decision_options']},
      "reasoning": "{t['json_reasoning']}",
      "feedback": "{t['json_feedback']}",
      "confidence": 0.9
    }},
    ...
  ]
}}
```"""

    return f"""═══════════════════════════════════════════════════════════════════
{t['header']}
═══════════════════════════════════════════════════════════════════

{t['intro']}
{t['must_decide']}

{disagreements_text}

═══════════════════════════════════════════════════════════════════

# {t['rules_title']}

{rules}

**{t['warning_title']}**:
{warnings}

═══════════════════════════════════════════════════════════════════

# RESPONSE FORMAT (JSON)

{json_example}

═══════════════════════════════════════════════════════════════════

{t['final_instruction']}
"""


def _build_ultimatum_disagreements_section(
    disagreements: List[Dict[str, Any]],
    provider_name: str,
    t: dict
) -> str:
    """Build the ultimatum disagreements section."""
    lines = []
    for i, d in enumerate(disagreements, 1):
        # Use frozen max_points (barème is frozen)
        frozen_max = d.get('max_points', 1)

        other_provider_display = t['other_grader_label']
        lines.append(f"""## {t['ultimatum_header']} {i}: Copie {d['copy_index']}, {d['question_id']}

**{t['you_after']}**: **{d['llm1_grade']}/{frozen_max}** pts
- {t['your_reasoning']}: {d.get('llm1_reasoning', '')}

**{t['other_after']}**: **{d['llm2_grade']}/{frozen_max}** pts
- {t['their_reasoning']}: {d.get('llm2_reasoning', '')}

{t['persistent_diff']}: {abs(d['llm1_grade'] - d['llm2_grade'])} points

""")

    return "".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# STUDENT NAME VERIFICATION PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

def build_student_name_verification_prompt(
    name_disagreements: List[Dict[str, Any]],
    provider_name: str,
    other_provider_name: str,
    language: str = "fr"
) -> str:
    """
    Build verification prompt for student name disagreements.

    Args:
        name_disagreements: List of dicts with copy_index, llm1_name, llm2_name
        provider_name: Name of the LLM receiving this prompt
        other_provider_name: Name of the other LLM
        language: Language for prompts

    Returns:
        Complete prompt string
    """
    t = get_translations(language)["name_verification"]
    vt = get_translations(language)["verification"]  # For shared strings

    # Build disagreements text
    disagreements_text = ""
    for i, d in enumerate(name_disagreements, 1):
        disagreements_text += f"""## {t.get('disagreement_header', 'DÉSACCORD')} {i}: Copie {d['copy_index']}

**{vt['you_read']}**: **"{d['llm1_name']}"**
**{vt['other_read']}**: **"{d['llm2_name']}"**

"""

    return f"""═══════════════════════════════════════════════════════════════════
{t['header']}
═══════════════════════════════════════════════════════════════════

{t['intro']}
{t['instruction']}

{disagreements_text}
# RESPONSE FORMAT (JSON)

```json
{{
  "name_verifications": [
    {{
      "copy_index": 1,
      "my_new_name": "Jean Du Pont",
      "confidence": 0.9
    }},
    ...
  ]
}}
```

{t['final_instruction']}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# STUDENT NAME ULTIMATUM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

def build_student_name_ultimatum_prompt(
    persistent_disagreements: List[Dict[str, Any]],
    provider_name: str,
    other_provider_name: str,
    language: str = "fr"
) -> str:
    """
    Build ultimatum prompt for persistent student name disagreements.

    Args:
        persistent_disagreements: List of disagreements that persist after verification
        provider_name: Name of the LLM receiving this prompt
        other_provider_name: Name of the other LLM
        language: Language for prompts

    Returns:
        Complete prompt string
    """
    t = get_translations(language)["name_ultimatum"]
    vt = get_translations(language)["verification"]  # For shared strings

    # Build disagreements text
    disagreements_text = ""
    for i, d in enumerate(persistent_disagreements, 1):
        disagreements_text += f"""## {t.get('ultimatum_header', 'ULTIMATUM')} {i}: Copie {d['copy_index']}

**{vt['you_read']}**: **"{d['llm1_name']}"**
**{vt['other_read']}**: **"{d['llm2_name']}"**

"""

    return f"""═══════════════════════════════════════════════════════════════════
{t['header']}
═══════════════════════════════════════════════════════════════════

{t['intro']}
{t['must_decide']}

{disagreements_text}
# RESPONSE FORMAT (JSON)

```json
{{
  "name_ultimatum_decisions": [
    {{
      "copy_index": 1,
      "my_final_name": "Jean Dupont",
      "confidence": 0.9
    }},
    ...
  ]
}}
```

{t['final_instruction']}
"""
