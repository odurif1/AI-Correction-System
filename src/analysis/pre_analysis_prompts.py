"""
Prompts for PDF pre-analysis.

This module contains prompts for analyzing PDF structure and content
before grading to detect:
- Document type (student copies, subject only, random document)
- PDF structure (one student per PDF or all students in one PDF)
- Grading scale / barème
- Student names and page assignments
- Quality issues and blocking problems
"""

from typing import List, Dict, Any
from analysis.pre_analysis_translations import get_translations


def build_pre_analysis_prompt(
    language: str = "fr"
) -> str:
    """
    Build the prompt for pre-analyzing a PDF.

    Args:
        language: Language for prompts (fr, en)

    Returns:
        Complete prompt string
    """
    t = get_translations(language)["pre_analysis"]

    # Build detection steps
    detection_steps = "\n".join(
        f"{i+1}. **{step}**"
        for i, step in enumerate(t["detection_steps"])
    )

    # Build critical instructions
    critical_instructions = "\n".join(
        f"   - {instruction}"
        for instruction in t.get("critical_instructions", [])
    )

    # Build blocking criteria
    blocking_criteria = "\n".join(
        f"- {criterion}"
        for criterion in t["blocking_criteria"]
    )

    # Build quality issues
    quality_issues = "\n".join(
        f"- {issue}"
        for issue in t["quality_issues"]
    )

    # Build JSON example
    json_example = _build_json_example(t)

    return f"""{t['role']}

═══════════════════════════════════════════════════════════════════

# {t['mission_title']}

{t['mission_intro']}

{detection_steps}

═══════════════════════════════════════════════════════════════════

# {t.get('critical_title', '⚠️ INSTRUCTIONS CRITIQUES')}

{critical_instructions}

═══════════════════════════════════════════════════════════════════

# {t['blocking_title']}

{t['blocking_intro']}

{blocking_criteria}

═══════════════════════════════════════════════════════════════════

# {t['quality_title']}

{t['quality_intro']}

{quality_issues}

═══════════════════════════════════════════════════════════════════

# {t['response_format_title']}

{json_example}

═══════════════════════════════════════════════════════════════════

{t['final_instruction']}
"""


def _build_json_example(t: dict) -> str:
    """Build the JSON example for pre-analysis response."""
    return f"""```json
{{
  "document_type": "student_copies",
  "confidence_document_type": 0.95,

  "structure": "one_pdf_all_students",
  "subject_integration": "integrated",
  "num_students_detected": 3,
  "students": [
    {{
      "index": 1,
      "name": "{t['json_student_name']}",
      "start_page": 1,
      "end_page": 2,
      "confidence": 0.9
    }},
    {{
      "index": 2,
      "name": "{t['json_other_student']}",
      "start_page": 3,
      "end_page": 4,
      "confidence": 0.8
    }},
    {{
      "index": 3,
      "name": null,
      "start_page": 5,
      "end_page": 6,
      "confidence": 0.6
    }}
  ],

  "grading_scale": {{
    "Q1": 2.0,
    "Q2": 3.0,
    "Q3": 2.5,
    "Q4": 4.0
  }},
  "confidence_grading_scale": 0.85,
  "questions_detected": ["Q1", "Q2", "Q3", "Q4"],

  "quality_issues": [],
  "overall_quality_score": 0.9,

  "blocking_issues": [],
  "has_blocking_issues": false,
  "warnings": [],

  "detected_language": "fr",
  "exam_name": "Mathématiques - Contrôle"
}}
```

**{t['field_descriptions_title']}**:
- `document_type`: {t['field_document_type']}
- `structure`: {t['field_structure']}
- `subject_integration`: {t['field_subject_integration']}
- `grading_scale`: {t['field_grading_scale']}
- `blocking_issues`: {t['field_blocking_issues']}
- `warnings`: {t['field_warnings']}
- `exam_name`: {t['field_exam_name']}

**{t['document_types_title']}**:
- `student_copies`: {t['doc_type_student_copies']}
- `subject_only`: {t['doc_type_subject_only']}
- `random_document`: {t['doc_type_random']}
- `unclear`: {t['doc_type_unclear']}

**{t['structure_types_title']}**:
- `one_pdf_one_student`: {t['structure_one_student']}
- `one_pdf_all_students`: {t['structure_all_students']}
- `ambiguous`: {t['structure_ambiguous']}
"""


def build_quick_structure_prompt(
    page_count: int,
    language: str = "fr"
) -> str:
    """
    Build a quick prompt to detect PDF structure only.

    This is a lighter-weight analysis for faster results.

    Args:
        page_count: Number of pages in the PDF
        language: Language for prompts

    Returns:
        Simplified prompt string
    """
    t = get_translations(language)["quick_analysis"]

    return f"""{t['role']}

{t['mission']}

**{t['page_count_label']}**: {page_count}

```json
{{
  "document_type": "student_copies",
  "structure": "one_pdf_all_students",
  "num_students": 3,
  "confidence": 0.85,
  "has_blocking_issues": false
}}
```

{t['final_instruction']}
"""
