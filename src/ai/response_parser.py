"""
Shared response parser for AI providers.

Parses structured grading responses in a provider-independent way.
Handles multi-line content for fields like INTERNAL_REASONING.
"""

from typing import Dict, Any, List


# Fields that can span multiple lines (until next field starts)
MULTILINE_FIELDS = {
    "INTERNAL_REASONING": "reasoning",
    "STUDENT_FEEDBACK": "student_feedback",
    "STUDENT_ANSWER_READ": "student_answer_read",
    "IF_UNCERTAIN": "if_uncertain"
}

# Single-line fields with their result keys
SINGLE_LINE_FIELDS = {
    "GRADE": "grade",
    "CONFIDENCE": "confidence",
    "UNCERTAINTY_TYPE": "uncertainty_type"
}


def parse_grading_response(response: str) -> Dict[str, Any]:
    """
    Parse structured grading response from AI.

    Expected format:
    GRADE: 5/5
    CONFIDENCE: 0.9
    UNCERTAINTY_TYPE: none
    IF_UNCERTAIN: N/A
    STUDENT_ANSWER_READ: what was read from student's response
    INTERNAL_REASONING: technical analysis (can be multi-line)
    STUDENT_FEEDBACK: pedagogical feedback (can be multi-line)

    Args:
        response: Raw text response from AI

    Returns:
        Dict with parsed fields
    """
    result = {
        "grade": None,
        "confidence": 0.5,
        "confidence_factors": [],
        "uncertainty_type": "other",
        "if_uncertain": "",
        "student_answer_read": "",
        "reasoning": "",
        "student_feedback": ""
    }

    if not response:
        return result

    lines = response.strip().split('\n')

    current_field = None
    current_content = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Check if this line starts a multi-line field
        is_new_field = False
        for field_prefix, result_key in MULTILINE_FIELDS.items():
            if stripped.startswith(field_prefix + ":"):
                # Save previous field content
                if current_field and current_content:
                    result[current_field] = " ".join(current_content)
                # Start new field
                current_field = result_key
                current_content = [stripped.split(":", 1)[1].strip()]
                is_new_field = True
                break

        if is_new_field:
            continue

        # Check for single-line fields
        if stripped.startswith("GRADE:"):
            _save_current_field(result, current_field, current_content)
            current_field = None
            current_content = []
            try:
                grade_str = stripped.split("GRADE:")[1].strip()
                if "/" in grade_str:
                    result["grade"] = float(grade_str.split("/")[0])
                else:
                    result["grade"] = float(grade_str)
            except (ValueError, IndexError):
                pass

        elif stripped.startswith("CONFIDENCE:"):
            _save_current_field(result, current_field, current_content)
            current_field = None
            current_content = []
            try:
                result["confidence"] = float(stripped.split("CONFIDENCE:")[1].strip())
            except (ValueError, IndexError):
                pass

        elif stripped.startswith("UNCERTAINTY_TYPE:"):
            _save_current_field(result, current_field, current_content)
            current_field = None
            current_content = []
            result["uncertainty_type"] = stripped.split("UNCERTAINTY_TYPE:")[1].strip().lower()

        elif stripped.startswith("-") and ":" in stripped:
            # Confidence factor line
            result["confidence_factors"].append(stripped[1:].strip())

        elif current_field:
            # Continue capturing multi-line content
            current_content.append(stripped)

    # Save last field
    _save_current_field(result, current_field, current_content)

    return result


def _save_current_field(result: Dict, current_field: str, current_content: List[str]):
    """Save current multi-line field to result."""
    if current_field and current_content:
        result[current_field] = " ".join(current_content)
