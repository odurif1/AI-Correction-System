"""
Shared response parser for AI providers.

Parses structured grading responses in a provider-independent way.
Expects JSON output from the AI.
"""

import json
import re
from typing import Dict, Any, List

def _strip_markdown_json(text: str) -> str:
    """Remove markdown formatting around JSON if present."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def parse_grading_response(response: str) -> Dict[str, Any]:
    """
    Parse structured JSON grading response from AI.

    Expected JSON format:
    {
      "grade": 5.0,
      "confidence": 0.9,
      "uncertainty_type": "none",
      "if_uncertain": "",
      "student_answer_read": "what was read from student's response",
      "reasoning": "technical analysis",
      "student_feedback": "pedagogical feedback"
    }

    Args:
        response: Raw JSON response from AI (potentially wrapped in markdown)

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

    cleaned_response = _strip_markdown_json(response)

    try:
        data = json.loads(cleaned_response)
        
        # Extract fields safely
        if "grade" in data:
            try:
                # Sometimes LLMs might output "5/5" instead of just 5 even in JSON
                if isinstance(data["grade"], str) and "/" in data["grade"]:
                    result["grade"] = float(data["grade"].split("/")[0])
                elif data["grade"] is not None:
                    result["grade"] = float(data["grade"])
            except (ValueError, TypeError):
                pass
                
        if "confidence" in data and data["confidence"] is not None:
            try:
                result["confidence"] = float(data["confidence"])
            except (ValueError, TypeError):
                pass
                
        if "uncertainty_type" in data:
            result["uncertainty_type"] = str(data["uncertainty_type"]).lower()
            
        if "if_uncertain" in data:
            result["if_uncertain"] = str(data["if_uncertain"])
            
        if "student_answer_read" in data:
            result["student_answer_read"] = str(data["student_answer_read"])
            
        if "reasoning" in data:
            result["reasoning"] = str(data["reasoning"])
            
        if "student_feedback" in data:
            result["student_feedback"] = str(data["student_feedback"])
            
    except json.JSONDecodeError:
        # Fallback for old format or catastrophic JSON failure
        # For simplicity in this fallback, we just log it as reasoning
        result["reasoning"] = f"JSON PARSE ERROR. Raw output:\n{response}"

    return result
