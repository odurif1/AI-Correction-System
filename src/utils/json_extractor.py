"""
JSON extraction utilities for LLM responses.

Provides a common function to extract JSON from various LLM response formats.
Handles markdown code blocks, raw JSON, and edge cases.
"""

import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def extract_json_from_response(raw_response: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from an LLM response.

    Handles multiple formats:
    - ```json code blocks
    - ``` code blocks (without language specifier)
    - Raw JSON objects embedded in text

    Args:
        raw_response: The raw text response from an LLM

    Returns:
        Parsed JSON dictionary, or None if extraction/parsing fails
    """
    if not raw_response:
        return None

    json_match = raw_response.strip()

    # Try ```json block first (most common format)
    if '```json' in json_match:
        json_start = json_match.find('```json') + 7
        json_end = json_match.find('```', json_start)
        if json_end > json_start:
            json_match = json_match[json_start:json_end].strip()

    # Try ``` code block (without language specifier)
    elif '```' in json_match:
        json_start = json_match.find('```') + 3
        # Skip language identifier if present (e.g., ```python)
        while json_start < len(json_match) and json_match[json_start] not in '\n\r{':
            json_start += 1
        if json_start < len(json_match) and json_match[json_start] in '\n\r':
            json_start += 1
        json_end = json_match.find('```', json_start)
        if json_end > json_start:
            json_match = json_match[json_start:json_end].strip()

    # Find JSON object bounds (first { to last })
    brace_start = json_match.find('{')
    brace_end = json_match.rfind('}')

    if brace_start >= 0 and brace_end > brace_start:
        json_str = json_match[brace_start:brace_end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse error: {e}")
            # Try to fix common issues
            return _try_repair_and_parse(json_str)

    return None


def _try_repair_and_parse(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to repair common JSON issues and parse.

    Args:
        json_str: JSON string that failed to parse

    Returns:
        Parsed JSON dictionary, or None if repair fails
    """
    import re

    # Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', json_str)

    # Replace smart quotes with regular quotes
    repaired = repaired.replace('"', '"').replace('"', '"')
    repaired = repaired.replace(''', "'").replace(''', "'")

    # Remove control characters except newline and tab
    repaired = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', repaired)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def extract_json_list_from_response(raw_response: str) -> Optional[list]:
    """
    Extract a JSON array from an LLM response.

    Similar to extract_json_from_response but looks for arrays [...] instead
    of objects {...}.

    Args:
        raw_response: The raw text response from an LLM

    Returns:
        Parsed JSON list, or None if extraction/parsing fails
    """
    if not raw_response:
        return None

    json_match = raw_response.strip()

    # Try ```json block first
    if '```json' in json_match:
        json_start = json_match.find('```json') + 7
        json_end = json_match.find('```', json_start)
        if json_end > json_start:
            json_match = json_match[json_start:json_end].strip()

    # Try ``` code block
    elif '```' in json_match:
        json_start = json_match.find('```') + 3
        while json_start < len(json_match) and json_match[json_start] not in '\n\r[':
            json_start += 1
        if json_start < len(json_match) and json_match[json_start] in '\n\r':
            json_start += 1
        json_end = json_match.find('```', json_start)
        if json_end > json_start:
            json_match = json_match[json_start:json_end].strip()

    # Find JSON array bounds (first [ to last ])
    bracket_start = json_match.find('[')
    bracket_end = json_match.rfind(']')

    if bracket_start >= 0 and bracket_end > bracket_start:
        json_str = json_match[bracket_start:bracket_end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    return None
