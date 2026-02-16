"""
Confidence-based utility functions.

Provides helpers for choosing between multiple LLM results based on confidence scores.
"""

from typing import Any, Dict, Optional, Tuple


def choose_by_confidence(
    llm1_result: Dict[str, Any],
    llm2_result: Dict[str, Any],
    field: str,
    confidence_key: str = "confidence",
    default: Any = None
) -> Any:
    """
    Choose a field value from the result with higher confidence.

    This is a common pattern used throughout the codebase when comparing
    results from multiple LLMs.

    Args:
        llm1_result: Result dictionary from first LLM
        llm2_result: Result dictionary from second LLM
        field: The field to extract (e.g., "name", "reading", "grade")
        confidence_key: Key for confidence value (default: "confidence")
        default: Default value if field is not present in either result

    Returns:
        The value of the field from the result with higher confidence

    Example:
        >>> llm1 = {"name": "John", "confidence": 0.9}
        >>> llm2 = {"name": "Jane", "confidence": 0.7}
        >>> choose_by_confidence(llm1, llm2, "name")
        'John'
    """
    conf1 = llm1_result.get(confidence_key, 0)
    conf2 = llm2_result.get(confidence_key, 0)

    if conf1 >= conf2:
        return llm1_result.get(field) or default
    else:
        return llm2_result.get(field) or default


def choose_higher_confidence_result(
    llm1_result: Dict[str, Any],
    llm2_result: Dict[str, Any],
    confidence_key: str = "confidence"
) -> Tuple[Dict[str, Any], int]:
    """
    Return the entire result with higher confidence.

    Args:
        llm1_result: Result dictionary from first LLM
        llm2_result: Result dictionary from second LLM
        confidence_key: Key for confidence value (default: "confidence")

    Returns:
        Tuple of (winning_result, winner_index) where winner_index is 0 or 1
    """
    conf1 = llm1_result.get(confidence_key, 0)
    conf2 = llm2_result.get(confidence_key, 0)

    if conf1 >= conf2:
        return llm1_result, 0
    else:
        return llm2_result, 1


def merge_by_confidence(
    llm1_result: Dict[str, Any],
    llm2_result: Dict[str, Any],
    fields: list,
    confidence_key: str = "confidence"
) -> Dict[str, Any]:
    """
    Merge results by choosing each field from the higher confidence result.

    Args:
        llm1_result: Result dictionary from first LLM
        llm2_result: Result dictionary from second LLM
        fields: List of fields to merge
        confidence_key: Key for confidence value (default: "confidence")

    Returns:
        Merged dictionary with each field from the higher confidence result
    """
    conf1 = llm1_result.get(confidence_key, 0)
    conf2 = llm2_result.get(confidence_key, 0)

    primary = llm1_result if conf1 >= conf2 else llm2_result
    secondary = llm2_result if conf1 >= conf2 else llm1_result

    merged = {}
    for field in fields:
        merged[field] = primary.get(field) or secondary.get(field)

    return merged
