"""
Type guard functions for runtime type checking.

Provides type guards for validating API responses and data structures.
"""

from typing import Any, TypeGuard


def is_dict_with_keys(data: Any, keys: list[str]) -> TypeGuard[dict[str, Any]]:
    """
    Check if data is a dict containing all specified keys.

    Args:
        data: Value to check
        keys: Required keys

    Returns:
        True if data is a dict with all required keys
    """
    return isinstance(data, dict) and all(k in data for k in keys)


def is_grading_result(data: Any) -> TypeGuard[dict[str, Any]]:
    """
    Check if data is a valid grading result.

    A valid grading result contains:
    - grade: float
    - max_points: float
    - confidence: float (optional)
    - reasoning: str (optional)
    """
    if not isinstance(data, dict):
        return False

    required = {'grade', 'max_points'}
    if not required.issubset(data.keys()):
        return False

    # Check types
    if not isinstance(data.get('grade'), (int, float)):
        return False
    if not isinstance(data.get('max_points'), (int, float)):
        return False

    return True


def is_reading_result(data: Any) -> TypeGuard[dict[str, Any]]:
    """
    Check if data is a valid reading result.

    A valid reading result contains:
    - reading: str
    """
    if not isinstance(data, dict):
        return False

    return 'reading' in data and isinstance(data['reading'], str)


def is_question_detection(data: Any) -> TypeGuard[dict[str, Any]]:
    """
    Check if data is a valid question detection result.

    Contains:
    - question_id: str
    - question_text: str (optional)
    - max_points: float (optional)
    """
    if not isinstance(data, dict):
        return False

    return 'question_id' in data and isinstance(data['question_id'], str)


def is_student_name_result(data: Any) -> TypeGuard[dict[str, Any]]:
    """
    Check if data is a valid student name detection result.

    Contains:
    - name: str or None
    - confidence: float (optional)
    """
    if not isinstance(data, dict):
        return False

    name = data.get('name')
    return name is None or isinstance(name, str)


def is_api_error(data: Any) -> TypeGuard[dict[str, Any]]:
    """
    Check if data is an API error response.

    Contains:
    - error: str or dict
    """
    if not isinstance(data, dict):
        return False

    return 'error' in data


def is_list_of[T](data: Any, item_guard: callable[[Any], TypeGuard[T]]) -> TypeGuard[list[T]]:
    """
    Check if data is a list where all items pass the item guard.

    Args:
        data: Value to check
        item_guard: Type guard function for list items

    Returns:
        True if data is a list and all items pass the guard
    """
    if not isinstance(data, list):
        return False

    return all(item_guard(item) for item in data)


def is_scale_detection(data: Any) -> TypeGuard[dict[str, Any]]:
    """
    Check if data is a valid grading scale detection.

    Contains:
    - scales: dict mapping question_id to max_points
    """
    if not isinstance(data, dict):
        return False

    scales = data.get('scales')
    if not isinstance(scales, dict):
        return False

    # Check all values are numeric
    return all(isinstance(v, (int, float)) for v in scales.values())


def ensure_dict(data: Any, default: dict | None = None) -> dict[str, Any]:
    """
    Ensure data is a dict, returning default if not.

    Args:
        data: Value to check
        default: Default dict to return (default: empty dict)

    Returns:
        data if it's a dict, otherwise default or empty dict
    """
    if isinstance(data, dict):
        return data
    return default if default is not None else {}


def ensure_list(data: Any, default: list | None = None) -> list[Any]:
    """
    Ensure data is a list, returning default if not.

    Args:
        data: Value to check
        default: Default list to return (default: empty list)

    Returns:
        data if it's a list, otherwise default or empty list
    """
    if isinstance(data, list):
        return data
    return default if default is not None else []


def ensure_str(data: Any, default: str = "") -> str:
    """
    Ensure data is a string, returning default if not.

    Args:
        data: Value to check
        default: Default string to return

    Returns:
        data if it's a string, otherwise default
    """
    if isinstance(data, str):
        return data
    return default


def ensure_float(data: Any, default: float = 0.0) -> float:
    """
    Ensure data is a float, returning default if not.

    Args:
        data: Value to check
        default: Default float to return

    Returns:
        data converted to float if possible, otherwise default
    """
    try:
        return float(data)
    except (TypeError, ValueError):
        return default
