"""
Sorting utilities for the AI correction system.

Provides natural sorting for question IDs and other identifiers.
"""

import re
from typing import List, Any


def natural_sort_key(s: Any) -> List:
    """
    Sort key for natural sorting of question IDs.

    Sorts Q1, Q2, Q10 instead of Q1, Q10, Q2 (lexicographic).

    Args:
        s: Value to generate sort key for (will be converted to string)

    Returns:
        List suitable for use as a sort key

    Examples:
        >>> sorted(['Q1', 'Q10', 'Q2'], key=natural_sort_key)
        ['Q1', 'Q2', 'Q10']
    """
    def convert(text: str) -> int | str:
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split(r'(\d+)', str(s))]


def question_sort_key(s: Any) -> tuple:
    """
    Sort key specifically for question IDs like Q1, Q2, Q10.

    Returns a tuple (type_order, numeric_value) for proper sorting.

    Args:
        s: Question ID string (e.g., "Q1", "Q10")

    Returns:
        Tuple for sorting: (0, 1) for Q1, (0, 10) for Q10, (1, "q...") for non-Q

    Examples:
        >>> sorted(['Q1', 'Q10', 'Q2', 'A1'], key=question_sort_key)
        ['Q1', 'Q2', 'Q10', 'A1']
    """
    match = re.match(r'Q(\d+)', str(s))
    if match:
        return (0, int(match.group(1)))
    return (1, str(s).lower())
