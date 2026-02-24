"""
Name matching utilities for cross-verifying student names between LLMs.
"""

from difflib import SequenceMatcher
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class NameMatchResult:
    """Result of matching names between two LLMs."""
    matches: List[Dict[str, Any]]  # List of matched copies
    mismatches: List[Dict[str, Any]]  # List of mismatched copies
    llm1_only: List[int]  # Copy indices only in LLM1
    llm2_only: List[int]  # Copy indices only in LLM2
    all_matched: bool  # True if all copies matched
    requires_user_action: bool  # True if user needs to intervene


def fuzzy_match(name1: str, name2: str, threshold: float = 0.85) -> bool:
    """
    Check if two names are similar enough to be considered the same.

    Args:
        name1: First name
        name2: Second name
        threshold: Similarity threshold (0-1)

    Returns:
        True if names match above threshold
    """
    if not name1 or not name2:
        return False

    # Normalize names
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()

    # Exact match
    if n1 == n2:
        return True

    # Fuzzy match using SequenceMatcher
    ratio = SequenceMatcher(None, n1, n2).ratio()
    return ratio >= threshold


def cross_verify_student_names(
    llm1_copies: List[Any],
    llm2_copies: List[Any],
    require_exact_match: bool = True
) -> NameMatchResult:
    """
    Cross-verify student names between two LLM results.

    Args:
        llm1_copies: List of BatchCopyResult from LLM1
        llm2_copies: List of BatchCopyResult from LLM2
        require_exact_match: If True, only exact matches are considered valid (default)

    Returns:
        NameMatchResult with match information
    """
    matches = []
    mismatches = []
    llm1_indices = {c.copy_index for c in llm1_copies}
    llm2_indices = {c.copy_index for c in llm2_copies}

    # Find copies only in one LLM
    llm1_only = sorted(llm1_indices - llm2_indices)
    llm2_only = sorted(llm2_indices - llm1_indices)

    # Build lookup for LLM2
    llm2_by_index = {c.copy_index: c for c in llm2_copies}

    # Compare copies by index
    for llm1_copy in llm1_copies:
        copy_idx = llm1_copy.copy_index
        llm2_copy = llm2_by_index.get(copy_idx)

        if not llm2_copy:
            continue  # Handled by llm1_only

        name1 = llm1_copy.student_name or f"Élève {copy_idx}"
        name2 = llm2_copy.student_name or f"Élève {copy_idx}"

        # Check for match - EXACT ONLY
        is_exact = name1.lower().strip() == name2.lower().strip()

        # If require_exact_match, only exact matches count
        # Otherwise, fall back to fuzzy matching for informational purposes
        is_fuzzy = False
        if not is_exact and not require_exact_match:
            is_fuzzy = fuzzy_match(name1, name2, 0.85)

        match_info = {
            'copy_index': copy_idx,
            'llm1_name': name1,
            'llm2_name': name2,
            'match_type': 'exact' if is_exact else ('fuzzy' if is_fuzzy else 'none'),
            'similarity': SequenceMatcher(None, name1.lower(), name2.lower()).ratio() if not is_exact else 1.0
        }

        if is_exact or is_fuzzy:
            matches.append(match_info)
        else:
            mismatches.append(match_info)

    # Determine if user action is needed
    requires_user_action = bool(mismatches) or bool(llm1_only) or bool(llm2_only)
    all_matched = not requires_user_action

    return NameMatchResult(
        matches=matches,
        mismatches=mismatches,
        llm1_only=llm1_only,
        llm2_only=llm2_only,
        all_matched=all_matched,
        requires_user_action=requires_user_action
    )


def format_name_mismatch_message(result: NameMatchResult, language: str = "fr") -> str:
    """
    Format a user-friendly message about name mismatches.

    Args:
        result: NameMatchResult from cross_verify_student_names
        language: Language for the message

    Returns:
        Formatted message string
    """
    if language == "fr":
        lines = ["⚠️  PROBLÈME DE DÉTECTION DES ÉLÈVES"]
        lines.append("=" * 50)

        if result.mismatches:
            lines.append("\nLes deux LLMs ont détecté des noms différents:")
            for m in result.mismatches:
                lines.append(f"  Copie {m['copy_index']}:")
                lines.append(f"    LLM1: \"{m['llm1_name']}\"")
                lines.append(f"    LLM2: \"{m['llm2_name']}\"")
                lines.append(f"    Similarité: {m['similarity']:.0%}")

        if result.llm1_only:
            lines.append(f"\nCopies détectées uniquement par LLM1: {result.llm1_only}")

        if result.llm2_only:
            lines.append(f"\nCopies détectées uniquement par LLM2: {result.llm2_only}")

        lines.append("\n" + "-" * 50)
        lines.append("SOLUTIONS SUGGÉRÉES:")
        lines.append("  1. Utilisez --pages-per-copy N pour un découpage mécanique")
        lines.append("  2. Utilisez --auto-detect-structure pour pré-analyser")
        lines.append("=" * 50)

    else:
        lines = ["⚠️  STUDENT DETECTION PROBLEM"]
        lines.append("=" * 50)

        if result.mismatches:
            lines.append("\nThe two LLMs detected different names:")
            for m in result.mismatches:
                lines.append(f"  Copy {m['copy_index']}:")
                lines.append(f"    LLM1: \"{m['llm1_name']}\"")
                lines.append(f"    LLM2: \"{m['llm2_name']}\"")
                lines.append(f"    Similarity: {m['similarity']:.0%}")

        if result.llm1_only:
            lines.append(f"\nCopies detected only by LLM1: {result.llm1_only}")

        if result.llm2_only:
            lines.append(f"\nCopies detected only by LLM2: {result.llm2_only}")

        lines.append("\n" + "-" * 50)
        lines.append("SUGGESTED SOLUTIONS:")
        lines.append("  1. Use --pages-per-copy N for mechanical splitting")
        lines.append("  2. Use --auto-detect-structure for pre-analysis")
        lines.append("=" * 50)

    return "\n".join(lines)
