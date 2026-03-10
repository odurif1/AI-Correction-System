"""
Page-based matching for dual-LLM grading.

When two LLMs detect different numbers of copies or different page boundaries,
matching by copy_index fails. This module provides page-range-based matching
with validation against the actual PDF structure.

Architecture:
1. VALIDATION: Check each LLM's detection covers the PDF correctly
2. COMPARISON: Compare both LLMs' detections for consistency
3. MATCHING: Match copies by page range overlap (fallback to index)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any


@dataclass
class ValidationResult:
    """Result of validating a detection against PDF structure."""
    is_valid: bool
    coverage_gaps: List[int]      # Pages not covered
    coverage_overlaps: List[int]  # Pages covered multiple times
    invalid_pages: List[int]      # Pages that don't exist in PDF
    warnings: List[str]


@dataclass
class PageMatchResult:
    """Result of matching copies between two LLMs."""
    matches: List[Dict[str, Any]]     # Matched pairs with overlap %
    llm1_unmatched: List[Any]         # Copies only in LLM1
    llm2_unmatched: List[Any]         # Copies only in LLM2
    ambiguous_matches: List[Dict[str, Any]]  # Uncertain matches
    match_method: str                 # "page_range" | "index_fallback"


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_detection_coverage(
    copies: List[Any],
    pdf_page_count: int
) -> ValidationResult:
    """
    Verify that a detection correctly covers the PDF.

    Checks:
    - All pages 1..pdf_page_count are covered
    - No page is covered multiple times
    - No non-existent pages are referenced

    Args:
        copies: List of copy objects with .page_range attribute (tuple or None)
        pdf_page_count: Total number of pages in the PDF

    Returns:
        ValidationResult with validity status and any issues found
    """
    all_pages = set()
    overlaps = []
    warnings = []

    for copy in copies:
        page_range = getattr(copy, 'page_range', None)
        if not page_range:
            continue

        start, end = page_range
        for p in range(start, end + 1):
            if p in all_pages:
                overlaps.append(p)
            all_pages.add(p)

    expected = set(range(1, pdf_page_count + 1))
    gaps = sorted(expected - all_pages)
    invalid = sorted(all_pages - expected)

    if gaps:
        warnings.append(f"Pages non couvertes: {gaps}")
    if overlaps:
        warnings.append(f"Pages en double: {sorted(set(overlaps))}")
    if invalid:
        warnings.append(f"Pages inexistantes: {invalid}")

    return ValidationResult(
        is_valid=not warnings,
        coverage_gaps=gaps,
        coverage_overlaps=sorted(set(overlaps)),
        invalid_pages=invalid,
        warnings=warnings
    )


def compare_llm_detections(
    llm1_copies: List[Any],
    llm2_copies: List[Any],
    pdf_page_count: int
) -> Dict[str, Any]:
    """
    Compare detections from two LLMs and identify issues.

    Args:
        llm1_copies: List of LLM1's copy results
        llm2_copies: List of LLM2's copy results
        pdf_page_count: Total pages in PDF

    Returns:
        Dict with validation results for both LLMs and comparison info
    """
    val1 = validate_detection_coverage(llm1_copies, pdf_page_count)
    val2 = validate_detection_coverage(llm2_copies, pdf_page_count)

    return {
        'llm1_valid': val1.is_valid,
        'llm2_valid': val2.is_valid,
        'llm1_warnings': val1.warnings,
        'llm2_warnings': val2.warnings,
        'llm1_copy_count': len(llm1_copies),
        'llm2_copy_count': len(llm2_copies),
        'copy_count_mismatch': len(llm1_copies) != len(llm2_copies),
        'has_issues': not (val1.is_valid and val2.is_valid)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_overlap(range1: Tuple[int, int], range2: Tuple[int, int]) -> float:
    """
    Calculate overlap percentage between two page ranges.

    Args:
        range1: (start, end) of first range (1-based, inclusive)
        range2: (start, end) of second range (1-based, inclusive)

    Returns:
        Overlap ratio (0.0 = no overlap, 1.0 = identical)
    """
    start1, end1 = range1
    start2, end2 = range2

    # No overlap
    if end1 < start2 or end2 < start1:
        return 0.0

    # Calculate intersection
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_pages = overlap_end - overlap_start + 1

    # Normalize by smaller range
    min_len = min(end1 - start1 + 1, end2 - start2 + 1)
    return overlap_pages / min_len


def match_by_page_ranges(
    llm1_copies: List[Any],
    llm2_copies: List[Any],
    overlap_threshold: float = 0.5
) -> PageMatchResult:
    """
    Match copies by page range overlap.

    Greedy algorithm: for each LLM1 copy, find the best LLM2 match
    with at least overlap_threshold overlap.

    Args:
        llm1_copies: List of LLM1 copy results (need .page_range, .copy_index)
        llm2_copies: List of LLM2 copy results
        overlap_threshold: Minimum overlap ratio to consider a match (default 0.5)

    Returns:
        PageMatchResult with matches, unmatched copies, and ambiguous cases
    """
    matches = []
    llm1_matched = set()
    llm2_matched = set()
    ambiguous = []

    for llm1_copy in llm1_copies:
        llm1_range = getattr(llm1_copy, 'page_range', None)
        if not llm1_range:
            continue

        best_match = None
        best_overlap = 0.0
        second_best = 0.0

        for llm2_copy in llm2_copies:
            llm2_range = getattr(llm2_copy, 'page_range', None)
            if not llm2_range:
                continue

            llm2_idx = getattr(llm2_copy, 'copy_index', id(llm2_copy))
            if llm2_idx in llm2_matched:
                continue

            overlap = calculate_overlap(llm1_range, llm2_range)

            if overlap > best_overlap:
                second_best = best_overlap
                best_overlap = overlap
                best_match = llm2_copy
            elif overlap > second_best:
                second_best = overlap

        llm1_idx = getattr(llm1_copy, 'copy_index', id(llm1_copy))

        if best_match and best_overlap >= overlap_threshold:
            # Ambiguous if second best is close to best
            is_ambiguous = second_best >= (overlap_threshold * 0.8)

            matches.append({
                'llm1_copy': llm1_copy,
                'llm2_copy': best_match,
                'overlap': best_overlap,
                'is_ambiguous': is_ambiguous
            })

            llm1_matched.add(llm1_idx)
            llm2_matched.add(getattr(best_match, 'copy_index', id(best_match)))

            if is_ambiguous:
                ambiguous.append({
                    'llm1_copy': llm1_copy,
                    'best_match': best_match,
                    'best_overlap': best_overlap,
                    'second_best_overlap': second_best
                })

    llm1_unmatched = [c for c in llm1_copies
                      if getattr(c, 'copy_index', id(c)) not in llm1_matched]
    llm2_unmatched = [c for c in llm2_copies
                      if getattr(c, 'copy_index', id(c)) not in llm2_matched]

    return PageMatchResult(
        matches=matches,
        llm1_unmatched=llm1_unmatched,
        llm2_unmatched=llm2_unmatched,
        ambiguous_matches=ambiguous,
        match_method='page_range'
    )


def match_with_fallback(
    llm1_copies: List[Any],
    llm2_copies: List[Any]
) -> PageMatchResult:
    """
    Match copies with fallback: page_range → index.

    First attempts page-based matching if page info is available.
    Falls back to index-based matching if no page info or no matches.

    Args:
        llm1_copies: List of LLM1 copy results
        llm2_copies: List of LLM2 copy results

    Returns:
        PageMatchResult with matches and method used
    """
    # Check if we have page info
    has_page_info = any(
        getattr(c, 'page_range', None) is not None
        for c in llm1_copies + llm2_copies
    )

    if has_page_info:
        result = match_by_page_ranges(llm1_copies, llm2_copies)
        if result.matches:
            return result

    # Fallback: index-based matching
    matches = []
    llm2_by_index = {}
    for c in llm2_copies:
        idx = getattr(c, 'copy_index', None)
        if idx is not None:
            llm2_by_index[idx] = c

    llm2_matched = set()

    for llm1_copy in llm1_copies:
        llm1_idx = getattr(llm1_copy, 'copy_index', None)
        if llm1_idx is None:
            continue

        llm2_copy = llm2_by_index.get(llm1_idx)
        if llm2_copy and llm1_idx not in llm2_matched:
            matches.append({
                'llm1_copy': llm1_copy,
                'llm2_copy': llm2_copy,
                'overlap': 1.0,
                'is_ambiguous': False
            })
            llm2_matched.add(llm1_idx)

    llm1_matched = {m['llm1_copy'].copy_index for m in matches}
    llm1_unmatched = [c for c in llm1_copies
                      if getattr(c, 'copy_index', id(c)) not in llm1_matched]
    llm2_unmatched = [c for c in llm2_copies
                      if getattr(c, 'copy_index', id(c)) not in llm2_matched]

    return PageMatchResult(
        matches=matches,
        llm1_unmatched=llm1_unmatched,
        llm2_unmatched=llm2_unmatched,
        ambiguous_matches=[],
        match_method='index_fallback'
    )
