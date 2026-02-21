"""
Prompt templates - backwards compatibility module.

This module re-exports all prompts from the new prompts/ package.
New code should import directly from prompts instead.

Example:
    # Old (deprecated)
    from config.prompts import build_grading_prompt

    # New (recommended)
    from prompts import build_grading_prompt
"""

# Re-export everything from prompts module for backwards compatibility
from prompts import (
    # Common
    detect_language,
    get_system_message,
    get_uncertainty_prompt,
    build_name_detection_prompt,
    FEEDBACK_GUIDELINE_FR,
    FEEDBACK_GUIDELINE_EN,
    # Grading
    build_grading_prompt,
    build_vision_grading_prompt,
    build_multi_question_grading_prompt,
    build_auto_detect_grading_prompt,
    build_feedback_prompt,
    # Verification
    build_unified_verification_prompt,
    build_unified_ultimatum_prompt,
    # Batch
    build_batch_grading_prompt,
    build_dual_llm_verification_prompt,
    build_ultimatum_prompt,
    # Analysis
    build_rule_extraction_prompt,
    build_cross_copy_analysis_prompt,
)

__all__ = [
    'detect_language',
    'get_system_message',
    'get_uncertainty_prompt',
    'build_name_detection_prompt',
    'FEEDBACK_GUIDELINE_FR',
    'FEEDBACK_GUIDELINE_EN',
    'build_grading_prompt',
    'build_vision_grading_prompt',
    'build_multi_question_grading_prompt',
    'build_auto_detect_grading_prompt',
    'build_feedback_prompt',
    'build_unified_verification_prompt',
    'build_unified_ultimatum_prompt',
    'build_batch_grading_prompt',
    'build_dual_llm_verification_prompt',
    'build_ultimatum_prompt',
    'build_rule_extraction_prompt',
    'build_cross_copy_analysis_prompt',
]
