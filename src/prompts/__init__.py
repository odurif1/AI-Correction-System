"""
Prompt templates for the AI correction system.

Organized by category:
- common: Language detection, system messages, shared utilities
- grading: Main grading prompts (single, multi, vision, auto-detect)
- verification: Verification and ultimatum prompts
- batch: Batch grading prompts
- analysis: Cross-copy analysis and rule extraction
"""

# Common utilities
from prompts.common import (
    detect_language,
    get_system_message,
    get_uncertainty_prompt,
    build_name_detection_prompt,
    FEEDBACK_GUIDELINE_FR,
    FEEDBACK_GUIDELINE_EN,
)

# Grading prompts
from prompts.grading import (
    build_grading_prompt,
    build_vision_grading_prompt,
    build_multi_question_grading_prompt,
    build_auto_detect_grading_prompt,
    build_feedback_prompt,
)

# Verification prompts
from prompts.verification import (
    build_unified_verification_prompt,
    build_unified_ultimatum_prompt,
)

# Batch prompts
from prompts.batch import (
    build_batch_grading_prompt,
    build_dual_llm_verification_prompt,
    build_ultimatum_prompt,
)

# Analysis prompts
from prompts.analysis import (
    build_rule_extraction_prompt,
    build_cross_copy_analysis_prompt,
)

# Annotation prompts
from prompts.annotation import (
    build_direct_annotation_prompt,
    parse_annotation_response,
)


__all__ = [
    # Common
    'detect_language',
    'get_system_message',
    'get_uncertainty_prompt',
    'build_name_detection_prompt',
    'FEEDBACK_GUIDELINE_FR',
    'FEEDBACK_GUIDELINE_EN',
    # Grading
    'build_grading_prompt',
    'build_vision_grading_prompt',
    'build_multi_question_grading_prompt',
    'build_auto_detect_grading_prompt',
    'build_feedback_prompt',
    # Verification
    'build_unified_verification_prompt',
    'build_unified_ultimatum_prompt',
    # Batch
    'build_batch_grading_prompt',
    'build_dual_llm_verification_prompt',
    'build_ultimatum_prompt',
    # Analysis
    'build_rule_extraction_prompt',
    'build_cross_copy_analysis_prompt',
    # Annotation
    'build_direct_annotation_prompt',
    'parse_annotation_response',
]
