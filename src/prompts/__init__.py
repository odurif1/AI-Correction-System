"""
Prompt templates for the AI correction system.

Organized by grading mode:
- batch: Prompts for batch grading (all copies at once)
- individual: Prompts for individual grading (one copy at a time)
- verification: Prompts for verification and ultimatum rounds
"""

from prompts.batch import (
    build_batch_grading_prompt,
    build_dual_llm_verification_prompt,
    build_ultimatum_prompt,
    build_grouped_verification_prompt,
    build_per_question_verification_prompt,
)

__all__ = [
    'build_batch_grading_prompt',
    'build_dual_llm_verification_prompt',
    'build_ultimatum_prompt',
    'build_grouped_verification_prompt',
    'build_per_question_verification_prompt',
]
