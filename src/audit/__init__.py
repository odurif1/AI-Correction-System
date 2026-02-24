"""
Audit module for unified grading audit structure.

This module provides the AuditBuilder class for constructing
unified grading audit structures that work across all modes:
- Single LLM and Dual LLM
- All verification modes (grouped, per-copy, per-question, none)
"""

from audit.builder import AuditBuilder

__all__ = ['AuditBuilder']
