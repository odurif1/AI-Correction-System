"""
Correction workflow state management.

Provides an immutable state container for the correction workflow.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


class WorkflowPhase(str, Enum):
    """Phases of the correction workflow."""
    INITIALIZATION = "initialization"  # Chargement PDF + découpe + pré-vérification
    GRADING = "grading"                # Correction + détection élèves/questions/langue
    VERIFICATION = "verification"      # Cross-verification
    ULTIMATUM = "ultimatum"            # Résolution désaccords persistants
    CALIBRATION = "calibration"        # Consistency check
    EXPORT = "export"                  # JSON, CSV, analytics
    ANNOTATION = "annotation"          # PDFs annotés + overlays
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class CorrectionState:
    """
    Immutable state container for correction workflow.

    Replaces mutable dict containers (lang_container, jurisprudence)
    with an explicit, immutable state object.

    Usage:
        state = CorrectionState(language='fr', auto_mode=True)
        state = state.with_phase(WorkflowPhase.GRADING)
        state = state.with_jurisprudence('Q1', decision)
    """
    # Language and mode
    language: str = "fr"
    auto_mode: bool = False

    # Current phase
    phase: WorkflowPhase = WorkflowPhase.INITIALIZATION

    # Jurisprudence: store user decisions to inform future grading
    # Format: {question_id: {'decision': grade, 'reasoning': str, 'llm1_grade': x, 'llm2_grade': y}}
    jurisprudence: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Session info
    session_id: Optional[str] = None
    total_copies: int = 0
    processed_copies: int = 0

    # Token usage per phase
    # Format: {phase_name: {'prompt': int, 'completion': int, 'total': int}}
    token_usage_by_phase: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)

    # Chat continuation: session IDs for multi-turn conversations
    # Format: {copy_index: session_id} or {"batch": session_id}
    # Used when chat continuation is enabled to maintain context across phases
    chat_sessions: Dict[str, str] = field(default_factory=dict)

    # Whether chat continuation is enabled
    use_chat_continuation: bool = False

    def with_phase(self, phase: WorkflowPhase) -> 'CorrectionState':
        """
        Create a new state with updated phase.

        Args:
            phase: New workflow phase

        Returns:
            New CorrectionState instance
        """
        return CorrectionState(
            language=self.language,
            auto_mode=self.auto_mode,
            phase=phase,
            jurisprudence=self.jurisprudence.copy(),
            session_id=self.session_id,
            total_copies=self.total_copies,
            processed_copies=self.processed_copies,
            token_usage_by_phase=self.token_usage_by_phase.copy(),
            errors=self.errors.copy(),
            chat_sessions=self.chat_sessions.copy(),
            use_chat_continuation=self.use_chat_continuation
        )

    def with_token_usage(
        self,
        phase: WorkflowPhase,
        prompt_tokens: int,
        completion_tokens: int
    ) -> 'CorrectionState':
        """
        Record token usage for a phase.

        Args:
            phase: Phase to record tokens for
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            New CorrectionState instance
        """
        new_usage = self.token_usage_by_phase.copy()
        phase_name = phase.value

        if phase_name not in new_usage:
            new_usage[phase_name] = {'prompt': 0, 'completion': 0, 'total': 0}

        new_usage[phase_name]['prompt'] += prompt_tokens
        new_usage[phase_name]['completion'] += completion_tokens
        new_usage[phase_name]['total'] += prompt_tokens + completion_tokens

        return CorrectionState(
            language=self.language,
            auto_mode=self.auto_mode,
            phase=self.phase,
            jurisprudence=self.jurisprudence.copy(),
            session_id=self.session_id,
            total_copies=self.total_copies,
            processed_copies=self.processed_copies,
            token_usage_by_phase=new_usage,
            errors=self.errors.copy(),
            chat_sessions=self.chat_sessions.copy(),
            use_chat_continuation=self.use_chat_continuation
        )

    def get_token_summary(self) -> Dict[str, Any]:
        """Get token usage summary by phase."""
        total_prompt = 0
        total_completion = 0

        for phase_usage in self.token_usage_by_phase.values():
            total_prompt += phase_usage.get('prompt', 0)
            total_completion += phase_usage.get('completion', 0)

        return {
            'by_phase': self.token_usage_by_phase.copy(),
            'total_prompt': total_prompt,
            'total_completion': total_completion,
            'total': total_prompt + total_completion
        }

    def with_language(self, language: str) -> 'CorrectionState':
        """Create a new state with updated language."""
        return CorrectionState(
            language=language,
            auto_mode=self.auto_mode,
            phase=self.phase,
            jurisprudence=self.jurisprudence.copy(),
            session_id=self.session_id,
            total_copies=self.total_copies,
            processed_copies=self.processed_copies,
            token_usage_by_phase=self.token_usage_by_phase.copy(),
            errors=self.errors.copy(),
            chat_sessions=self.chat_sessions.copy(),
            use_chat_continuation=self.use_chat_continuation
        )

    def with_jurisprudence(
        self,
        question_id: str,
        decision: float,
        reasoning: str = "",
        llm1_grade: float = None,
        llm2_grade: float = None
    ) -> 'CorrectionState':
        """
        Create a new state with added jurisprudence entry.

        Args:
            question_id: Question identifier
            decision: Final grade decision
            reasoning: Explanation for the decision
            llm1_grade: Grade from LLM1
            llm2_grade: Grade from LLM2

        Returns:
            New CorrectionState instance
        """
        new_jurisprudence = self.jurisprudence.copy()
        new_jurisprudence[question_id] = {
            'decision': decision,
            'reasoning': reasoning,
            'llm1_grade': llm1_grade,
            'llm2_grade': llm2_grade
        }
        return CorrectionState(
            language=self.language,
            auto_mode=self.auto_mode,
            phase=self.phase,
            jurisprudence=new_jurisprudence,
            session_id=self.session_id,
            total_copies=self.total_copies,
            processed_copies=self.processed_copies,
            token_usage_by_phase=self.token_usage_by_phase.copy(),
            errors=self.errors.copy(),
            chat_sessions=self.chat_sessions.copy(),
            use_chat_continuation=self.use_chat_continuation
        )

    def with_error(self, error: Exception, context: Dict[str, Any] = None) -> 'CorrectionState':
        """Create a new state with added error."""
        new_errors = self.errors.copy()
        new_errors.append({
            'error': str(error),
            'type': type(error).__name__,
            'context': context or {}
        })
        return CorrectionState(
            language=self.language,
            auto_mode=self.auto_mode,
            phase=self.phase,
            jurisprudence=self.jurisprudence.copy(),
            session_id=self.session_id,
            total_copies=self.total_copies,
            processed_copies=self.processed_copies,
            token_usage_by_phase=self.token_usage_by_phase.copy(),
            errors=new_errors,
            chat_sessions=self.chat_sessions.copy(),
            use_chat_continuation=self.use_chat_continuation
        )

    def with_progress(self, processed: int, total: int = None) -> 'CorrectionState':
        """Create a new state with updated progress."""
        return CorrectionState(
            language=self.language,
            auto_mode=self.auto_mode,
            phase=self.phase,
            jurisprudence=self.jurisprudence.copy(),
            session_id=self.session_id,
            total_copies=total if total is not None else self.total_copies,
            processed_copies=processed,
            token_usage_by_phase=self.token_usage_by_phase.copy(),
            errors=self.errors.copy(),
            chat_sessions=self.chat_sessions.copy(),
            use_chat_continuation=self.use_chat_continuation
        )

    def with_chat_session(self, session_key: str, session_id: str) -> 'CorrectionState':
        """
        Add or update a chat session mapping.

        Args:
            session_key: Key for the session (e.g., "copy_1", "batch")
            session_id: The chat session ID

        Returns:
            New CorrectionState instance
        """
        new_sessions = self.chat_sessions.copy()
        new_sessions[session_key] = session_id
        return CorrectionState(
            language=self.language,
            auto_mode=self.auto_mode,
            phase=self.phase,
            jurisprudence=self.jurisprudence.copy(),
            session_id=self.session_id,
            total_copies=self.total_copies,
            processed_copies=self.processed_copies,
            token_usage_by_phase=self.token_usage_by_phase.copy(),
            errors=self.errors.copy(),
            chat_sessions=new_sessions,
            use_chat_continuation=self.use_chat_continuation
        )

    def with_chat_continuation(self, enabled: bool = True) -> 'CorrectionState':
        """
        Enable or disable chat continuation mode.

        Args:
            enabled: Whether to enable chat continuation

        Returns:
            New CorrectionState instance
        """
        return CorrectionState(
            language=self.language,
            auto_mode=self.auto_mode,
            phase=self.phase,
            jurisprudence=self.jurisprudence.copy(),
            session_id=self.session_id,
            total_copies=self.total_copies,
            processed_copies=self.processed_copies,
            token_usage_by_phase=self.token_usage_by_phase.copy(),
            errors=self.errors.copy(),
            chat_sessions=self.chat_sessions.copy(),
            use_chat_continuation=enabled
        )

    def get_chat_session(self, session_key: str) -> Optional[str]:
        """Get chat session ID by key."""
        return self.chat_sessions.get(session_key)

    def has_chat_session(self, session_key: str) -> bool:
        """Check if a chat session exists."""
        return session_key in self.chat_sessions

    def get_jurisprudence(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Get jurisprudence for a question."""
        return self.jurisprudence.get(question_id)

    def has_jurisprudence(self, question_id: str) -> bool:
        """Check if jurisprudence exists for a question."""
        return question_id in self.jurisprudence

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            'language': self.language,
            'auto_mode': self.auto_mode,
            'phase': self.phase.value,
            'jurisprudence': self.jurisprudence.copy(),
            'session_id': self.session_id,
            'total_copies': self.total_copies,
            'processed_copies': self.processed_copies,
            'token_usage_by_phase': self.token_usage_by_phase.copy(),
            'errors': self.errors.copy(),
            'chat_sessions': self.chat_sessions.copy(),
            'use_chat_continuation': self.use_chat_continuation
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CorrectionState':
        """Create state from dictionary."""
        return cls(
            language=data.get('language', 'fr'),
            auto_mode=data.get('auto_mode', False),
            phase=WorkflowPhase(data.get('phase', 'initialization')),
            jurisprudence=data.get('jurisprudence', {}),
            session_id=data.get('session_id'),
            total_copies=data.get('total_copies', 0),
            processed_copies=data.get('processed_copies', 0),
            token_usage_by_phase=data.get('token_usage_by_phase', {}),
            errors=data.get('errors', []),
            chat_sessions=data.get('chat_sessions', {}),
            use_chat_continuation=data.get('use_chat_continuation', False)
        )
