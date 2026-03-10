import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.models import GradedCopy, SessionStatus
from utils.sorting import question_sort_key

logger = logging.getLogger(__name__)


@dataclass
class GradingContext:
    """All state needed by graders, extracted from the orchestrator."""
    session: Any  # GradingSession
    store: Any  # FileStore
    ai: Any  # AI provider (single or comparison)
    grading_mode: str
    comparison_mode: bool
    second_reading: bool
    parallel: int
    detected_questions: Dict[str, str]
    grading_scale: Dict[str, float]
    analysis_complete: bool
    pages_per_copy: Optional[int] = None
    # Orchestrator reference for attributes not yet fully extracted
    # (batch_verify, use_chat_continuation, disagreement_callback, etc.)
    orchestrator: Any = None

    def get_orchestrator_attr(self, name: str, default=None):
        """Safely get an attribute from the orchestrator."""
        if self.orchestrator is not None:
            return getattr(self.orchestrator, name, default)
        return default


class BaseGrader(ABC):
    """Abstract base for all grading strategies."""

    def __init__(self, ctx: GradingContext):
        self.ctx = ctx

    @abstractmethod
    async def grade_all(self, progress_callback=None) -> List[GradedCopy]:
        ...

    # ── Shared utilities ──

    def _save_sync(self, last_graded=None):
        """Save session state to storage. If last_graded is provided, only save that copy."""
        self.ctx.store.save_session(self.ctx.session)
        if last_graded:
            self.ctx.store.save_graded_copy(last_graded, last_graded.copy_id)
        else:
            for graded in self.ctx.session.graded_copies:
                self.ctx.store.save_graded_copy(graded, graded.copy_id)

    async def _call_callback(self, callback, event_type: str, data: dict):
        """Safely call a callback, handling both sync and async."""
        if callback is None:
            return
        try:
            result = callback(event_type, data)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.warning(f"Callback error for {event_type}: {e}")

    def _build_questions_list(self) -> list:
        """Build the questions list from detected_questions + grading_scale."""
        questions = []
        for q_id in sorted(self.ctx.detected_questions.keys(), key=question_sort_key):
            q_text = self.ctx.detected_questions[q_id]
            questions.append({
                "id": q_id,
                "text": q_text,
                "criteria": "",
                "max_points": self.ctx.grading_scale.get(q_id, 1.0)
            })
        return questions
