"""
Real-time progress display for parallel copy processing.

Provides a Rich Live dashboard that shows the status of all copies
being processed in parallel, with per-copy progress and scores.

Features:
- Multi-panel layout with global stats
- ETA estimation
- Agreements/disagreements tracking
- Recent results display
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from time import time
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.progress import BarColumn, Progress, TextColumn


@dataclass
class CopyStatus:
    """Status of a single copy being processed."""
    student_name: str = "???"
    status: str = "pending"  # pending, processing, done, error
    questions_done: int = 0
    questions_total: int = 0
    score: Optional[float] = None
    max_score: Optional[float] = None
    error: Optional[str] = None


@dataclass
class SessionStats:
    """Global session statistics."""
    agreements: int = 0
    disagreements: int = 0
    total_questions: int = 0
    avg_confidence: float = 0.0
    start_time: float = field(default_factory=time)
    recent_results: List[Dict] = field(default_factory=list)  # Last 5 results


class LiveProgressDisplay:
    """
    Real-time dashboard for parallel copy grading.

    Displays a table showing all copies with their status, progress,
    and scores. Updates in real-time as copies are processed.

    Usage:
        display = LiveProgressDisplay(console, total_copies=12)
        with display:
            # In callback:
            display.update_copy(1, student_name="Jean", status="processing")
            display.update_copy(1, questions_done=3, questions_total=5)
            display.update_copy(1, status="done", score=12.5, max_score=20)
    """

    STATUS_ICONS = {
        "pending": "[dim]⏳ En attente[/dim]",
        "processing": "[yellow]⚙ En cours[/yellow]",
        "done": "[green]✓ Terminé[/green]",
        "error": "[red]✗ Erreur[/red]"
    }

    # For English display
    STATUS_ICONS_EN = {
        "pending": "[dim]⏳ Waiting[/dim]",
        "processing": "[yellow]⚙ Processing[/yellow]",
        "done": "[green]✓ Done[/green]",
        "error": "[red]✗ Error[/red]"
    }

    def __init__(
        self,
        console: Console,
        total_copies: int,
        language: str = "fr"
    ):
        """
        Initialize the live progress display.

        Args:
            console: Rich console to use for display
            total_copies: Total number of copies to process
            language: Display language ("fr" or "en")
        """
        self.console = console
        self.total_copies = total_copies
        self.language = language
        self.copies: Dict[int, CopyStatus] = {}
        self.completed = 0
        self.errors = 0
        self._live: Optional[Live] = None

        # New stats tracking
        self.stats = SessionStats()

        # Track currently processing copy for display
        self.current_copy: Optional[int] = None
        self.current_question: int = 0

        # Initialize all copies as pending
        for i in range(1, total_copies + 1):
            self.copies[i] = CopyStatus()

    def update_copy(self, copy_index: int, **kwargs):
        """
        Update status of a specific copy.

        Args:
            copy_index: 1-based index of the copy
            **kwargs: Fields to update (student_name, status, questions_done,
                     questions_total, score, max_score, error)
        """
        current = self.copies.get(copy_index, CopyStatus())

        # Track status changes for counters
        old_status = current.status
        new_status = kwargs.get('status', old_status)

        # Update only provided fields
        for key, value in kwargs.items():
            if hasattr(current, key):
                setattr(current, key, value)

        self.copies[copy_index] = current

        # Update counters based on status change
        if old_status != new_status:
            if old_status == 'done':
                self.completed -= 1
            elif old_status == 'error':
                self.errors -= 1

            if new_status == 'done':
                self.completed += 1
            elif new_status == 'error':
                self.errors += 1

        if self._live:
            self._live.update(self._render())

    def mark_processing(self, copy_index: int, student_name: str = None, questions_total: int = 0):
        """Convenience method to mark a copy as processing."""
        self.update_copy(
            copy_index,
            status="processing",
            student_name=student_name or "???",
            questions_total=questions_total
        )

    def mark_question_done(self, copy_index: int):
        """Convenience method to increment questions done for a copy."""
        current = self.copies.get(copy_index, CopyStatus())
        self.update_copy(
            copy_index,
            questions_done=current.questions_done + 1
        )

    def mark_done(self, copy_index: int, score: float, max_score: float):
        """Convenience method to mark a copy as done with score."""
        status = self.copies.get(copy_index, CopyStatus())
        self.update_copy(
            copy_index,
            status="done",
            score=score,
            max_score=max_score,
            questions_done=status.questions_total
        )
        # Track in recent results
        self._add_recent_result(copy_index, score, max_score)

    def mark_current_activity(self, copy_index: int, question_num: int = 0, question_total: int = 0):
        """Update current processing activity."""
        self.current_copy = copy_index
        self.current_question = question_num
        if question_total > 0:
            self.update_copy(copy_index, questions_total=question_total)

    def record_agreement(self, is_agreement: bool = True):
        """Record LLM agreement or disagreement."""
        if is_agreement:
            self.stats.agreements += 1
        else:
            self.stats.disagreements += 1
        self.stats.total_questions += 1

    def update_confidence(self, confidence: float):
        """Update running average confidence."""
        if self.stats.total_questions > 0:
            # Rolling average
            self.stats.avg_confidence = (
                (self.stats.avg_confidence * (self.stats.total_questions - 1) + confidence)
                / self.stats.total_questions
            )
        else:
            self.stats.avg_confidence = confidence

    def _add_recent_result(self, copy_index: int, score: float, max_score: float):
        """Add a result to recent results list."""
        status = self.copies.get(copy_index, CopyStatus())
        self.stats.recent_results.append({
            'name': status.student_name,
            'score': score,
            'max': max_score,
            'pct': (score / max_score * 100) if max_score > 0 else 0
        })
        # Keep only last 5
        if len(self.stats.recent_results) > 5:
            self.stats.recent_results.pop(0)

    def _get_eta(self) -> str:
        """Calculate estimated time remaining."""
        if self.completed == 0:
            return "..."
        elapsed = time() - self.stats.start_time
        avg_time = elapsed / self.completed
        remaining = self.total_copies - self.completed
        eta_seconds = avg_time * remaining

        if eta_seconds < 60:
            return f"{int(eta_seconds)}s"
        elif eta_seconds < 3600:
            return f"{int(eta_seconds / 60)}m {int(eta_seconds % 60)}s"
        else:
            return f"{int(eta_seconds / 3600)}h {int((eta_seconds % 3600) / 60)}m"

    def mark_error(self, copy_index: int, error: str):
        """Convenience method to mark a copy as having an error."""
        self.update_copy(
            copy_index,
            status="error",
            error=error
        )

    def _get_status_icon(self, status: str) -> str:
        """Get the icon for a status based on language."""
        if self.language == "en":
            return self.STATUS_ICONS_EN.get(status, "?")
        return self.STATUS_ICONS.get(status, "?")

    def _render(self) -> Panel:
        """Render the complete dashboard with multi-panel layout."""
        # Build stats header
        pct_done = (self.completed / self.total_copies * 100) if self.total_copies > 0 else 0

        if self.language == "en":
            header_text = (
                f"[bold cyan]📊 {self.completed}/{self.total_copies} copies[/bold cyan] │ "
                f"[dim]⏱ ~{self._get_eta()} remaining[/dim] │ "
                f"[bold]{pct_done:.0f}%[/bold]"
            )
        else:
            header_text = (
                f"[bold cyan]📊 {self.completed}/{self.total_copies} copies[/bold cyan] │ "
                f"[dim]⏱ ~{self._get_eta()} restant[/dim] │ "
                f"[bold]{pct_done:.0f}%[/bold]"
            )

        # Agreement stats if we have comparison data
        stats_line = ""
        if self.stats.total_questions > 0:
            agree_pct = (self.stats.agreements / self.stats.total_questions * 100) if self.stats.total_questions > 0 else 0
            if self.language == "en":
                stats_line = (
                    f"[green]Agreements: {self.stats.agreements}/{self.stats.total_questions} ({agree_pct:.0f}%)[/green] │ "
                    f"[yellow]Disagreements: {self.stats.disagreements}[/yellow]"
                )
                if self.stats.avg_confidence > 0:
                    stats_line += f" │ [dim]Avg conf: {self.stats.avg_confidence:.0%}[/dim]"
            else:
                stats_line = (
                    f"[green]Accords: {self.stats.agreements}/{self.stats.total_questions} ({agree_pct:.0f}%)[/green] │ "
                    f"[yellow]Désaccords: {self.stats.disagreements}[/yellow]"
                )
                if self.stats.avg_confidence > 0:
                    stats_line += f" │ [dim]Conf moy: {self.stats.avg_confidence:.0%}[/dim]"

        # Current activity
        activity_line = ""
        if self.current_copy:
            status = self.copies.get(self.current_copy, CopyStatus())
            if self.language == "en":
                activity_line = f"[bold]📄 Copy {self.current_copy}/{self.total_copies}[/bold] │ {status.student_name}"
                if status.questions_total > 0:
                    activity_line += f" │ ⚙ Q{self.current_question}/{status.questions_total}..."
            else:
                activity_line = f"[bold]📄 Copie {self.current_copy}/{self.total_copies}[/bold] │ {status.student_name}"
                if status.questions_total > 0:
                    activity_line += f" │ ⚙ Q{self.current_question}/{status.questions_total}..."

        # Recent results
        recent_line = ""
        if self.stats.recent_results:
            recent_parts = []
            for r in self.stats.recent_results[-3:]:  # Show last 3
                pct = r['pct']
                if pct >= 70:
                    color = "green"
                elif pct >= 50:
                    color = "yellow"
                else:
                    color = "red"
                name_short = r['name'][:10] if r['name'] else "???"
                recent_parts.append(f"[{color}]✓ {name_short} {r['score']:.1f}/{r['max']:.0f}[/{color}]")

            if self.language == "en":
                recent_line = "[dim]Recent:[/dim] " + " │ ".join(recent_parts)
            else:
                recent_line = "[dim]Récents:[/dim] " + " │ ".join(recent_parts)

        # Combine all lines
        content_lines = [header_text]
        if stats_line:
            content_lines.append(stats_line)
        if activity_line:
            content_lines.append(activity_line)
        if recent_line:
            content_lines.append(recent_line)

        # Add copies table if there's activity
        content_lines.append("")  # Spacer
        content_lines.append(self._render_copies_table())

        return Panel(
            "\n".join(content_lines),
            title="[bold cyan]Grading Progress[/bold cyan]" if self.language == "en" else "[bold cyan]Progression[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        )

    def _render_copies_table(self) -> str:
        """Render the copies status as a compact string table."""
        lines = []

        # Show first 3, processing ones, and last 2
        max_show = 8
        indices_to_show = set()

        # First 3
        for i in range(1, min(4, self.total_copies + 1)):
            indices_to_show.add(i)

        # Processing ones
        for idx, status in self.copies.items():
            if status.status == "processing":
                indices_to_show.add(idx)

        # Last 2
        for i in range(max(1, self.total_copies - 1), self.total_copies + 1):
            indices_to_show.add(i)

        # Sort and limit
        sorted_indices = sorted(indices_to_show)[:max_show]

        for idx in sorted_indices:
            status = self.copies.get(idx, CopyStatus())
            icon = self._get_status_icon(status.status)

            # Format: "1/12 Jean M. ✓ 14.5/20"
            name = (status.student_name or "???")[:12]
            if status.score is not None and status.max_score is not None:
                pct = (status.score / status.max_score * 100) if status.max_score > 0 else 0
                if pct >= 50:
                    score_str = f"[green]{status.score:.1f}/{status.max_score:.0f}[/green]"
                else:
                    score_str = f"[red]{status.score:.1f}/{status.max_score:.0f}[/red]"
            else:
                score_str = "..."

            lines.append(f"  [dim]{idx:2d}[/dim] {name:12s} {icon:20s} {score_str}")

        if self.language == "en":
            if self.total_copies > max_show:
                lines.append(f"  [dim]... and {self.total_copies - max_show} more[/dim]")
        else:
            if self.total_copies > max_show:
                lines.append(f"  [dim]... et {self.total_copies - max_show} autres[/dim]")

        return "\n".join(lines)

    def get_summary(self) -> Dict[str, int]:
        """Get summary statistics."""
        return {
            "total": self.total_copies,
            "completed": self.completed,
            "errors": self.errors,
            "pending": self.total_copies - self.completed - self.errors -
                      sum(1 for s in self.copies.values() if s.status == "processing"),
            "processing": sum(1 for s in self.copies.values() if s.status == "processing")
        }

    def __enter__(self):
        """Start the live display."""
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=4,
            transient=False  # Keep the final table visible
        )
        self._live.__enter__()  # Start the live display
        return self  # Return self so methods like mark_processing() work

    def __exit__(self, *args):
        """Stop the live display."""
        if self._live:
            return self._live.__exit__(*args)


def create_live_progress_callback(
    display: LiveProgressDisplay,
    original_callback=None
):
    """
    Create a progress callback that updates the live display.

    Args:
        display: LiveProgressDisplay instance to update
        original_callback: Optional original callback to also call

    Returns:
        Async callback function
    """
    async def live_callback(event_type: str, data: dict):
        """Callback that updates the live display."""
        if event_type == 'copy_start':
            copy_idx = data.get('copy_index', 0)
            student = data.get('student_name', '???')
            questions = data.get('questions', [])
            display.mark_processing(
                copy_idx,
                student_name=student,
                questions_total=len(questions) if questions else 0
            )
            display.mark_current_activity(copy_idx, 0, len(questions) if questions else 0)

        elif event_type == 'question_start':
            copy_idx = data.get('copy_index', 0)
            q_num = data.get('question_num', 0)
            q_total = data.get('questions_total', 0)
            display.mark_current_activity(copy_idx, q_num, q_total)

        elif event_type == 'question_done':
            copy_idx = data.get('copy_index', 0)
            if copy_idx:
                display.mark_question_done(copy_idx)
            # Track agreement if available
            agreement = data.get('agreement')
            if agreement is not None:
                display.record_agreement(agreement)
            confidence = data.get('confidence')
            if confidence is not None:
                display.update_confidence(confidence)

        elif event_type == 'copy_done':
            copy_idx = data.get('copy_index', 0)
            score = data.get('total_score', 0) or 0
            max_score = data.get('max_score', 20) or 20
            display.mark_done(copy_idx, score, max_score)

        elif event_type == 'copy_error':
            copy_idx = data.get('copy_index', 0)
            error = data.get('error', 'Erreur')
            display.mark_error(copy_idx, error)

        elif event_type == 'llm_parallel_start':
            # Mark that we're in comparison mode
            copy_idx = data.get('copy_index', 0)
            display.mark_current_activity(copy_idx, 0, 0)

        elif event_type == 'analysis_complete':
            # Update agreement stats from analysis
            agreements = data.get('agreements', 0)
            disagreements = data.get('disagreements', 0)
            for _ in range(agreements):
                display.record_agreement(True)
            for _ in range(disagreements):
                display.record_agreement(False)

        # Call original callback if provided
        if original_callback:
            await original_callback(event_type, data)

    return live_callback
