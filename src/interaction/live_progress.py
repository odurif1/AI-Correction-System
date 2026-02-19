"""
Real-time progress display for parallel copy processing.

Provides a Rich Live dashboard that shows the status of all copies
being processed in parallel, with per-copy progress and scores.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from rich.live import Live
from rich.table import Table
from rich.console import Console


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
            self._live.update(self._render_table())

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
        self.update_copy(
            copy_index,
            status="done",
            score=score,
            max_score=max_score,
            questions_done=self.copies.get(copy_index, CopyStatus()).questions_total
        )

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

    def _render_table(self) -> Table:
        """Render the status table."""
        if self.language == "en":
            title = f"[bold cyan]Grading in Progress - {self.completed}/{self.total_copies} copies[/bold cyan]"
        else:
            title = f"[bold cyan]Correction en cours - {self.completed}/{self.total_copies} copies[/bold cyan]"

        table = Table(
            title=title,
            show_header=True,
            header_style="bold magenta",
            box=None,
            padding=(0, 1),
            expand=False
        )

        if self.language == "en":
            table.add_column("Copy", style="dim", width=6)
            table.add_column("Student", width=18)
            table.add_column("Status", width=14)
            table.add_column("Questions", justify="center", width=10)
            table.add_column("Score", justify="right", width=10)
        else:
            table.add_column("Copie", style="dim", width=6)
            table.add_column("Élève", width=18)
            table.add_column("Statut", width=14)
            table.add_column("Questions", justify="center", width=10)
            table.add_column("Note", justify="right", width=10)

        # Show all copies, but limit display if too many
        max_display = 15
        display_copies = list(self.copies.items())

        if len(display_copies) > max_display:
            # Show first few, last few, and current processing
            processing_indices = [
                idx for idx, status in display_copies
                if status.status == "processing"
            ]
            done_count = self.completed

            # Strategy: show first 5, processing ones, and last 3
            shown = set()

            # First 5
            for idx, _ in display_copies[:5]:
                shown.add(idx)

            # Processing ones
            for idx in processing_indices:
                shown.add(idx)

            # Last 3
            for idx, _ in display_copies[-3:]:
                shown.add(idx)

            # Fill remaining with next pending
            for idx, _ in display_copies:
                if len(shown) >= max_display:
                    break
                shown.add(idx)

            display_copies = [
                (idx, status) for idx, status in display_copies
                if idx in shown
            ]

        for idx, status in display_copies:
            icon = self._get_status_icon(status.status)

            # Questions display
            if status.questions_total > 0:
                questions = f"{status.questions_done}/{status.questions_total}"
            else:
                questions = "-"

            # Score display
            if status.score is not None and status.max_score is not None:
                score_text = f"{status.score:.1f}/{status.max_score:.0f}"
                # Color based on percentage
                pct = (status.score / status.max_score * 100) if status.max_score > 0 else 0
                if pct >= 50:
                    score = f"[green bold]{score_text}[/green bold]"
                else:
                    score = f"[red bold]{score_text}[/red bold]"
            elif status.error:
                error_display = status.error[:15] if len(status.error) > 15 else status.error
                score = f"[red]{error_display}[/red]"
            else:
                score = "..."

            # Truncate student name if too long
            name_display = status.student_name[:16] if status.student_name and len(status.student_name) > 16 else (status.student_name or "???")

            table.add_row(
                f"{idx}/{self.total_copies}",
                name_display,
                icon,
                questions,
                score
            )

        return table

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
            self._render_table(),
            console=self.console,
            refresh_per_second=4,
            transient=False  # Keep the final table visible
        )
        return self._live.__enter__()

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

        elif event_type == 'question_done':
            copy_idx = data.get('copy_index', 0)
            if copy_idx:
                display.mark_question_done(copy_idx)

        elif event_type == 'copy_done':
            copy_idx = data.get('copy_index', 0)
            score = data.get('total_score', 0) or 0
            max_score = data.get('max_score', 20) or 20
            display.mark_done(copy_idx, score, max_score)

        elif event_type == 'copy_error':
            copy_idx = data.get('copy_index', 0)
            error = data.get('error', 'Erreur')
            display.mark_error(copy_idx, error)

        # Call original callback if provided
        if original_callback:
            await original_callback(event_type, data)

    return live_callback
