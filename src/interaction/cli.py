"""
CLI Interface for AI correction system.

Handles user interactions with rich console prompts:
- Scale confirmation
- Doubts review
- Results display
- Live progress dashboard

UX Principles:
- Progressive disclosure: Show info when needed
- Visual hierarchy: Color and spacing guide attention
- Smart defaults: Minimize user input
- Forgiving design: Easy to understand and recover
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from time import time
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text
from rich.live import Live
from rich.layout import Layout

from core.models import CopyDocument, GradedCopy, TeacherDecision, ConfidenceLevel


# Shared console instance for CLI output
console = Console()


# Semantic color palette
class Colors:
    """Semantic colors for consistent UI."""
    SUCCESS = "bright_green"
    WARNING = "yellow"
    ERROR = "red"
    INFO = "blue"
    PRIMARY = "cyan"
    ACCENT = "magenta"
    DIM = "grey50"


class Decision:
    """Represents a user's decision on a doubtful case."""

    def __init__(
        self,
        question_id: str,
        copy_id: str,
        original_grade: float,
        new_grade: float,
        propagate: bool = False,
        similar_copy_ids: List[str] = None
    ):
        self.question_id = question_id
        self.copy_id = copy_id
        self.original_grade = original_grade
        self.new_grade = new_grade
        self.propagate = propagate
        self.similar_copy_ids = similar_copy_ids or []


class CLI:
    """
    Command-line interface for user interactions.

    Provides methods for:
    - Confirming detected grading scale
    - Asking for missing scale values
    - Reviewing doubtful cases
    - Displaying results
    """

    def __init__(self):
        self.console = Console()

    def show_startup(
        self,
        pdf_files: List[str] = None,
        mode: str = "single",
        grading_method: str = "individual",
        pages_per_copy: int = 2,
        language: str = "auto",
        llm1_name: str = None,
        llm2_name: str = None,
        display_language: str = "fr",
        session_id: str = None,
        options: Dict[str, Any] = None
    ):
        """
        Display compact startup screen with configuration.

        Args:
            pdf_files: List of PDF file paths
            mode: Grading mode ("single", "comparison", "conversation")
            grading_method: Grading method ("individual", "batch", "hybrid")
            pages_per_copy: Pages per student
            language: Content language ("auto", "fr", "en")
            llm1_name: First LLM name (for comparison mode)
            llm2_name: Second LLM name (for comparison mode)
            display_language: UI language
            session_id: Session identifier
            options: Dict of additional options (auto_confirm, batch_verify, annotate, etc.)
        """
        options = options or {}

        # Header
        header = Panel(
            "[bold cyan]🎓  AI Correction[/bold cyan]",
            border_style="cyan",
            padding=(0, 2)
        )
        self.console.print(header)

        # Configuration panel
        config_lines = []

        # Files
        file_count = len(pdf_files) if pdf_files else 0
        if display_language == "en":
            config_lines.append(f"[bold]📁 PDFs:[/bold]        {file_count} files")
            mode_display = {
                "single": "Single LLM",
                "comparison": "Dual LLM Comparison",
                "conversation": "Conversation"
            }.get(mode, mode)
            config_lines.append(f"[bold]🤖 Mode:[/bold]        {mode_display}")
            # Grading method
            method_display = {
                "individual": "Individual",
                "batch": "Batch",
                "hybrid": "Hybrid"
            }.get(grading_method, grading_method)
            config_lines.append(f"[bold]📋 Method:[/bold]      {method_display}")
            config_lines.append(f"[bold]🔧 Pages/student:[/bold] {pages_per_copy if pages_per_copy else 'Auto'}")
            lang_display = "Auto-detect" if language == "auto" else language.upper()
            config_lines.append(f"[bold]🌐 Language:[/bold]     {lang_display}")
            if session_id:
                config_lines.append(f"[bold]🔑 Session ID:[/bold]  {session_id}")
        else:
            config_lines.append(f"[bold]📁 PDFs:[/bold]        {file_count} fichiers")
            mode_display = {
                "single": "LLM Simple",
                "comparison": "Comparaison Double LLM",
                "conversation": "Conversation"
            }.get(mode, mode)
            config_lines.append(f"[bold]🤖 Mode:[/bold]        {mode_display}")
            # Grading method
            method_display = {
                "individual": "Individuel",
                "batch": "Lot",
                "hybrid": "Hybride"
            }.get(grading_method, grading_method)
            config_lines.append(f"[bold]📋 Méthode:[/bold]     {method_display}")
            pages_display = str(pages_per_copy) if pages_per_copy else "Auto"
            config_lines.append(f"[bold]🔧 Pages/élève:[/bold]  {pages_display}")
            lang_display = "Auto-détection" if language == "auto" else language.upper()
            config_lines.append(f"[bold]🌐 Langue:[/bold]      {lang_display}")
            if session_id:
                config_lines.append(f"[bold]🔑 Session ID:[/bold]  {session_id}")

        # LLM info for comparison mode
        if mode == "comparison" and llm1_name and llm2_name:
            if display_language == "en":
                config_lines.append(f"[dim]LLM1: {llm1_name}[/dim]")
                config_lines.append(f"[dim]LLM2: {llm2_name}[/dim]")
            else:
                config_lines.append(f"[dim]LLM1: {llm1_name}[/dim]")
                config_lines.append(f"[dim]LLM2: {llm2_name}[/dim]")

        # Options line
        opts_parts = []
        if options.get('auto_confirm'):
            opts_parts.append("--auto-confirm")
        if options.get('batch_verify'):
            opts_parts.append(f"--batch-verify={options['batch_verify']}")
        if options.get('annotate'):
            opts_parts.append("--annotate")
        if options.get('parallel', 1) > 1:
            opts_parts.append(f"--parallel={options['parallel']}")
        if options.get('second_reading'):
            opts_parts.append("--second-reading")
        if options.get('auto_detect_structure'):
            opts_parts.append("--auto-detect-structure")
        if options.get('skip_reading'):
            opts_parts.append("--skip-reading")

        if opts_parts:
            config_lines.append(f"[dim]Options: {' '.join(opts_parts)}[/dim]")

        config_panel = Panel(
            "\n".join(config_lines),
            title=f"[bold]{display_language == 'en' and 'Configuration' or 'Configuration'}[/bold]",
            border_style="dim",
            padding=(0, 1)
        )
        self.console.print(config_panel)

        # Processing message
        if display_language == "en":
            self.console.print(f"\n[bold cyan]▶[/bold cyan] Analyzing copies... [dim](this may take a moment)[/dim]\n")
        else:
            self.console.print(f"\n[bold cyan]▶[/bold cyan] Analyse des copies... [dim](cela peut prendre un moment)[/dim]\n")

    def show_info(self, message: str):
        """Display an info message."""
        self.console.print(f"[cyan]{message}[/cyan]")

    def show_success(self, message: str):
        """Display a success message."""
        self.console.print(f"[green]✓ {message}[/green]")

    def show_warning(self, message: str):
        """Display a warning message."""
        self.console.print(f"[yellow]⚠ {message}[/yellow]")

    def show_error(self, message: str, solution: str = None, language: str = 'fr'):
        """
        Display an error message with optional solution guidance.

        Args:
            message: Error message
            solution: Optional solution or guidance
            language: Display language
        """
        if solution:
            if language == 'fr':
                panel = Panel(
                    f"[red]✗ {message}[/red]\n\n"
                    f"[bold]Solution:[/bold] {solution}",
                    title="[bold red]Erreur[/bold red]",
                    border_style="red",
                    padding=(0, 1)
                )
            else:
                panel = Panel(
                    f"[red]✗ {message}[/red]\n\n"
                    f"[bold]Solution:[/bold] {solution}",
                    title="[bold red]Error[/bold red]",
                    border_style="red",
                    padding=(0, 1)
                )
            self.console.print(panel)
        else:
            self.console.print(f"[red]✗ {message}[/red]")

    def confirm_scale(
        self,
        detected_scale: Dict[str, float],
        questions: Dict[str, str] = None,
        language: str = 'fr'
    ) -> Dict[str, float]:
        """
        Display detected scale and ask for confirmation.

        Args:
            detected_scale: Dict of {question_id: max_points}
            questions: Optional dict of {question_id: question_text}
            language: Language for prompts ('fr' or 'en')

        Returns:
            Confirmed or modified scale
        """
        if language == 'fr':
            self.console.print("\n[bold]Barème détecté:[/bold]")
        else:
            self.console.print("\n[bold]Detected grading scale:[/bold]")

        # Display scale as a table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Question", style="cyan")
        if questions:
            table.add_column("Description", style="dim")
        table.add_column("Points", style="green", justify="right")

        for q_id, points in detected_scale.items():
            if questions and q_id in questions:
                table.add_row(q_id, questions[q_id][:50] + "...", str(points))
            else:
                table.add_row(q_id, str(points))

        self.console.print(table)

        total = sum(detected_scale.values())
        if language == 'fr':
            self.console.print(f"\n[bold]Total: {total} points[/bold]")
            if Confirm.ask("\nCe barème est-il correct ?"):
                return detected_scale
            return self._edit_scale(detected_scale, questions, language)
        else:
            self.console.print(f"\n[bold]Total: {total} points[/bold]")
            if Confirm.ask("\nIs this scale correct?"):
                return detected_scale
            return self._edit_scale(detected_scale, questions, language)

    def _edit_scale(
        self,
        current_scale: Dict[str, float],
        questions: Dict[str, str] = None,
        language: str = 'fr'
    ) -> Dict[str, float]:
        """
        Allow user to edit the scale.

        Args:
            current_scale: Current scale values
            questions: Optional question descriptions
            language: Language for prompts

        Returns:
            Modified scale
        """
        scale = dict(current_scale)

        if language == 'fr':
            self.console.print("\n[bold yellow]Modification du barème[/bold yellow]")
            self.console.print("[dim]Appuyez sur Entrée pour garder la valeur actuelle[/dim]")
        else:
            self.console.print("\n[bold yellow]Edit grading scale[/bold yellow]")
            self.console.print("[dim]Press Enter to keep current value[/dim]")

        for q_id in list(scale.keys()):
            current = scale[q_id]
            if questions and q_id in questions:
                desc = f" ({questions[q_id][:30]}...)"
            else:
                desc = ""

            if language == 'fr':
                new_value = Prompt.ask(
                    f"Points pour {q_id}{desc}",
                    default=str(current)
                )
            else:
                new_value = Prompt.ask(
                    f"Points for {q_id}{desc}",
                    default=str(current)
                )

            try:
                scale[q_id] = float(new_value.replace(',', '.'))
            except ValueError:
                self.show_warning(f"Valeur invalide, garde {current}")

        return scale

    def ask_for_single_scale(
        self,
        question_id: str,
        answer_summary: str = None,
        language: str = 'fr'
    ) -> float:
        """
        Ask user for scale of a single question.

        Args:
            question_id: The question identifier
            answer_summary: Optional summary of student's answer
            language: Language for prompts

        Returns:
            Scale value for this question
        """
        if language == 'fr':
            self.console.print(f"\n[bold yellow]Barème manquant pour {question_id}[/bold yellow]")
            if answer_summary:
                self.console.print(f"[dim]Résumé de la réponse: {answer_summary[:100]}...[/dim]")

            while True:
                value = Prompt.ask(f"Combien de points pour {question_id} ?")
                try:
                    scale = float(value.replace(',', '.').replace('pts', '').replace('points', '').strip())
                    if scale > 0:
                        return scale
                    self.show_error("Le barème doit être positif")
                except ValueError:
                    self.show_error("Veuillez entrer un nombre valide (ex: 2, 2.5, 3)")
        else:
            self.console.print(f"\n[bold yellow]Missing scale for {question_id}[/bold yellow]")
            if answer_summary:
                self.console.print(f"[dim]Answer summary: {answer_summary[:100]}...[/dim]")

            while True:
                value = Prompt.ask(f"How many points for {question_id}?")
                try:
                    scale = float(value.replace(',', '.').replace('pts', '').replace('points', '').strip())
                    if scale > 0:
                        return scale
                    self.show_error("Scale must be positive")
                except ValueError:
                    self.show_error("Please enter a valid number (e.g., 2, 2.5, 3)")

    def review_doubts(
        self,
        doubts: List[Tuple[CopyDocument, GradedCopy, str, float]],
        language: str = 'fr'
    ) -> List[Decision]:
        """
        Present doubtful cases with bulk actions and navigation.

        Args:
            doubts: List of (copy, graded_copy, question_id, confidence) tuples
            language: Language for prompts

        Returns:
            List of Decision objects
        """
        decisions = []

        if not doubts:
            return decisions

        if language == 'fr':
            self.console.print(f"\n[bold yellow]⚠ {len(doubts)} cas(s) douteux à réviser[/bold yellow]")
            self.console.print("[dim]Raccourcis: a=accepter, m=modifier, s=passer, ent=tout accepter, q=quitter[/dim]")
        else:
            self.console.print(f"\n[bold yellow]⚠ {len(doubts)} doubtful case(s) to review[/bold yellow]")
            self.console.print("[dim]Shortcuts: a=accept, m=modify, s=skip, ent=accept all, q=quit[/dim]")

        for i, (copy, graded, q_id, confidence) in enumerate(doubts, 1):
            decision = self._review_single_doubt(copy, graded, q_id, confidence, i, len(doubts), language)

            # Check for bulk actions
            if decision and decision.new_grade == "ACCEPT_ALL":
                # Accept all remaining
                for j in range(i, len(doubts)):
                    remaining_copy, remaining_graded, remaining_q, remaining_conf = doubts[j]
                    decisions.append(Decision(
                        question_id=remaining_q,
                        copy_id=remaining_copy.id,
                        original_grade=remaining_graded.grades.get(remaining_q, 0),
                        new_grade=remaining_graded.grades.get(remaining_q, 0),
                        propagate=True
                    ))
                break

            if decision:
                decisions.append(decision)

        return decisions

    def _review_single_doubt(
        self,
        copy: CopyDocument,
        graded: GradedCopy,
        question_id: str,
        confidence: float,
        current: int,
        total: int,
        language: str
    ) -> Optional[Decision]:
        """
        Review a single doubtful case with shortcuts.

        Args:
            copy: Student's copy
            graded: Graded copy with current grade
            question_id: Question identifier
            confidence: AI confidence level
            current: Current doubt number
            total: Total doubts
            language: Language for prompts

        Returns:
            Decision or None if skipped
        """
        current_grade = graded.grades.get(question_id, 0)
        reasoning = graded.reasoning.get(question_id, "")

        # Compact panel with doubt details
        conf_display = (confidence or 0.5) * 100
        student = copy.student_name or "???"

        if language == 'fr':
            panel_content = (
                f"[bold]{current}/{total}[/bold] │ {student} │ {question_id}\n"
                f"[bold]Note:[/bold] {current_grade}  │  [bold]Confiance:[/bold] {conf_display:.0f}%\n"
                f"[dim]{reasoning[:80]}{'...' if len(reasoning) > 80 else ''}[/dim]"
            )
            prompt_text = "[a]ccepter [m]odifier [s]kip [v]oir [ent] tout accepter"
        else:
            panel_content = (
                f"[bold]{current}/{total}[/bold] │ {student} │ {question_id}\n"
                f"[bold]Grade:[/bold] {current_grade}  │  [bold]Confidence:[/bold] {conf_display:.0f}%\n"
                f"[dim]{reasoning[:80]}{'...' if len(reasoning) > 80 else ''}[/dim]"
            )
            prompt_text = "[a]ccept [m]odify [s]kip [v]iew [ent] accept all"

        self.console.print(Panel(panel_content, border_style="yellow", padding=(0, 1)))

        # Single letter shortcuts
        choice = Prompt.ask(f"\n{prompt_text}", default="a")
        self.console.print("")

        choice_lower = choice.lower().strip()

        if choice_lower in ["a", "accept", "1"]:
            return Decision(
                question_id=question_id,
                copy_id=copy.id,
                original_grade=current_grade,
                new_grade=current_grade,
                propagate=True
            )
        elif choice_lower in ["m", "modify", "2"]:
            new_grade = self._ask_new_grade(current_grade, language)
            return Decision(
                question_id=question_id,
                copy_id=copy.id,
                original_grade=current_grade,
                new_grade=new_grade,
                propagate=True
            )
        elif choice_lower in ["s", "skip", "3"]:
            return None
        elif choice_lower in ["v", "view", "4"]:
            self._show_question_context(copy, question_id, reasoning, language)
            # Re-ask for decision
            return self._review_single_doubt(copy, graded, question_id, confidence, current, total, language)
        elif choice_lower in ["", "ent", "all", "q", "quit"]:
            # Bulk accept or quit
            bulk = Decision(
                question_id=question_id,
                copy_id=copy.id,
                original_grade=current_grade,
                new_grade=current_grade,
                propagate=True
            )
            # Special marker for bulk accept
            if choice_lower in ["", "ent", "all"]:
                bulk.new_grade = "ACCEPT_ALL"  # Special marker
            return bulk

        # Default: accept
        return Decision(
            question_id=question_id,
            copy_id=copy.id,
            original_grade=current_grade,
            new_grade=current_grade,
            propagate=True
        )

    def _show_question_context(
        self,
        copy: CopyDocument,
        question_id: str,
        reasoning: str,
        language: str
    ):
        """Show question context to help human grader."""
        # Get the detected answer from content summary
        answer = copy.content_summary.get(question_id, "Non détecté")
        points = copy.content_summary.get(f"{question_id}_points", "?")
        confidence = copy.content_summary.get(f"{question_id}_confidence", "?")

        if language == 'fr':
            self.console.print(f"\n[bold cyan]── Contexte pour {question_id} ──[/bold cyan]")
            self.console.print(f"[bold]Élève:[/bold] {copy.student_name or 'Non identifié'}")
            self.console.print(f"[bold]Réponse détectée:[/bold] {answer}")
            self.console.print(f"[bold]Barème:[/bold] {points} points")
            self.console.print(f"[bold]Confiance barème:[/bold] {confidence}")
            self.console.print(f"[bold]Justification IA:[/bold] {reasoning}")
            self.console.print("")
        else:
            self.console.print(f"\n[bold cyan]── Context for {question_id} ──[/bold cyan]")
            self.console.print(f"[bold]Student:[/bold] {copy.student_name or 'Not identified'}")
            self.console.print(f"[bold]Detected answer:[/bold] {answer}")
            self.console.print(f"[bold]Scale:[/bold] {points} points")
            self.console.print(f"[bold]Scale confidence:[/bold] {confidence}")
            self.console.print(f"[bold]AI reasoning:[/bold] {reasoning}")
            self.console.print("")

    def _ask_new_grade(self, current: float, language: str) -> float:
        """Ask user for a new grade."""
        if language == 'fr':
            value = Prompt.ask(f"Nouvelle note (actuelle: {current})")
        else:
            value = Prompt.ask(f"New grade (current: {current})")

        try:
            return float(value.replace(',', '.'))
        except ValueError:
            self.show_warning("Valeur invalide, garde la note actuelle")
            return current

    def _ask_propagate(self, language: str) -> bool:
        """Ask if decision should be propagated to similar answers."""
        if language == 'fr':
            return Confirm.ask("Appliquer cette décision aux réponses similaires ?", default=True)
        else:
            return Confirm.ask("Apply this decision to similar answers?", default=True)

    def show_summary(
        self,
        copies_count: int,
        graded_count: int,
        scores: List[float],
        duration: float = None,
        mode: str = "single",
        exports: Dict[str, str] = None,
        top_performers: List[Dict] = None,
        language: str = 'fr'
    ):
        """
        Display the final summary with rich dashboard.

        Args:
            copies_count: Total copies processed
            graded_count: Number of graded copies
            scores: List of all scores
            duration: Session duration in seconds
            mode: Grading mode used
            exports: Dict of exported files {format: path}
            top_performers: List of top performers [{name, score, max}]
            language: Language for display
        """
        # Build summary content
        lines = []

        # Duration info only
        duration_str = ""
        if duration:
            if duration < 60:
                duration_str = f"{int(duration)}s"
            elif duration < 3600:
                duration_str = f"{int(duration / 60)}m {int(duration % 60)}s"
            else:
                duration_str = f"{int(duration / 3600)}h {int((duration % 3600) / 60)}m"

        # Calculate average score
        avg_score = sum(scores) / len(scores) if scores else 0

        if language == 'fr':
            mode_display = {"single": "LLM Simple", "comparison": "Double LLM", "conversation": "Conversation"}.get(mode, mode)
            lines.append(f"[bold]Durée:[/bold] {duration_str}  │  [bold]Mode:[/bold] {mode_display}  │  [bold]Copies:[/bold] {copies_count}  │  [bold]Moyenne:[/bold] {avg_score:.1f}")
        else:
            mode_display = {"single": "Single LLM", "comparison": "Dual LLM", "conversation": "Conversation"}.get(mode, mode)
            lines.append(f"[bold]Duration:[/bold] {duration_str}  │  [bold]Mode:[/bold] {mode_display}  │  [bold]Copies:[/bold] {copies_count}  │  [bold]Average:[/bold] {avg_score:.1f}")

        lines.append("")

        # Score distribution (main visual)
        if scores:
            # Determine actual max score from first graded copy (if available)
            # Otherwise default to 20
            actual_max = 20
            if top_performers and len(top_performers) > 0:
                actual_max = top_performers[0].get('max', 20)

            # Score distribution
            if language == 'fr':
                lines.append("[bold cyan]Distribution des notes:[/bold cyan]")
            else:
                lines.append("[bold cyan]Score Distribution:[/bold cyan]")

            dist_str = self._render_distribution_bar(scores, actual_max=actual_max)
            lines.append(dist_str)
            lines.append("")

            # Top performers
            if top_performers and len(top_performers) > 0:
                if language == 'fr':
                    lines.append("[bold cyan]Top Performers:[/bold cyan]")
                else:
                    lines.append("[bold cyan]Top Performers:[/bold cyan]")

                medals = ["🥇", "🥈", "🥉"]
                for i, performer in enumerate(top_performers[:3]):
                    medal = medals[i] if i < 3 else "  "
                    name = performer.get('name') or '???'
                    score = performer.get('score', 0)
                    max_score_p = performer.get('max', 20)
                    pct = (score / max_score_p * 100) if max_score_p > 0 else 0
                    lines.append(f"  {medal} {name:20s}  [green]{score:.1f}/{max_score_p:.0f}[/green] ({pct:.0f}%)")
                lines.append("")

        # Exports
        if exports:
            export_strs = []
            for fmt, path in exports.items():
                export_strs.append(f"[green]✓[/green] {fmt}")
            if language == 'fr':
                lines.append(f"[bold]Exports:[/bold] {' '.join(export_strs)}")
            else:
                lines.append(f"[bold]Exports:[/bold] {' '.join(export_strs)}")

            # Output directory
            first_path = list(exports.values())[0] if exports else ""
            if first_path:
                import os
                output_dir = os.path.dirname(first_path)
                lines.append(f"[dim]📁 {output_dir}[/dim]")

        panel = Panel(
            "\n".join(lines),
            title=f"[bold green]{'✓ Correction terminée' if language == 'fr' else '✓ Grading Complete'}[/bold green]",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print("\n")
        self.console.print(panel)

    def _render_distribution_bar(self, scores: List[float], max_bar: int = 10, actual_max: float = 20) -> str:
        """Render a distribution bar chart with dynamic buckets."""
        if not scores:
            return ""

        # Create 4 dynamic buckets based on actual_max
        bucket_size = actual_max / 4
        buckets = {}
        for i in range(4):
            low = i * bucket_size
            high = (i + 1) * bucket_size
            key = f"{low:.0f}-{high:.0f}"
            buckets[key] = 0

        for score in scores:
            bucket_idx = min(int(score / bucket_size), 3) if bucket_size > 0 else 0
            low = bucket_idx * bucket_size
            high = (bucket_idx + 1) * bucket_size
            key = f"{low:.0f}-{high:.0f}"
            buckets[key] = buckets.get(key, 0) + 1

        max_count = max(buckets.values()) if buckets else 1
        lines = []

        for i, (bucket, count) in enumerate(buckets.items()):
            if max_count > 0:
                bar_len = int((count / max_count) * max_bar)
            else:
                bar_len = 0
            bar = "█" * bar_len + "░" * (max_bar - bar_len)

            # Color: first bucket red, second yellow, rest green
            if i == 0:
                color = "red"
            elif i == 1:
                color = "yellow"
            else:
                color = "green"

            lines.append(f"  {bucket:6s} [{color}]{bar}[/{color}] ({count})")

        return "\n".join(lines)

    def show_copy_analysis(
        self,
        copy_id: str,
        student_name: str,
        questions: Dict[str, str],
        scale: Dict[str, float],
        low_confidence: List[str] = None,
        unknown_scale: List[str] = None,
        language: str = 'fr'
    ):
        """
        Display analysis results for a single copy.

        Args:
            copy_id: Copy identifier
            student_name: Student name (or None if not found)
            questions: Dict of {question_id: question_summary}
            scale: Dict of {question_id: max_points}
            low_confidence: List of questions with low confidence
            unknown_scale: List of questions with unknown scale
            language: Language for display
        """
        low_confidence = low_confidence or []
        unknown_scale = unknown_scale or []

        if language == 'fr':
            header = f"📄 Copie {copy_id[:8]}"
        else:
            header = f"📄 Copy {copy_id[:8]}"

        # Add student name or warning
        if student_name:
            header += f" - {student_name}"
        else:
            header += " - [yellow]Nom non détecté[/yellow]"

        self.console.print(f"\n[bold cyan]{header}[/bold cyan]")
        self.console.print("─" * 50)

        table = Table(show_header=True, header_style="bold")
        table.add_column("Question", style="cyan", width=8)
        table.add_column("Points", style="green", justify="right", width=8)
        table.add_column("Statut", style="yellow", width=15)

        for q_id in sorted(questions.keys()):
            if q_id in unknown_scale:
                pts = "?"
                status = "🔴 non détecté"
            elif q_id in low_confidence:
                pts = scale.get(q_id, "?")
                status = "🟡 confiance basse"
            else:
                pts = scale.get(q_id, "?")
                status = "🟢 ok"
            table.add_row(q_id, str(pts), status)

        self.console.print(table)

        # Summary
        total = sum(scale.values()) if scale else 0
        self.console.print(f"[bold]Total détecté:[/bold] {total} points")

        # Warnings
        if unknown_scale:
            self.console.print(f"[red]⚠ Barème non détecté pour: {', '.join(unknown_scale)}[/red]")
        if low_confidence and not unknown_scale:
            self.console.print(f"[yellow]⚠ Confiance basse pour: {', '.join(low_confidence)}[/yellow]")

    def show_copy_grade(
        self,
        copy_number: int,
        student_name: str,
        grades: Dict[str, float],
        feedback: Dict[str, str],
        total_score: float,
        max_score: float,
        confidence: float,
        language: str = 'fr',
        llm_comparison: Dict = None
    ):
        """
        Display grading results for a single copy - simple and clean.

        Args:
            copy_number: Sequential copy number (1, 2, 3...)
            student_name: Student name (or None if not found)
            grades: Dict of {question_id: points_earned}
            feedback: Dict of {question_id: student_feedback}
            total_score: Total points earned
            max_score: Maximum possible points
            confidence: Overall confidence
            language: Language for display
            llm_comparison: Optional comparison data from dual-LLM mode
        """
        percentage = (total_score / max_score * 100) if max_score > 0 else 0

        # Determine color based on score
        if percentage >= 70:
            score_color = "green"
        elif percentage >= 50:
            score_color = "yellow"
        else:
            score_color = "red"

        # Build header
        if student_name:
            header = f"📝 {student_name}"
        else:
            if language == 'fr':
                header = f"📝 Copie {copy_number} — [yellow]Nom non détecté[/yellow]"
            else:
                header = f"📝 Copy {copy_number} — [yellow]Name not detected[/yellow]"

        self.console.print(f"\n[bold]{header}[/bold]")

        # Simple grades table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Q", style="cyan", width=4)
        table.add_column("Note", justify="right", width=8)

        for q_id in sorted(grades.keys()):
            grade = grades[q_id]
            # Find max points for this question from comparison data or estimate
            if llm_comparison and q_id in llm_comparison:
                comp = llm_comparison[q_id]
                max_q = comp.get('llm1', {}).get('grade', 0)
                if max_q is None:
                    max_q = 1
                # Try to get actual max from the data
                g1 = comp.get('llm1', {}).get('grade', 0) or 0
                g2 = comp.get('llm2', {}).get('grade', 0) or 0
                # Use higher grade as reference to guess max (imperfect but works)
            else:
                pass

            # Color code individual grades
            if grade >= 1.5:
                grade_str = f"[green]{grade:.1f}[/green]"
            elif grade >= 0.5:
                grade_str = f"[yellow]{grade:.1f}[/yellow]"
            else:
                grade_str = f"[red]{grade:.1f}[/red]"

            table.add_row(q_id, grade_str)

        self.console.print(table)

        # Final score line
        self.console.print(
            f"  [bold]{language == 'fr' and 'Total' or 'Total'}:[/bold] [{score_color}]{total_score:.1f}/{max_score} ({percentage:.0f}%)[/{score_color}]"
        )

    def show_all_copies_summary(
        self,
        copies_data: List[Dict],
        language: str = 'fr'
    ):
        """
        Display a compact summary of all copies.

        Args:
            copies_data: List of dicts with copy_number, student_name, total, max, confidence
            language: Language for display
        """
        if language == 'fr':
            self.console.print("\n" + "─" * 60)
            self.console.print("[bold]📋 Résumé[/bold]")
        else:
            self.console.print("\n" + "─" * 60)
            self.console.print("[bold]📋 Summary[/bold]")

        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("Élève", style="cyan", width=22)
        table.add_column("Note", justify="right", width=10)
        table.add_column("Appréciation", width=40)

        for data in copies_data:
            student_name = data.get('student_name')
            total = data.get('total', 0)
            max_score = data.get('max', 1)
            feedback = data.get('feedback', '')

            # Display name or placeholder
            if student_name:
                display_name = student_name[:20]
            else:
                display_name = "[dim](nom ?)[/dim]" if language == 'fr' else "[dim](name?)[/dim]"

            pct = (total / max_score * 100) if max_score > 0 else 0

            # Color based on percentage
            if pct >= 70:
                note_str = f"[green]{total:.1f}/{max_score}[/green]"
            elif pct >= 50:
                note_str = f"[yellow]{total:.1f}/{max_score}[/yellow]"
            else:
                note_str = f"[red]{total:.1f}/{max_score}[/red]"

            # Truncate feedback if too long
            if feedback:
                display_feedback = feedback[:50] + "..." if len(feedback) > 50 else feedback
            else:
                display_feedback = "[dim]—[/dim]"

            table.add_row(display_name, note_str, display_feedback)

        self.console.print(table)

    def show_export_results(self, exports: Dict[str, str], language: str = 'fr'):
        """Display export results."""
        if language == 'fr':
            self.console.print("\n[bold]Fichiers exportés:[/bold]")
        else:
            self.console.print("\n[bold]Exported files:[/bold]")

        for fmt, path in exports.items():
            self.console.print(f"  [{fmt.upper()}] {path}")

    def create_progress(self) -> Progress:
        """Create a progress bar for operations."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        )

    def show_disagreement(
        self,
        question_id: str,
        question_text: str,
        llm1_name: str,
        llm1_result: Dict,
        llm2_name: str,
        llm2_result: Dict,
        language: str = 'fr'
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Display a disagreement between two LLMs side-by-side and ask user to choose.

        Args:
            question_id: Question identifier
            question_text: The question text
            llm1_name: Name of first LLM
            llm1_result: Result from first LLM
            llm2_name: Name of second LLM
            llm2_result: Result from second LLM
            language: Display language

        Returns:
            Tuple of (chosen_grade, feedback_source) where feedback_source is:
            - "llm1": use LLM1's feedback
            - "llm2": use LLM2's feedback
            - "merge": merge both feedbacks (for average/manual cases)
            - None: if user skipped
        """
        reasoning1 = llm1_result.get('reasoning', '')[:100]
        reasoning2 = llm2_result.get('reasoning', '')[:100]
        grade1 = llm1_result.get('grade', 0) or 0
        grade2 = llm2_result.get('grade', 0) or 0
        conf1 = (llm1_result.get('confidence') or 0) * 100
        conf2 = (llm2_result.get('confidence') or 0) * 100

        # Create side-by-side panels
        if language == 'fr':
            panel1_content = (
                f"[bold]Note:[/bold] {grade1:.1f}\n"
                f"[bold]Confiance:[/bold] {conf1:.0f}%\n"
                f"[dim]{reasoning1}{'...' if len(llm1_result.get('reasoning', '')) > 100 else ''}[/dim]"
            )
            panel2_content = (
                f"[bold]Note:[/bold] {grade2:.1f}\n"
                f"[bold]Confiance:[/bold] {conf2:.0f}%\n"
                f"[dim]{reasoning2}{'...' if len(llm2_result.get('reasoning', '')) > 100 else ''}[/dim]"
            )

            panel1 = Panel(panel1_content, title=f"[cyan]{llm1_name}[/cyan]", border_style="cyan", padding=(0, 1))
            panel2 = Panel(panel2_content, title=f"[magenta]{llm2_name}[/magenta]", border_style="magenta", padding=(0, 1))

            # Header with question
            self.console.print(f"\n[bold yellow]⚠ Désaccord détecté[/bold yellow] - {question_id}")
            self.console.print(Columns([panel1, panel2]))

            # Options
            avg_grade = (grade1 + grade2) / 2
            self.console.print(
                f"\n[bold][1][/bold] {llm1_name} ({grade1:.1f})  "
                f"[bold][2][/bold] {llm2_name} ({grade2:.1f})  "
                f"[bold][3][/bold] Moyenne ({avg_grade:.2f})  "
                f"[bold][4][/bold] Personnalisé"
            )
            choice = Prompt.ask(f"\n[bold]Choix[/bold] [dim](Entrée = moyenne)[/dim]", default="3")
            self.console.print("")
        else:
            panel1_content = (
                f"[bold]Grade:[/bold] {grade1:.1f}\n"
                f"[bold]Confidence:[/bold] {conf1:.0f}%\n"
                f"[dim]{reasoning1}{'...' if len(llm1_result.get('reasoning', '')) > 100 else ''}[/dim]"
            )
            panel2_content = (
                f"[bold]Grade:[/bold] {grade2:.1f}\n"
                f"[bold]Confidence:[/bold] {conf2:.0f}%\n"
                f"[dim]{reasoning2}{'...' if len(llm2_result.get('reasoning', '')) > 100 else ''}[/dim]"
            )

            panel1 = Panel(panel1_content, title=f"[cyan]{llm1_name}[/cyan]", border_style="cyan", padding=(0, 1))
            panel2 = Panel(panel2_content, title=f"[magenta]{llm2_name}[/magenta]", border_style="magenta", padding=(0, 1))

            # Header with question
            self.console.print(f"\n[bold yellow]⚠ Disagreement detected[/bold yellow] - {question_id}")
            self.console.print(Columns([panel1, panel2]))

            # Options
            avg_grade = (grade1 + grade2) / 2
            self.console.print(
                f"\n[bold][1][/bold] {llm1_name} ({grade1:.1f})  "
                f"[bold][2][/bold] {llm2_name} ({grade2:.1f})  "
                f"[bold][3][/bold] Average ({avg_grade:.2f})  "
                f"[bold][4][/bold] Custom"
            )
            choice = Prompt.ask(f"\n[bold]Choice[/bold] [dim](Enter = average)[/dim]", default="3")
            self.console.print("")

        # Process choice
        if choice == "1":
            return grade1, "llm1"
        elif choice == "2":
            return grade2, "llm2"
        elif choice == "3" or choice == "":
            return avg_grade, "merge"
        elif choice == "4":
            # Ask for custom grade
            try:
                custom = Prompt.ask(f"[bold]Custom grade[/bold]" if language == 'en' else "[bold]Note personnalisée[/bold]")
                return float(custom.replace(',', '.')), "merge"
            except ValueError:
                return avg_grade, "merge"
        else:
            # Try to parse as number directly
            try:
                return float(choice.replace(',', '.')), "merge"
            except ValueError:
                return avg_grade, "merge"

    def show_comparison_summary(self, disagreements: List[Dict], language: str = 'fr'):
        """Show summary of LLM comparison results."""
        if language == 'fr':
            self.console.print("\n[bold]Résumé de la comparaison LLM:[/bold]")
            self.console.print(f"  Désaccords: {len(disagreements)}")
        else:
            self.console.print("\n[bold]LLM Comparison Summary:[/bold]")
            self.console.print(f"  Disagreements: {len(disagreements)}")

    def show_name_disagreement(
        self,
        llm1_result: Dict,
        llm2_result: Dict,
        language: str = 'fr'
    ) -> str:
        """
        Display a name disagreement between two LLMs and ask user to choose.

        Args:
            llm1_result: Dict with 'provider', 'name', 'confidence', 'reasoning'
            llm2_result: Dict with 'provider', 'name', 'confidence', 'reasoning'
            language: Display language

        Returns:
            Chosen name
        """
        from rich.prompt import Prompt
        from rich.panel import Panel

        name1 = llm1_result.get('name') or "Inconnu"
        name2 = llm2_result.get('name') or "Inconnu"
        llm1_name = llm1_result.get('provider', 'LLM1')
        llm2_name = llm2_result.get('provider', 'LLM2')

        if language == 'fr':
            self.console.print(f"\n[bold yellow]⚠ DÉSACCORD NOM ÉLÈVE[/bold yellow]")

            # LLM1 results
            self.console.print(f"\n[cyan]{llm1_name.upper()}:[/cyan]")
            self.console.print(f"  Nom: [bold]{name1}[/bold]")
            self.console.print(f"  Confiance: {(llm1_result.get('confidence') or 0):.0%}")
            self.console.print(f"  [dim]Raisonnement: {llm1_result.get('reasoning', '')}[/dim]")

            # LLM2 results
            self.console.print(f"\n[magenta]{llm2_name.upper()}:[/magenta]")
            self.console.print(f"  Nom: [bold]{name2}[/bold]")
            self.console.print(f"  Confiance: {(llm2_result.get('confidence') or 0):.0%}")
            self.console.print(f"  [dim]Raisonnement: {llm2_result.get('reasoning', '')}[/dim]")

            self.console.print(f"\n[bold]Options:[/bold]")
            self.console.print(f"  1 = {name1} (selon {llm1_name})")
            self.console.print(f"  2 = {name2} (selon {llm2_name})")
            self.console.print(f"  Ou entrez directement le nom correct")
            self.console.print(f"  Entrée = Nom avec la plus haute confiance")

            choice = Prompt.ask("Choix du nom", default="")
            self.console.print("")  # Add newline after input

        else:
            self.console.print(f"\n[bold yellow]⚠ STUDENT NAME DISAGREEMENT[/bold yellow]")

            # LLM1 results
            self.console.print(f"\n[cyan]{llm1_name.upper()}:[/cyan]")
            self.console.print(f"  Name: [bold]{name1}[/bold]")
            self.console.print(f"  Confidence: {(llm1_result.get('confidence') or 0):.0%}")
            self.console.print(f"  [dim]Reasoning: {llm1_result.get('reasoning', '')}[/dim]")

            # LLM2 results
            self.console.print(f"\n[magenta]{llm2_name.upper()}:[/magenta]")
            self.console.print(f"  Name: [bold]{name2}[/bold]")
            self.console.print(f"  Confidence: {(llm2_result.get('confidence') or 0):.0%}")
            self.console.print(f"  [dim]Reasoning: {llm2_result.get('reasoning', '')}[/dim]")

            self.console.print(f"\n[bold]Options:[/bold]")
            self.console.print(f"  1 = {name1} (per {llm1_name})")
            self.console.print(f"  2 = {name2} (per {llm2_name})")
            self.console.print(f"  Or enter the correct name directly")
            self.console.print(f"  Enter = Name with highest confidence")

            choice = Prompt.ask("Name choice", default="")
            self.console.print("")  # Add newline after input

        # Process choice
        if not choice or choice.strip() == "":
            # Use highest confidence
            if (llm1_result.get('confidence') or 0) >= (llm2_result.get('confidence') or 0):
                return name1
            else:
                return name2
        elif choice.strip() == "1":
            return name1
        elif choice.strip() == "2":
            return name2
        else:
            # Direct name input
            return choice.strip()

    def show_reading_disagreement(
        self,
        llm1_result: Dict,
        llm2_result: Dict,
        question_text: str,
        image_path,
        language: str = 'fr'
    ) -> str:
        """
        Display a reading disagreement between two LLMs and ask user to choose.

        Args:
            llm1_result: Dict with 'reading', 'confidence', 'contradictions'
            llm2_result: Dict with 'reading', 'confidence', 'contradictions'
            question_text: The question being read
            image_path: Path to the image (for reference)
            language: Display language

        Returns:
            Chosen reading string
        """
        from rich.prompt import Prompt

        reading1 = llm1_result.get('reading', '')
        reading2 = llm2_result.get('reading', '')
        conf1 = llm1_result.get('confidence') or 0
        conf2 = llm2_result.get('confidence') or 0
        llm1_name = llm1_result.get('provider', 'LLM1')
        llm2_name = llm2_result.get('provider', 'LLM2')

        # Truncate long readings for display
        reading1_display = reading1[:200] + "..." if len(reading1) > 200 else reading1
        reading2_display = reading2[:200] + "..." if len(reading2) > 200 else reading2

        if language == 'fr':
            self.console.print(f"\n[bold yellow]📖 DÉSACCORD DE LECTURE[/bold yellow]")
            self.console.print(f"[dim]Question: {question_text[:80]}...[/dim]")

            # LLM1 reading
            self.console.print(f"\n[cyan]{llm1_name.upper()} (confiance: {conf1:.0%}):[/cyan]")
            self.console.print(f"  {reading1_display}")

            # LLM2 reading
            self.console.print(f"\n[magenta]{llm2_name.upper()} (confiance: {conf2:.0%}):[/magenta]")
            self.console.print(f"  {reading2_display}")

            self.console.print(f"\n[bold]Options:[/bold]")
            self.console.print(f"  1 = Lecture de {llm1_name}")
            self.console.print(f"  2 = Lecture de {llm2_name}")
            self.console.print(f"  Ou entrez directement la bonne lecture")
            self.console.print(f"  Entrée = Lecture avec la plus haute confiance")

            choice = Prompt.ask("Choix de la lecture", default="")
            self.console.print("")
        else:
            self.console.print(f"\n[bold yellow]📖 READING DISAGREEMENT[/bold yellow]")
            self.console.print(f"[dim]Question: {question_text[:80]}...[/dim]")

            # LLM1 reading
            self.console.print(f"\n[cyan]{llm1_name.upper()} (confidence: {conf1:.0%}):[/cyan]")
            self.console.print(f"  {reading1_display}")

            # LLM2 reading
            self.console.print(f"\n[magenta]{llm2_name.upper()} (confidence: {conf2:.0%}):[/magenta]")
            self.console.print(f"  {reading2_display}")

            self.console.print(f"\n[bold]Options:[/bold]")
            self.console.print(f"  1 = {llm1_name}'s reading")
            self.console.print(f"  2 = {llm2_name}'s reading")
            self.console.print(f"  Or enter the correct reading directly")
            self.console.print(f"  Enter = Reading with highest confidence")

            choice = Prompt.ask("Reading choice", default="")
            self.console.print("")

        # Process choice
        if not choice or choice.strip() == "":
            # Use highest confidence
            if conf1 >= conf2:
                return reading1
            else:
                return reading2
        elif choice.strip() == "1":
            return reading1
        elif choice.strip() == "2":
            return reading2
        else:
            # Direct reading input
            return choice.strip()


# =============================================================================
# LIVE PROGRESS DISPLAY - Real-time progress for parallel processing
# =============================================================================

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
    confidence_count: int = 0  # Track number of confidence values for rolling average
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

    def mark_done(self, copy_index: int, score: float, max_score: float, student_name: str = None):
        """Convenience method to mark a copy as done with score."""
        status = self.copies.get(copy_index, CopyStatus())
        update_kwargs = {
            "status": "done",
            "score": score,
            "max_score": max_score,
            "questions_done": status.questions_total
        }
        if student_name:
            update_kwargs["student_name"] = student_name
        self.update_copy(copy_index, **update_kwargs)
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
        # Proper rolling average using separate counter
        self.stats.confidence_count += 1
        count = self.stats.confidence_count
        # Rolling average: new_avg = old_avg + (new_value - old_avg) / count
        self.stats.avg_confidence = (
            self.stats.avg_confidence + (confidence - self.stats.avg_confidence) / count
        )

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

            # Don't use format specifier on icon - it contains Rich markup
            lines.append(f"  [dim]{idx:2d}[/dim] {name:12s} {icon}  {score_str}")

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
            student_name = data.get('student_name')
            display.mark_done(copy_idx, score, max_score, student_name=student_name)

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
