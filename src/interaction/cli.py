"""
CLI Interface for AI correction system.

Handles user interactions with rich console prompts:
- Scale confirmation
- Doubts review
- Results display

UX Principles:
- Progressive disclosure: Show info when needed
- Visual hierarchy: Color and spacing guide attention
- Smart defaults: Minimize user input
- Forgiving design: Easy to understand and recover
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text

from core.models import CopyDocument, GradedCopy, TeacherDecision, ConfidenceLevel


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
        pages_per_student: int = 2,
        language: str = "auto",
        llm1_name: str = None,
        llm2_name: str = None,
        display_language: str = "fr"
    ):
        """
        Display compact startup screen with configuration.

        Args:
            pdf_files: List of PDF file paths
            mode: Grading mode ("single", "comparison", "conversation")
            pages_per_student: Pages per student
            language: Content language ("auto", "fr", "en")
            llm1_name: First LLM name (for comparison mode)
            llm2_name: Second LLM name (for comparison mode)
            display_language: UI language
        """
        today = datetime.now().strftime("%d %b. %Y")

        # Header
        header = Panel(
            f"[bold cyan]🎓  AI Correction v2.0[/bold cyan]  [dim]{today}[/dim]",
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
            config_lines.append(f"[bold]🔧 Pages/student:[/bold] {pages_per_student}")
            lang_display = "Auto-detect" if language == "auto" else language.upper()
            config_lines.append(f"[bold]🌐 Language:[/bold]     {lang_display}")
        else:
            config_lines.append(f"[bold]📁 PDFs:[/bold]        {file_count} fichiers")
            mode_display = {
                "single": "LLM Simple",
                "comparison": "Comparaison Double LLM",
                "conversation": "Conversation"
            }.get(mode, mode)
            config_lines.append(f"[bold]🤖 Mode:[/bold]        {mode_display}")
            config_lines.append(f"[bold]🔧 Pages/élève:[/bold]  {pages_per_student}")
            lang_display = "Auto-détection" if language == "auto" else language.upper()
            config_lines.append(f"[bold]🌐 Langue:[/bold]      {lang_display}")

        # LLM info for comparison mode
        if mode == "comparison" and llm1_name and llm2_name:
            if display_language == "en":
                config_lines.append(f"[dim]LLM1: {llm1_name}[/dim]")
                config_lines.append(f"[dim]LLM2: {llm2_name}[/dim]")
            else:
                config_lines.append(f"[dim]LLM1: {llm1_name}[/dim]")
                config_lines.append(f"[dim]LLM2: {llm2_name}[/dim]")

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
        reasoning = graded.internal_reasoning.get(question_id, "")

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
        session_id: str,
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
            session_id: Session identifier
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

        # Session info
        duration_str = ""
        if duration:
            if duration < 60:
                duration_str = f"{int(duration)}s"
            elif duration < 3600:
                duration_str = f"{int(duration / 60)}m {int(duration % 60)}s"
            else:
                duration_str = f"{int(duration / 3600)}h {int((duration % 3600) / 60)}m"

        if language == 'fr':
            mode_display = {"single": "LLM Simple", "comparison": "Double LLM", "conversation": "Conversation"}.get(mode, mode)
            lines.append(f"[bold]Session:[/bold] {session_id}  │  [bold]Durée:[/bold] {duration_str}  │  [bold]Mode:[/bold] {mode_display}")
        else:
            mode_display = {"single": "Single LLM", "comparison": "Dual LLM", "conversation": "Conversation"}.get(mode, mode)
            lines.append(f"[bold]Session:[/bold] {session_id}  │  [bold]Duration:[/bold] {duration_str}  │  [bold]Mode:[/bold] {mode_display}")

        lines.append("")

        # Core stats
        if scores:
            avg = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)

            if language == 'fr':
                lines.append(f"[bold]Copies:[/bold] {copies_count}  │  [bold]Moyenne:[/bold] {avg:.1f}/20 ({avg/20*100:.0f}%)  │  [bold]Étendue:[/bold] {min_score:.1f} - {max_score:.1f}")
            else:
                lines.append(f"[bold]Copies:[/bold] {copies_count}  │  [bold]Average:[/bold] {avg:.1f}/20 ({avg/20*100:.0f}%)  │  [bold]Range:[/bold] {min_score:.1f} - {max_score:.1f}")

            lines.append("")

            # Score distribution
            if language == 'fr':
                lines.append("[bold cyan]Distribution des notes:[/bold cyan]")
            else:
                lines.append("[bold cyan]Score Distribution:[/bold cyan]")

            dist_str = self._render_distribution_bar(scores)
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
                    name = performer.get('name', '???')
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

    def _render_distribution_bar(self, scores: List[float], max_bar: int = 10) -> str:
        """Render a distribution bar chart."""
        if not scores:
            return ""

        # Create buckets: 0-4, 5-9, 10-14, 15-20
        buckets = {"0-4": 0, "5-9": 0, "10-14": 0, "15-20": 0}
        for score in scores:
            if score < 5:
                buckets["0-4"] += 1
            elif score < 10:
                buckets["5-9"] += 1
            elif score < 15:
                buckets["10-14"] += 1
            else:
                buckets["15-20"] += 1

        max_count = max(buckets.values()) if buckets else 1
        lines = []

        for bucket, count in buckets.items():
            if max_count > 0:
                bar_len = int((count / max_count) * max_bar)
            else:
                bar_len = 0
            bar = "█" * bar_len + "░" * (max_bar - bar_len)

            # Color based on bucket
            if bucket in ["0-4", "5-9"]:
                color = "red"
            elif bucket == "10-14":
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
        reasoning1 = llm1_result.get('internal_reasoning', '')[:100]
        reasoning2 = llm2_result.get('internal_reasoning', '')[:100]
        grade1 = llm1_result.get('grade', 0) or 0
        grade2 = llm2_result.get('grade', 0) or 0
        conf1 = (llm1_result.get('confidence') or 0) * 100
        conf2 = (llm2_result.get('confidence') or 0) * 100

        # Create side-by-side panels
        if language == 'fr':
            panel1_content = (
                f"[bold]Note:[/bold] {grade1:.1f}\n"
                f"[bold]Confiance:[/bold] {conf1:.0f}%\n"
                f"[dim]{reasoning1}{'...' if len(llm1_result.get('internal_reasoning', '')) > 100 else ''}[/dim]"
            )
            panel2_content = (
                f"[bold]Note:[/bold] {grade2:.1f}\n"
                f"[bold]Confiance:[/bold] {conf2:.0f}%\n"
                f"[dim]{reasoning2}{'...' if len(llm2_result.get('internal_reasoning', '')) > 100 else ''}[/dim]"
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
                f"[dim]{reasoning1}{'...' if len(llm1_result.get('internal_reasoning', '')) > 100 else ''}[/dim]"
            )
            panel2_content = (
                f"[bold]Grade:[/bold] {grade2:.1f}\n"
                f"[bold]Confidence:[/bold] {conf2:.0f}%\n"
                f"[dim]{reasoning2}{'...' if len(llm2_result.get('internal_reasoning', '')) > 100 else ''}[/dim]"
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
