"""
CLI Interface for AI correction system.

Handles user interactions with rich console prompts:
- Scale confirmation
- Doubts review
- Results display
"""

from typing import Dict, List, Optional, Tuple, Any
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from core.models import CopyDocument, GradedCopy, TeacherDecision, ConfidenceLevel


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

    def show_header(self, title: str = "AI Correction System"):
        """Display the application header."""
        self.console.print(f"\n[bold green]{title}[/bold green]\n")

    def show_info(self, message: str):
        """Display an info message."""
        self.console.print(f"[cyan]{message}[/cyan]")

    def show_success(self, message: str):
        """Display a success message."""
        self.console.print(f"[green]✓ {message}[/green]")

    def show_warning(self, message: str):
        """Display a warning message."""
        self.console.print(f"[yellow]⚠ {message}[/yellow]")

    def show_error(self, message: str):
        """Display an error message."""
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
        Present doubtful cases and collect user decisions.

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
            self.console.print(f"\n[bold yellow]{len(doubts)} cas(s) douteux à réviser[/bold yellow]")
        else:
            self.console.print(f"\n[bold yellow]{len(doubts)} doubtful case(s) to review[/bold yellow]")

        for i, (copy, graded, q_id, confidence) in enumerate(doubts, 1):
            decision = self._review_single_doubt(copy, graded, q_id, confidence, i, len(doubts), language)
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
        Review a single doubtful case.

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

        # Create panel with doubt details
        conf_display = (confidence or 0.5) * 100  # Handle None
        if language == 'fr':
            panel_content = f"""[bold]Copie:[/bold] {copy.id}
[bold]Question:[/bold] {question_id}
[bold]Note proposée:[/bold] {current_grade} points
[bold]Confiance:[/bold] {conf_display:.0f}%
[bold]Justification:[/bold] {reasoning[:200]}..."""
        else:
            panel_content = f"""[bold]Copy:[/bold] {copy.id}
[bold]Question:[/bold] {question_id}
[bold]Proposed grade:[/bold] {current_grade} points
[bold]Confidence:[/bold] {conf_display:.0f}%
[bold]Reasoning:[/bold] {reasoning[:200]}..."""

        self.console.print(Panel(
            panel_content,
            title=f"[bold yellow]Cas douteux {current}/{total}[/bold yellow]",
            border_style="yellow"
        ))

        # Ask for decision - simplified options
        if language == 'fr':
            options = [
                "Accepter la note proposée",
                "Modifier la note",
                "Voir le contexte (question + réponse)"
            ]
        else:
            options = [
                "Accept proposed grade",
                "Modify grade",
                "View context (question + answer)"
            ]

        self.console.print("\nOptions:")
        for i, opt in enumerate(options, 1):
            self.console.print(f"  [{i}] {opt}")

        choice = Prompt.ask(
            "Votre choix",
            choices=["1", "2", "3"],
            default="1"
        )

        if choice == "1":
            # Accept
            return Decision(
                question_id=question_id,
                copy_id=copy.id,
                original_grade=current_grade,
                new_grade=current_grade,
                propagate=self._ask_propagate(language)
            )
        elif choice == "2":
            # Modify
            new_grade = self._ask_new_grade(current_grade, language)
            return Decision(
                question_id=question_id,
                copy_id=copy.id,
                original_grade=current_grade,
                new_grade=new_grade,
                propagate=self._ask_propagate(language)
            )
        elif choice == "3":
            # Show context - display question and detected answer
            self._show_question_context(copy, question_id, reasoning, language)
            # Re-ask for decision
            return self._review_single_doubt(copy, graded, question_id, confidence, current, total, language)

        return None

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
        language: str = 'fr'
    ):
        """
        Display the final summary.

        Args:
            session_id: Session identifier
            copies_count: Total copies processed
            graded_count: Number of graded copies
            scores: List of all scores
            language: Language for display
        """
        self.console.print("\n" + "="*60)

        if language == 'fr':
            self.console.print("[bold green]✓ Correction terminée[/bold green]\n")
            self.console.print(f"[bold]Session:[/bold] {session_id}")
            self.console.print(f"[bold]Copies traitées:[/bold] {copies_count}")
            self.console.print(f"[bold]Copies corrigées:[/bold] {graded_count}")
        else:
            self.console.print("[bold green]✓ Grading complete[/bold green]\n")
            self.console.print(f"[bold]Session:[/bold] {session_id}")
            self.console.print(f"[bold]Copies processed:[/bold] {copies_count}")
            self.console.print(f"[bold]Copies graded:[/bold] {graded_count}")

        if scores:
            avg = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)

            if language == 'fr':
                self.console.print(f"[bold]Note moyenne:[/bold] {avg:.1f}")
                self.console.print(f"[bold]Étendue:[/bold] {min_score:.1f} - {max_score:.1f}")
            else:
                self.console.print(f"[bold]Average score:[/bold] {avg:.1f}")
                self.console.print(f"[bold]Range:[/bold] {min_score:.1f} - {max_score:.1f}")

            # Score distribution
            self._show_distribution(scores, language)

        self.console.print("\n" + "="*60)

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

    def _show_distribution(self, scores: List[float], language: str):
        """Show score distribution as a simple histogram."""
        if not scores:
            return

        # Create buckets
        max_score = max(scores) if scores else 20
        bucket_size = max_score / 5 if max_score > 0 else 4
        buckets = {}

        for score in scores:
            bucket_idx = int(score / bucket_size) if bucket_size > 0 else 0
            bucket_key = f"{bucket_idx * bucket_size:.0f}-{(bucket_idx + 1) * bucket_size:.0f}"
            buckets[bucket_key] = buckets.get(bucket_key, 0) + 1

        if language == 'fr':
            self.console.print("\n[bold]Distribution des notes:[/bold]")
        else:
            self.console.print("\n[bold]Score distribution:[/bold]")

        for bucket in sorted(buckets.keys()):
            count = buckets[bucket]
            bar = "█" * min(count, 20)
            self.console.print(f"  {bucket}: {bar} ({count})")

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
        Display a disagreement between two LLMs and ask user to choose.

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
        from rich.prompt import Prompt
        from rich.panel import Panel

        if language == 'fr':
            reasoning1 = llm1_result.get('internal_reasoning', '')
            reasoning2 = llm2_result.get('internal_reasoning', '')

            # Build content for panel
            panel_content = f"""[dim]Question: {question_text[:80]}{'...' if len(question_text) > 80 else ''}[/dim]

[cyan bold]{llm1_name.upper()}:[/cyan bold]
  Note: [bold]{llm1_result.get('grade')}/{llm1_result.get('max_points', '?')}[/bold]
  Confiance: {(llm1_result.get('confidence') or 0):.0%}
  Raisonnement: [dim]{reasoning1}[/dim]

[magenta bold]{llm2_name.upper()}:[/magenta bold]
  Note: [bold]{llm2_result.get('grade')}/{llm2_result.get('max_points', '?')}[/bold]
  Confiance: {(llm2_result.get('confidence') or 0):.0%}
  Raisonnement: [dim]{reasoning2}[/dim]"""

            self.console.print(Panel(
                panel_content,
                title=f"[bold yellow]⚠ DÉSACCORD - {question_id}[/bold yellow]",
                border_style="yellow"
            ))

            self.console.print(f"\n[bold]Options:[/bold]")
            self.console.print(f"  LLM1 = Note de {llm1_name} → {llm1_result.get('grade')}")
            self.console.print(f"  LLM2 = Note de {llm2_name} → {llm2_result.get('grade')}")
            self.console.print(f"  Ou entrez directement une note")

            avg_grade = (llm1_result.get('grade', 0) + llm2_result.get('grade', 0)) / 2
            choice = Prompt.ask(f"[bold]Note[/bold] [dim](Entrée = moyenne {avg_grade:.2f})[/dim]")
            self.console.print("")  # Add newline after input

        else:
            reasoning1 = llm1_result.get('internal_reasoning', '')
            reasoning2 = llm2_result.get('internal_reasoning', '')

            # Build content for panel
            panel_content = f"""[dim]Question: {question_text[:80]}{'...' if len(question_text) > 80 else ''}[/dim]

[cyan bold]{llm1_name.upper()}:[/cyan bold]
  Grade: [bold]{llm1_result.get('grade')}/{llm1_result.get('max_points', '?')}[/bold]
  Confidence: {(llm1_result.get('confidence') or 0):.0%}
  Reasoning: [dim]{reasoning1}[/dim]

[magenta bold]{llm2_name.upper()}:[/magenta bold]
  Grade: [bold]{llm2_result.get('grade')}/{llm2_result.get('max_points', '?')}[/bold]
  Confidence: {(llm2_result.get('confidence') or 0):.0%}
  Reasoning: [dim]{reasoning2}[/dim]"""

            self.console.print(Panel(
                panel_content,
                title=f"[bold yellow]⚠ DISAGREEMENT - {question_id}[/bold yellow]",
                border_style="yellow"
            ))

            self.console.print(f"\n[bold]Options:[/bold]")
            self.console.print(f"  LLM1 = {llm1_name}'s grade → {llm1_result.get('grade')}")
            self.console.print(f"  LLM2 = {llm2_name}'s grade → {llm2_result.get('grade')}")
            self.console.print(f"  Or enter a grade directly")

            avg_grade = (llm1_result.get('grade', 0) + llm2_result.get('grade', 0)) / 2
            choice = Prompt.ask(f"[bold]Grade[/bold] [dim](Enter = average {avg_grade:.2f})[/dim]")
            self.console.print("")  # Add newline after input

        # Process choice - return (grade, feedback_source)
        # Empty input = average
        if not choice or choice.strip() == "":
            return (llm1_result.get('grade', 0) + llm2_result.get('grade', 0)) / 2, "merge"
        # Choice "LLM1" = LLM1's grade
        elif choice.strip().upper() == "LLM1" or choice.strip().upper() == "A":
            return llm1_result.get('grade'), "llm1"
        # Choice "LLM2" = LLM2's grade
        elif choice.strip().upper() == "LLM2" or choice.strip().upper() == "B":
            return llm2_result.get('grade'), "llm2"
        # Try to parse as a number (direct grade input) - PRIORITY
        else:
            try:
                manual_grade = float(choice.strip().replace(',', '.'))
                return manual_grade, "merge"
            except ValueError:
                # Invalid input - use average
                return (llm1_result.get('grade', 0) + llm2_result.get('grade', 0)) / 2, "merge"

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
