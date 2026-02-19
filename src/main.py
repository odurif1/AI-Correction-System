"""
Main CLI entry point for the AI correction system.

Usage:
    python src/main.py correct copies/*.pdf
    python src/main.py correct copies/*.pdf --auto
    python src/main.py correct copies.pdf --pages-per-student 2
    python src/main.py api --port 8000
    python src/main.py status <session_id>
    python src/main.py export <session_id>
    python src/main.py list

Options:
    --auto      Mode automatique sans interaction utilisateur.
                - Le barème est détecté automatiquement pendant la correction
                - En cas de désaccord entre les 2 IA: prend la moyenne
                Sans --auto: demande choix utilisateur pour les désaccords
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Dict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.session import GradingSessionOrchestrator
from core.workflow import WorkflowCallbacks
from core.workflow_state import CorrectionState, WorkflowPhase
from utils.sorting import natural_sort_key
from config.settings import get_settings
from config.constants import DEFAULT_PARALLEL_COPIES
from interaction.cli import CLI
from interaction.live_progress import LiveProgressDisplay


def check_api_key() -> bool:
    """Check if an AI API key is configured."""
    settings = get_settings()
    if not settings.gemini_api_key and not settings.openai_api_key:
        console = Console()
        console.print("[red]Error: No AI API key configured![/red]")
        console.print("Set AI_CORRECTION_GEMINI_API_KEY or AI_CORRECTION_OPENAI_API_KEY environment variable.")
        return False
    return True


def validate_pdf_path(path_str: str) -> tuple[bool, str]:
    """
    Validate a PDF path for security and correctness.

    Args:
        path_str: Path string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    from pathlib import Path

    # Check for empty path
    if not path_str or not path_str.strip():
        return False, "Empty path provided"

    path = Path(path_str)

    # Check for path traversal attempts
    try:
        # Resolve to absolute path
        resolved = path.resolve()
    except (OSError, ValueError) as e:
        return False, f"Invalid path: {e}"

    # Check for suspicious patterns
    suspicious = ['..', '~', '$', '|', ';', '&', '`']
    for pattern in suspicious:
        if pattern in path_str:
            return False, f"Suspicious pattern '{pattern}' in path"

    # Check extension
    if path.suffix.lower() != '.pdf':
        return False, f"Not a PDF file: {path.suffix}"

    return True, ""


def create_workflow_callbacks(
    cli: CLI,
    language: str,
    auto_mode: bool
) -> WorkflowCallbacks:
    """
    Create workflow callbacks that integrate with CLI.

    Args:
        cli: CLI instance for user interaction
        language: Language for prompts
        auto_mode: Whether to auto-resolve disagreements

    Returns:
        WorkflowCallbacks instance
    """
    async def on_disagreement(
        question_id: str,
        question_text: str,
        llm1_name: str,
        llm1_result: dict,
        llm2_name: str,
        llm2_result: dict,
        max_points: float
    ) -> tuple[float, str]:
        """Handle grading disagreement."""
        grade1 = llm1_result.get('grade', 0) or 0
        grade2 = llm2_result.get('grade', 0) or 0
        average_grade = (grade1 + grade2) / 2

        if auto_mode:
            cli.console.print(
                f"    [yellow]⚠ {llm1_name}: {grade1} vs "
                f"{llm2_name}: {grade2} → moyenne: {average_grade:.2f}[/yellow]"
            )
            return average_grade, "merge"

        try:
            llm1_result['max_points'] = max_points
            llm2_result['max_points'] = max_points
            return cli.show_disagreement(
                question_id=question_id or "Question",
                question_text=question_text,
                llm1_name=llm1_name,
                llm1_result=llm1_result,
                llm2_name=llm2_name,
                llm2_result=llm2_result,
                language=language
            )
        except (EOFError, KeyboardInterrupt):
            cli.console.print(f"    [dim]Utilisation de la moyenne: {average_grade:.2f}[/dim]")
            return average_grade, "merge"

    async def on_name_disagreement(llm1_result: dict, llm2_result: dict) -> str:
        """Handle name disagreement."""
        if auto_mode:
            if llm1_result.get('confidence', 0) >= llm2_result.get('confidence', 0):
                return llm1_result.get('name') or "Inconnu"
            return llm2_result.get('name') or "Inconnu"

        try:
            return cli.show_name_disagreement(
                llm1_result=llm1_result,
                llm2_result=llm2_result,
                language=language
            )
        except (EOFError, KeyboardInterrupt):
            if llm1_result.get('confidence', 0) >= llm2_result.get('confidence', 0):
                return llm1_result.get('name') or "Inconnu"
            return llm2_result.get('name') or "Inconnu"

    async def on_reading_disagreement(
        llm1_result: dict,
        llm2_result: dict,
        question_text: str,
        image_path
    ) -> str:
        """Handle reading disagreement."""
        if auto_mode:
            if llm1_result.get('confidence', 0) >= llm2_result.get('confidence', 0):
                return llm1_result.get('reading', '')
            return llm2_result.get('reading', '')

        try:
            return cli.show_reading_disagreement(
                llm1_result=llm1_result,
                llm2_result=llm2_result,
                question_text=question_text,
                image_path=image_path,
                language=language
            )
        except (EOFError, KeyboardInterrupt):
            if llm1_result.get('confidence', 0) >= llm2_result.get('confidence', 0):
                return llm1_result.get('reading', '')
            return llm2_result.get('reading', '')

    return WorkflowCallbacks(
        on_disagreement=on_disagreement,
        on_name_disagreement=on_name_disagreement,
        on_reading_disagreement=on_reading_disagreement
    )


# ==============================================================================
# Helper functions for command_correct
# ==============================================================================

def create_disagreement_callback(
    cli: CLI,
    state: CorrectionState,
    orchestrator,
    jurisprudence: Dict
):
    """
    Create disagreement callback for grading conflicts.

    Args:
        cli: CLI instance
        state: CorrectionState for language/mode
        orchestrator: GradingSessionOrchestrator
        jurisprudence: Mutable dict for storing decisions
    """
    async def callback(
        question_id: str,
        question_text: str,
        llm1_name: str,
        llm1_result: dict,
        llm2_name: str,
        llm2_result: dict,
        max_points: float
    ) -> tuple:
        grade1 = llm1_result.get('grade', 0) or 0
        grade2 = llm2_result.get('grade', 0) or 0
        average_grade = (grade1 + grade2) / 2

        # Check jurisprudence
        if question_id in jurisprudence:
            past = jurisprudence[question_id]
            cli.console.print(f"    [dim]📜 Jurisprudence: décision passée = {past['decision']:.1f}/{max_points}[/dim]")

        # Auto mode: use average
        if state.auto_mode:
            cli.console.print(f"    [yellow]⚠ {llm1_name}: {grade1} vs {llm2_name}: {grade2} → moyenne: {average_grade:.2f}[/yellow]")
            jurisprudence[question_id] = {
                'question_text': question_text,
                'decision': average_grade,
                'llm1_grade': grade1,
                'llm2_grade': grade2,
                'max_points': max_points,
                'auto': True
            }
            if hasattr(orchestrator.ai, 'set_jurisprudence'):
                orchestrator.ai.set_jurisprudence(jurisprudence)
            return average_grade, "merge"

        # Interactive mode
        llm1_result['max_points'] = max_points
        llm2_result['max_points'] = max_points
        try:
            chosen_grade, feedback_source = cli.show_disagreement(
                question_id=question_id or "Question",
                question_text=question_text,
                llm1_name=llm1_name,
                llm1_result=llm1_result,
                llm2_name=llm2_name,
                llm2_result=llm2_result,
                language=state.language
            )
            jurisprudence[question_id] = {
                'question_text': question_text,
                'decision': chosen_grade,
                'llm1_grade': grade1,
                'llm2_grade': grade2,
                'max_points': max_points,
                'auto': False,
                'feedback_source': feedback_source
            }
            if hasattr(orchestrator.ai, 'set_jurisprudence'):
                orchestrator.ai.set_jurisprudence(jurisprudence)
            return chosen_grade, feedback_source
        except (EOFError, KeyboardInterrupt):
            cli.console.print(f"    [dim]Utilisation de la moyenne: {average_grade:.2f}[/dim]")
            jurisprudence[question_id] = {
                'question_text': question_text,
                'decision': average_grade,
                'llm1_grade': grade1,
                'llm2_grade': grade2,
                'max_points': max_points,
                'auto': True
            }
            if hasattr(orchestrator.ai, 'set_jurisprudence'):
                orchestrator.ai.set_jurisprudence(jurisprudence)
            return average_grade, "merge"

    return callback


def create_name_disagreement_callback(cli: CLI, state: CorrectionState):
    """Create callback for name detection conflicts."""
    async def callback(llm1_result: Dict, llm2_result: Dict) -> str:
        if state.auto_mode:
            if llm1_result.get('confidence', 0) >= llm2_result.get('confidence', 0):
                return llm1_result.get('name') or "Inconnu"
            return llm2_result.get('name') or "Inconnu"

        try:
            return cli.show_name_disagreement(
                llm1_result=llm1_result,
                llm2_result=llm2_result,
                language=state.language
            )
        except (EOFError, KeyboardInterrupt):
            if llm1_result.get('confidence', 0) >= llm2_result.get('confidence', 0):
                return llm1_result.get('name') or "Inconnu"
            return llm2_result.get('name') or "Inconnu"

    return callback


def create_reading_disagreement_callback(cli: CLI, state: CorrectionState):
    """Create callback for reading (transcription) conflicts."""
    async def callback(
        llm1_result: Dict,
        llm2_result: Dict,
        question_text: str,
        image_path
    ) -> str:
        if state.auto_mode:
            if llm1_result.get('confidence', 0) >= llm2_result.get('confidence', 0):
                return llm1_result.get('reading', '')
            return llm2_result.get('reading', '')

        try:
            return cli.show_reading_disagreement(
                llm1_result=llm1_result,
                llm2_result=llm2_result,
                question_text=question_text,
                image_path=image_path,
                language=state.language
            )
        except (EOFError, KeyboardInterrupt):
            if llm1_result.get('confidence', 0) >= llm2_result.get('confidence', 0):
                return llm1_result.get('reading', '')
            return llm2_result.get('reading', '')

    return callback


async def command_correct(args):
    """Run correction on PDF files with interactive workflow."""
    if not check_api_key():
        return 1

    cli = CLI()
    cli.show_header()

    # Get PDF paths with validation
    pdf_paths = []
    for pattern in args.pdfs:
        # Validate path first
        is_valid, error = validate_pdf_path(pattern)
        if not is_valid:
            cli.show_warning(f"Invalid path '{pattern}': {error}")
            continue

        path = Path(pattern)
        if path.exists():
            if path.is_dir():
                # For directories, validate each PDF found
                for pdf_file in path.glob("*.pdf"):
                    is_valid, _ = validate_pdf_path(str(pdf_file))
                    if is_valid:
                        pdf_paths.append(str(pdf_file))
            else:
                pdf_paths.append(str(path))
        else:
            cli.show_warning(f"Path not found: {pattern}")

    if not pdf_paths:
        cli.show_error("No PDF files found to process.")
        return 1

    cli.show_info(f"Found {len(pdf_paths)} PDF file(s) to process")

    # Initialize workflow state (replaces mutable dicts)
    state = CorrectionState(
        language='fr',
        auto_mode=args.auto,
        phase=WorkflowPhase.INITIALIZATION
    )

    # Jurisprudence: store user decisions to inform future grading
    # (kept as mutable dict for compatibility with orchestrator.ai.set_jurisprudence)
    jurisprudence: Dict = {}

    # Validate required options
    if not args.pages_per_student:
        cli.console.print("[red]Erreur: L'option --pages-per-student est requise.[/red]")
        cli.console.print("[dim]Exemple: --pages-per-student 2 pour 2 pages par élève[/dim]")
        cli.console.print("[dim]Utilisez --help pour plus d'informations[/dim]")
        return 1

    # Create orchestrator with callbacks (using helper functions)
    orchestrator = GradingSessionOrchestrator(
        pdf_paths,
        disagreement_callback=None,  # Will be set after orchestrator is created
        name_disagreement_callback=None,
        reading_disagreement_callback=None,
        skip_reading_consensus=args.skip_reading,
        force_single_llm=args.single,
        pages_per_student=args.pages_per_student,
        second_reading=args.second_reading,
        parallel=args.parallel
    )

    # Create callbacks with access to orchestrator
    disagreement_callback = create_disagreement_callback(cli, state, orchestrator, jurisprudence)
    name_disagreement_callback = create_name_disagreement_callback(cli, state)
    reading_disagreement_callback = create_reading_disagreement_callback(cli, state)

    # Set callbacks on orchestrator
    orchestrator.disagreement_callback = disagreement_callback
    orchestrator.name_disagreement_callback = name_disagreement_callback
    orchestrator.reading_disagreement_callback = reading_disagreement_callback

    # Show which LLMs are being used
    if hasattr(orchestrator.ai, 'providers'):
        cli.console.print(f"\n[bold cyan]🤖 Modèles utilisés (mode comparaison):[/bold cyan]")
        for i, (name, provider) in enumerate(orchestrator.ai.providers):
            cli.console.print(f"  LLM{i+1}: [yellow]{name}[/yellow]")
        cli.console.print("")
    else:
        model_name = getattr(orchestrator.ai, 'model', None) or get_settings().gemini_model
        cli.console.print(f"\n[bold cyan]🤖 Modèle utilisé (mode simple):[/bold cyan]")
        cli.console.print(f"  [yellow]{model_name}[/yellow]")
        cli.console.print("")

    # ============================================================
    # Phase 1: Analyze
    # ============================================================
    state = state.with_phase(WorkflowPhase.ANALYSIS)
    cli.show_info("Analyzing copies...")
    cli.show_info("This may take a few minutes depending on API response time...")

    try:
        analysis = await orchestrator.analyze_only()
    except Exception as e:
        cli.show_error(f"Analysis failed: {e}")
        return 1

    # Get detected language and update state
    language = analysis.get('language', 'fr')
    state = state.with_language(language)

    # Show analysis result in detected language
    if language == 'fr':
        cli.show_success(f"{analysis['copies_count']} copie(s) analysée(s)")
    else:
        cli.show_success(f"Analyzed {analysis['copies_count']} copies")

    # ============================================================
    # Phase 2: Setup Scale (will be detected during grading)
    # ============================================================

    # Check if questions were detected or will be detected during grading
    questions_detected_during_grading = analysis.get('questions_detected_during_grading', False)

    if not analysis['questions'] and not questions_detected_during_grading:
        cli.show_error("Aucune question détectée. L'analyse a peut-être échoué.")
        return 1

    if questions_detected_during_grading:
        # In individual mode, questions will be detected during grading
        cli.show_info("Questions et barème seront détectés automatiquement pendant la correction.")
        # Use empty scale - will be populated during grading
        scale = {}
    else:
        # Use default scale of 1.0 for each question
        # The actual scale will be detected by the LLM during grading
        scale = {q: 1.0 for q in analysis['questions'].keys()}
        cli.show_info(f"Questions détectées: {list(analysis['questions'].keys())}")
        cli.show_info("Le barème sera détecté automatiquement pendant la correction.")

    orchestrator.confirm_scale(scale)

    # ============================================================
    # Phase 3: Grade All Copies (with live display)
    # ============================================================
    state = state.with_phase(WorkflowPhase.GRADING)
    console = cli.console
    is_comparison_mode = orchestrator._comparison_mode

    # Track LLM completion status for ordered display
    llm_status = {'results': {}, 'total': 2}
    provider_names = [name for name, _ in orchestrator.ai.providers] if is_comparison_mode else []

    # Track token usage per copy
    prev_tokens = {'total': 0}

    # Collect question results for sorted display at copy_done
    current_copy_questions = {}

    # Progress callback for real-time display
    async def progress_callback(event_type: str, data: dict):
        nonlocal prev_tokens

        if event_type == 'copy_start':
            copy_idx = data['copy_index']
            total = data['total_copies']
            name = data.get('student_name') or '???'
            llm_status['results'] = {}
            current_copy_questions.clear()  # Reset for new copy
            console.print(f"\n[bold cyan]━━━ Copie {copy_idx}/{total} ━━━[/bold cyan] [yellow]{name}[/yellow]")

        elif event_type == 'question_start':
            q_id = data['question_id']
            q_idx = data['question_index']
            total_q = data['total_questions']
            llm_status['results'] = {}  # Reset for new question
            console.print(f"  [dim]{q_id} ({q_idx}/{total_q})[/dim]", end="")

        elif event_type == 'llm_parallel_start':
            # Skip - too repetitive
            pass

        elif event_type == 'llm_complete':
            provider_index = data.get('provider_index', 0)
            provider = data.get('provider', '???')
            grade = data.get('grade')
            all_done = data.get('all_completed', False)

            # Store result by index for ordered display
            llm_status['results'][provider_index] = {'provider': provider, 'grade': grade}

            # Only display when ALL results are in (to maintain order)
            if all_done and len(llm_status['results']) == llm_status['total']:
                # Display in order (index 0, then index 1)
                first = True
                for idx in sorted(llm_status['results'].keys()):
                    r = llm_status['results'][idx]
                    if first:
                        prefix = " ▪ "
                        first = False
                    else:
                        prefix = " ┃ "
                    if r['grade'] is not None:
                        console.print(f"{prefix}[cyan]{r['provider']}:[/cyan] [bold]{r['grade']:.1f}[/bold]", end="")
                    else:
                        console.print(f"{prefix}[red]{r['provider']}: erreur[/red]", end="")
                console.print("")  # New line after both complete

        elif event_type == 'llm_error':
            provider = data.get('provider', '???')
            error = data.get('error', 'Unknown error')
            console.print(f"    [red]✗ {provider}: {error}[/red]")

        elif event_type == 'question_done':
            q_id = data['question_id']
            grade = data['grade']
            max_pts = data['max_points']
            agreement = data.get('agreement', True)
            final_method = data.get('final_method', 'consensus')

            # Color coding based on decision method
            # Green: consensus initial (accord immédiat)
            # Yellow: decision prise au cross verification
            # Orange: decision prise à l'ultimatum
            # Red: désaccord persistant (average ou user_choice)
            if final_method == 'consensus' or final_method == 'single_llm':
                color = "green"
                icon = "✓"
            elif final_method == 'verification_consensus':
                color = "yellow"
                icon = "✓"
            elif final_method == 'ultimatum_consensus':
                color = "dark_orange"
                icon = "✓"
            else:  # average, user_choice, merge
                color = "red"
                icon = "⚠"

            # Store for sorted display at copy_done
            current_copy_questions[q_id] = {
                'grade': grade,
                'max_pts': max_pts,
                'color': color,
                'icon': icon,
                'note': ''
            }

        elif event_type == 'copy_done':
            score = data['total_score']
            max_s = data['max_score']
            pct = (score / max_s * 100) if max_s > 0 else 0
            conf = data.get('confidence', 0.5) or 0.5

            # Display all questions sorted chronologically
            for qid in sorted(current_copy_questions.keys(), key=natural_sort_key):
                q = current_copy_questions[qid]
                console.print(f"  [{q['color']}]{q['icon']}[/{q['color']}] {qid}: [bold]{q['grade']:.1f}/{q['max_pts']}[/bold][dim]{q['note']}[/dim]")

            # Calculate tokens used for this copy
            token_usage = data.get('token_usage')
            tokens_str = ""
            if token_usage:
                current_total = token_usage.get('total_tokens', 0)
                tokens_this_copy = current_total - prev_tokens['total']
                prev_tokens['total'] = current_total
                if tokens_this_copy > 0:
                    tokens_str = f" [dim]│ {tokens_this_copy:,} tokens[/dim]"

            if pct >= 50:
                color = "green"
            else:
                color = "red"

            console.print(f"  [bold {color}]Total: {score:.1f}/{max_s} ({pct:.0f}%)[/bold {color}] [dim]confiance: {conf:.0%}[/dim]{tokens_str}")

        elif event_type == 'feedback_start':
            console.print(f"  [dim]Génération de l'appréciation...[/dim]", end="")

        elif event_type == 'feedback_done':
            feedback = data.get('feedback', '')
            if feedback:
                console.print(f" [green]✓[/green]")
                console.print(f"  [italic]{feedback}[/italic]")
            else:
                console.print(" [green]✓[/green]")

        # ===== CONVERSATION MODE EVENTS =====
        elif event_type == 'single_pass_start':
            num_q = data.get('num_questions', 0)
            providers = data.get('providers', [])
            # Shorten provider names for display
            short_names = [p.replace("gemini-", "g-").replace("-preview", "") for p in providers]
            console.print(f"  [dim]📤 Correction initiale: {len(providers)} LLM × {num_q} questions ({' vs '.join(short_names)})[/dim]")

        elif event_type == 'single_pass_complete':
            providers = data.get('providers', [])
            single_pass = data.get('single_pass', {})

            # Get all question IDs sorted naturally
            all_qids = set()
            for provider in providers:
                result = single_pass.get(provider, {})
                questions = result.get('questions', {})
                all_qids.update(questions.keys())
            sorted_qids = sorted(all_qids, key=natural_sort_key)

            # Build a readable table
            table = Table(show_header=True, header_style="bold dim", show_lines=False, box=None, padding=(0, 2))
            table.add_column("Question", style="bold", width=10)
            for idx, provider in enumerate(providers):
                # Cleaner provider name with index to distinguish
                if "gemini" in provider.lower():
                    if "3" in provider:
                        short_name = "Gemini 3"
                    elif "2.5" in provider or "flash" in provider.lower():
                        short_name = "Gemini 2.5"
                    else:
                        short_name = f"Gemini {idx+1}"
                elif "openai" in provider.lower() or "gpt" in provider.lower():
                    short_name = "GPT-4o" if "4o" in provider.lower() else "GPT"
                else:
                    short_name = f"LLM{idx+1}"
                table.add_column(short_name, justify="center", width=12)

            for qid in sorted_qids:
                row = [qid]
                for provider in providers:
                    result = single_pass.get(provider, {})
                    questions = result.get('questions', {})
                    q_data = questions.get(qid, {})
                    grade = q_data.get('grade', 0)
                    row.append(f"{grade:.1f}")  # Just the grade, no /max_pts (it's always /1 or /2, redundant)
                table.add_row(*row)

            console.print(table)

        elif event_type == 'analysis_complete':
            agreed = data.get('agreed', 0)
            flagged = data.get('flagged', 0)
            total = data.get('total', 0)
            flagged_questions = data.get('flagged_questions', [])

            if flagged == 0:
                console.print(f"  [green]✓ Analyse: {agreed}/{total} questions en accord[/green]")
            else:
                console.print(f"  [yellow]📊 Analyse: {agreed}/{total} accord, {flagged} désaccord(s)[/yellow]")
                for fq in flagged_questions:
                    qid = fq.get('question_id')
                    reason = fq.get('reason', '')
                    llm1_grade = fq.get('llm1', {}).get('grade', 0)
                    llm2_grade = fq.get('llm2', {}).get('grade', 0)
                    llm1_reading = fq.get('llm1', {}).get('reading', '')[:30]
                    llm2_reading = fq.get('llm2', {}).get('reading', '')[:30]

                    # Show appropriate info based on disagreement type
                    if 'Lectures' in reason or 'lecture' in reason:
                        # Reading disagreement - show readings
                        console.print(f"    [yellow]⚠ {qid}:[/yellow] {reason}")
                        console.print(f"        [dim]{llm1_reading}...[/dim] vs [dim]{llm2_reading}...[/dim]")
                    else:
                        # Grade disagreement - show grades
                        console.print(f"    [yellow]⚠ {qid}:[/yellow] {reason} ({llm1_grade:.1f} vs {llm2_grade:.1f})")

        elif event_type == 'verification_start':
            qid = data.get('question_id')
            reason = data.get('reason', '')
            console.print(f"  [dim]🔄 Vérification {qid}...[/dim]")

        elif event_type == 'question_done':
            qid = data.get('question_id')
            grade = data.get('grade', 0)
            max_pts = data.get('max_points', 1)
            method = data.get('method', 'unknown')
            agreement = data.get('agreement', True)

            # Determine display based on method
            if method == 'single_pass_consensus':
                color = "green"
                icon = "✓"
                note = ""
            elif 'verification' in method or 'ultimatum' in method:
                color = "yellow"
                icon = "✓"
                note = " (après vérification)"
            else:
                color = "green" if agreement else "red"
                icon = "✓" if agreement else "⚠"
                note = ""

            # Store for sorted display at copy_done
            current_copy_questions[qid] = {
                'grade': grade,
                'max_pts': max_pts,
                'color': color,
                'icon': icon,
                'note': note
            }

    # Run grading with progress updates
    if is_comparison_mode:
        # Show models being used
        providers_info = []
        for name, _ in orchestrator.ai.providers:
            providers_info.append(f"[cyan]{name}[/cyan]")
        console.print(f"\n[bold magenta]🤖 Double correction: {' vs '.join(providers_info)}[/bold magenta]")

    # Create live progress display for parallel processing visibility
    total_copies = len(orchestrator.session.copies)
    live_display = LiveProgressDisplay(console, total_copies=total_copies, language=language)

    with live_display:
        async def live_callback(event_type: str, data: dict):
            """Callback that updates live display and shows detailed progress."""
            # Update live display
            if event_type == 'copy_start':
                copy_idx = data.get('copy_index', 0)
                student = data.get('student_name', '???')
                questions = data.get('questions', [])
                live_display.mark_processing(
                    copy_idx,
                    student_name=student,
                    questions_total=len(questions) if questions else 0
                )

            elif event_type == 'question_done':
                copy_idx = data.get('copy_index', 0)
                if copy_idx:
                    live_display.mark_question_done(copy_idx)

            elif event_type == 'copy_done':
                copy_idx = data.get('copy_index', 0)
                score = data.get('total_score', 0) or 0
                max_score = data.get('max_score', 20) or 20
                live_display.mark_done(copy_idx, score, max_score)

            elif event_type == 'copy_error':
                copy_idx = data.get('copy_index', 0)
                error = data.get('error', 'Erreur')
                live_display.mark_error(copy_idx, error)

            # Also call original progress callback for detailed display
            await progress_callback(event_type, data)

        # Use dual-LLM grading when in comparison mode, otherwise use per-question
        if is_comparison_mode and hasattr(orchestrator.ai, 'grade_copy'):
            graded = await orchestrator.grade_all(progress_callback=live_callback)
        else:
            graded = await orchestrator.grade_all(progress_callback=live_callback)

    # Show final summary table
    console.print(f"\n[bold green]✓ {len(graded)} copie(s) corrigée(s)[/bold green]")

    # Build summary data with feedback
    copies_data = []
    for i, graded_copy in enumerate(graded, 1):
        copy = next(
            (c for c in orchestrator.session.copies if c.id == graded_copy.copy_id),
            None
        )
        if copy:
            copies_data.append({
                'copy_number': i,
                'student_name': copy.student_name,
                'total': graded_copy.total_score,
                'max': graded_copy.max_score,
                'confidence': graded_copy.confidence,
                'feedback': graded_copy.feedback
            })

    cli.show_all_copies_summary(copies_data, language=language)

    # ============================================================
    # Phase 4: Review Doubts (if not auto mode)
    # ============================================================
    state = state.with_phase(WorkflowPhase.VERIFICATION)
    if not args.auto:
        doubts = orchestrator.get_doubts(threshold=0.7)
        if doubts:
            decisions = cli.review_doubts(doubts, language=language)
            if decisions:
                await orchestrator.apply_decisions(decisions)
                cli.show_success(f"Applied {len(decisions)} decision(s)")

    # ============================================================
    # Phase 5: Calibration (internal consistency check)
    # ============================================================
    state = state.with_phase(WorkflowPhase.CALIBRATION)
    # Run calibration phase silently
    await orchestrator._calibration_phase()

    # ============================================================
    # Phase 6: Export
    # ============================================================
    state = state.with_phase(WorkflowPhase.EXPORT)
    cli.show_info("Exporting results...")

    exports = await orchestrator.export()

    # Mark complete
    from core.models import SessionStatus
    orchestrator.session.status = SessionStatus.COMPLETE
    orchestrator._save_sync()

    # ============================================================
    # Show Summary
    # ============================================================
    state = state.with_phase(WorkflowPhase.COMPLETE)
    scores = [g.total_score for g in graded]

    # Compact summary of all copies
    copies_data = []
    for i, g in enumerate(graded, 1):
        # Find the copy to get student name
        copy = next(
            (c for c in orchestrator.session.copies if c.id == g.copy_id),
            None
        )
        student_name = copy.student_name if copy else None

        copies_data.append({
            'copy_number': i,
            'student_name': student_name,
            'total': g.total_score,
            'max': g.max_score,
            'confidence': g.confidence
        })

    cli.show_all_copies_summary(copies_data, language=language)

    # Final summary
    cli.show_summary(
        session_id=orchestrator.session_id,
        copies_count=len(orchestrator.session.copies),
        graded_count=len(graded),
        scores=scores,
        language=language
    )

    # Show token usage
    if hasattr(orchestrator.ai, 'get_token_usage'):
        token_usage = orchestrator.ai.get_token_usage()
        if token_usage.get('total_tokens', 0) > 0:
            total = token_usage['total_tokens']
            prompt = token_usage.get('prompt_tokens', 0)
            completion = token_usage.get('completion_tokens', 0)

            cli.console.print(f"\n[bold cyan]📊 Token Usage:[/bold cyan]")
            cli.console.print(f"  Total: [bold]{total:,}[/bold] tokens")
            cli.console.print(f"  Prompt: {prompt:,} | Completion: {completion:,}")

            # Show by provider if available
            if 'by_provider' in token_usage:
                for provider_name, usage in token_usage['by_provider'].items():
                    cli.console.print(f"  [{provider_name}] {usage.get('total_tokens', 0):,} tokens ({usage.get('calls', 0)} calls)")

    # Annotate PDFs if requested
    if args.annotate:
        cli.show_info("Annotating PDFs...")
        # Would call PDFAnnotator here

    if language == 'fr':
        cli.show_success(f"Session sauvegardée: {orchestrator.session_id}")
    else:
        cli.show_success(f"Session saved: {orchestrator.session_id}")

    return 0


def command_status(args):
    """Show status of a session."""
    from storage.session_store import SessionStore

    console = Console()
    store = SessionStore(args.session)
    session = store.load_session()

    if not session:
        console.print(f"[red]Session not found: {args.session}[/red]")
        return 1

    # Display session info
    table = Table(title=f"Session: {session.session_id}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Status", session.status)
    table.add_row("Created", str(session.created_at))
    table.add_row("Total Copies", str(len(session.copies)))
    table.add_row("Graded Copies", str(len(session.graded_copies)))

    if session.graded_copies:
        scores = [g.total_score for g in session.graded_copies]
        table.add_row("Average", f"{sum(scores)/len(scores):.1f}")
        table.add_row("Min", f"{min(scores):.1f}")
        table.add_row("Max", f"{max(scores):.1f}")

    console.print(table)

    return 0


def command_analytics(args):
    """Show analytics for a session."""
    from storage.session_store import SessionStore

    console = Console()

    store = SessionStore(args.session)
    session = store.load_session()

    if not session:
        console.print(f"[red]Session not found: {args.session}[/red]")
        return 1

    orchestrator = GradingSessionOrchestrator(session_id=args.session)
    analytics = orchestrator.get_analytics()

    # Display analytics
    console.print(Panel(f"Session: {args.session}", title="Analytics"))

    table = Table(title="Score Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Mean", f"{analytics['mean_score']:.2f}")
    table.add_row("Median", f"{analytics['median_score']:.2f}")
    table.add_row("Std Dev", f"{analytics['std_dev']:.2f}")
    table.add_row("Min", f"{analytics['min_score']:.2f}")
    table.add_row("Max", f"{analytics['max_score']:.2f}")

    console.print(table)

    if analytics.get('score_distribution'):
        console.print("\n[bold]Score Distribution:[/bold]")
        for bucket, count in analytics['score_distribution'].items():
            bar = "█" * count
            console.print(f"  {bucket}: {bar} ({count})")

    return 0


def command_export(args):
    """Export session data."""
    from storage.session_store import SessionStore

    console = Console()

    store = SessionStore(args.session)
    session = store.load_session()

    if not session:
        console.print(f"[red]Session not found: {args.session}[/red]")
        return 1

    orchestrator = GradingSessionOrchestrator(session_id=args.session)
    exported = orchestrator.export_data(args.format)

    console.print(f"[green]Exported session {args.session}:[/green]")
    for fmt, path in exported.items():
        console.print(f"  {fmt.upper()}: {path}")

    return 0


def command_list(args):
    """List all sessions."""
    from storage.file_store import SessionIndex
    from storage.session_store import SessionStore

    console = Console()

    index = SessionIndex()
    sessions = index.list_sessions()

    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        return 0

    table = Table(title="Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Created", style="green")
    table.add_column("Status", style="yellow")

    for session_id in sessions:
        session_store = SessionStore(session_id)
        session = session_store.load_session()

        if session:
            table.add_row(
                session_id,
                str(session.created_at)[:19],
                session.status
            )

    console.print(table)
    return 0


def command_api(args):
    """Start the API server."""
    import uvicorn

    from api.app import create_app

    console = Console()

    app = create_app()

    console.print(f"[bold green]Starting API server[/bold green]")
    console.print(f"Host: {args.host}")
    console.print(f"Port: {args.port}")
    console.print(f"Docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers
    )

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Correction System - Intelligent grading of student work",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s correct copies/*.pdf
  %(prog)s correct copies/*.pdf --auto
  %(prog)s correct copies.pdf --pages-per-student 2
  %(prog)s correct copies/*.pdf --output ./my_grades
  %(prog)s status abc123
  %(prog)s analytics abc123
  %(prog)s export abc123 --format json,csv
  %(prog)s list
  %(prog)s api --port 8000

Note on --auto:
  En mode automatique (--auto), aucune interaction utilisateur n'est requise.
  - Le barème est détecté automatiquement pendant la correction
  - En cas de désaccord entre les 2 IA: la moyenne est automatiquement appliquée
  Sans --auto: le programme sollicite l'utilisateur pour arbitrer les désaccords entre IA.

Note on --single:
  Par défaut, si COMPARISON_MODE=true dans .env, le système utilise 2 LLM en parallèle.
  Utilisez --single pour forcer l'utilisation d'un seul LLM:
  - Plus rapide (une seule API call par question)
  - Moins coûteux (50% d'API calls en moins)
  - Pas de vérification croisée en cas d'erreur

Note on --pages-per-student (--pps):
  Active le mode lecture individuelle où le PDF est pré-découpé par copie élève.
  Chaque LLM analyse une copie à la fois (pas de lecture d'ensemble).
  Avantages:
  - Focus LLM concentré sur une seule copie
  - Aucune contamination entre copies
  - Pas d'appel IA pour détecter les élèves (découpage par pages fixes)
  Exemple: --pages-per-student 2 pour un PDF de 8 pages → 4 copies de 2 pages

Note on --second-reading:
  Active la deuxième lecture pour améliorer la qualité de correction.
  - Mode Single LLM (--single): 2 passes - le LLM reçoit ses propres résultats
    et peut les ajuster (2 appels API au lieu d'1)
  - Mode Dual LLM (défaut): Ajoute une instruction de relecture dans le prompt
    initial, demandant au LLM de vérifier sa correction dans le même appel
  Par défaut: OFF (pas de deuxième lecture)
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Correct command
    correct_parser = subparsers.add_parser("correct", help="Grade student copies")
    correct_parser.add_argument(
        "pdfs",
        nargs="+",
        help="PDF files or directories to process"
    )
    correct_parser.add_argument(
        "--auto",
        action="store_true",
        help="Mode automatique sans interaction: utilise le barème détecté et moyenne les notes en cas de désaccord entre IA"
    )
    correct_parser.add_argument(
        "--export",
        default="json,csv",
        help="Export formats (json,csv,analytics)"
    )
    correct_parser.add_argument(
        "--annotate",
        action="store_true",
        help="Generate annotated PDFs"
    )
    correct_parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory"
    )
    correct_parser.add_argument(
        "--subject",
        help="Subject/domain for grading"
    )
    correct_parser.add_argument(
        "--single",
        action="store_true",
        help="Utiliser un seul LLM au lieu du mode comparaison (plus rapide, moins coûteux)"
    )
    correct_parser.add_argument(
        "--skip-reading",
        action="store_true",
        help="Ignorer le consensus de lecture (les LLM notent directement sans valider ce qu'ils lisent)"
    )
    correct_parser.add_argument(
        "--pages-per-student", "--pps",
        type=int,
        help="Nombre de pages par élève (active le mode lecture individuelle - PDF pré-découpé)"
    )
    correct_parser.add_argument(
        "--second-reading",
        action="store_true",
        help="Active la deuxième lecture: en mode Single LLM, 2 passes (2 appels API); en mode Dual LLM, ajoute instruction de relecture dans le prompt"
    )
    correct_parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL_COPIES,
        help=f"Nombre de copies traitées en parallèle (défaut: {DEFAULT_PARALLEL_COPIES}). Accélère le traitement en mode individuel (Single ou Dual LLM)"
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show session status")
    status_parser.add_argument("session", help="Session ID")

    # Analytics command
    analytics_parser = subparsers.add_parser("analytics", help="Show session analytics")
    analytics_parser.add_argument("session", help="Session ID")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export session data")
    export_parser.add_argument("session", help="Session ID")
    export_parser.add_argument(
        "--format",
        default="json,csv",
        help="Export formats (json,csv,analytics)"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List all sessions")

    # API command
    api_parser = subparsers.add_parser("api", help="Start API server")
    api_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to"
    )
    api_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    api_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Execute command
    if args.command == "correct":
        return asyncio.run(command_correct(args))
    elif args.command == "status":
        return command_status(args)
    elif args.command == "analytics":
        return command_analytics(args)
    elif args.command == "export":
        return command_export(args)
    elif args.command == "list":
        return command_list(args)
    elif args.command == "api":
        return command_api(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
