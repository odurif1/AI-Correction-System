"""
Main CLI entry point for the AI correction system.

Usage:
    python src/main.py correct dual batch copies.pdf --pages-per-copy 2
    python src/main.py correct single individual copies.pdf --auto-confirm
    python src/main.py correct dual hybrid copies.pdf --pages-per-copy 2
    python src/main.py api --port 8000
    python src/main.py status <session_id>
    python src/main.py export <session_id>
    python src/main.py list

Arguments:
    llm_mode        single ou dual (nombre de LLM utilisés)
    grading_method  individual, batch, ou hybrid

Options:
    --pages-per-copy N  Découpe le PDF en copies de N pages chaque.
                        Si non spécifié, le PDF entier est envoyé au LLM.
    --auto-confirm      Mode automatique sans interaction
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

    # Validate mode/option combinations
    if args.grading_method == "individual":
        if not args.pages_per_copy and not args.auto_detect_structure:
            cli.show_error(
                "Mode 'individual' nécessite soit --pages-per-copy (--ppc) soit --auto-detect-structure.\n"
                "  --ppc N                 : Découpe mécanique en copies de N pages\n"
                "  --auto-detect-structure : Détection AI de la structure (cross-vérification en mode dual)"
            )
            return 1

    # Initialize workflow state (replaces mutable dicts)
    state = CorrectionState(
        language='fr',
        auto_mode=args.auto_confirm,
        phase=WorkflowPhase.INITIALIZATION
    )

    # Jurisprudence: store user decisions to inform future grading
    # (kept as mutable dict for compatibility with orchestrator.ai.set_jurisprudence)
    jurisprudence: Dict = {}

    # Create orchestrator with callbacks (using helper functions)
    orchestrator = GradingSessionOrchestrator(
        pdf_paths,
        disagreement_callback=None,  # Will be set after orchestrator is created
        name_disagreement_callback=None,
        reading_disagreement_callback=None,
        skip_reading_consensus=args.skip_reading,
        force_single_llm=(args.llm_mode == "single"),
        pages_per_copy=args.pages_per_copy,
        second_reading=args.second_reading,
        parallel=args.parallel,
        grading_mode=args.grading_method,
        batch_verify=args.batch_verify,
        auto_detect_structure=args.auto_detect_structure,
        workflow_state=state  # Pass the workflow state with auto_mode
    )

    # Create callbacks with access to orchestrator
    disagreement_callback = create_disagreement_callback(cli, state, orchestrator, jurisprudence)
    name_disagreement_callback = create_name_disagreement_callback(cli, state)
    reading_disagreement_callback = create_reading_disagreement_callback(cli, state)

    # Set callbacks on orchestrator
    orchestrator.disagreement_callback = disagreement_callback
    orchestrator.name_disagreement_callback = name_disagreement_callback
    orchestrator.reading_disagreement_callback = reading_disagreement_callback

    # Determine mode and LLM names for startup display
    is_comparison_mode = hasattr(orchestrator.ai, 'providers')
    if is_comparison_mode:
        mode = "comparison"
        llm_names = [name for name, _ in orchestrator.ai.providers]
        llm1_name = llm_names[0] if len(llm_names) > 0 else None
        llm2_name = llm_names[1] if len(llm_names) > 1 else None
    else:
        mode = "single"
        model_name = getattr(orchestrator.ai, 'model', None) or get_settings().gemini_model
        llm1_name = model_name
        llm2_name = None

    # Show startup banner with configuration
    cli.show_startup(
        pdf_files=pdf_paths,
        mode=mode,
        grading_method=args.grading_method,
        pages_per_copy=args.pages_per_copy,
        language='auto',
        llm1_name=llm1_name,
        llm2_name=llm2_name,
        display_language='fr',
        session_id=orchestrator.session_id,
        options={
            'auto_confirm': args.auto_confirm,
            'batch_verify': args.batch_verify,
            'annotate': args.annotate,
            'parallel': args.parallel,
            'second_reading': args.second_reading,
            'auto_detect_structure': args.auto_detect_structure,
            'skip_reading': args.skip_reading
        }
    )

    # ============================================================
    # Phase 1: Initialisation
    # - Chargement du PDF
    # - Découpe si --pages-per-copy
    # - Pré-vérification optionnelle (ex: ordre des copies)
    # ============================================================
    state = state.with_phase(WorkflowPhase.INITIALIZATION)

    try:
        analysis = await orchestrator.analyze_only()
    except Exception as e:
        cli.show_error(f"Analysis failed: {e}")
        return 1

    # Get detected language and update state
    language = analysis.get('language', 'fr')
    state = state.with_language(language)

    # Show analysis result
    copies_count = analysis['copies_count']
    questions_detected_during_grading = analysis.get('questions_detected_during_grading', False)

    if questions_detected_during_grading:
        # Structure will be detected during grading
        if language == 'fr':
            pdf_word = "fichier" if copies_count == 1 else "fichiers"
            cli.console.print(f"[green]✓ {copies_count} {pdf_word} PDF chargé(s) - structure détectée pendant la correction[/green]\n")
        else:
            pdf_word = "file" if copies_count == 1 else "files"
            cli.console.print(f"[green]✓ {copies_count} PDF {pdf_word} loaded - structure detected during grading[/green]\n")
    else:
        # Structure was pre-detected
        if language == 'fr':
            copy_word = "copie" if copies_count == 1 else "copies"
            cli.console.print(f"[green]✓ {copies_count} {copy_word} détectée(s)[/green]\n")
        else:
            copy_word = "copy" if copies_count == 1 else "copies"
            cli.console.print(f"[green]✓ {copies_count} {copy_word} detected[/green]\n")

    # Check if questions were detected or will be detected during grading
    if not analysis['questions'] and not questions_detected_during_grading:
        cli.show_error("Aucune question détectée. L'analyse a peut-être échoué.")
        return 1

    # Scale will be empty - it will be detected during grading
    # If not detected and not auto mode, user will be prompted after grading
    scale = {}
    orchestrator.confirm_scale(scale)

    # Helper function to prompt user for missing scale
    async def prompt_for_missing_scale(question_ids: List[str]) -> Dict[str, float]:
        """Prompt user for max points of questions without scale."""
        if args.auto_confirm:
            # Auto mode: default to 1.0 for all
            return {qid: 1.0 for qid in question_ids}

        cli.console.print(f"\n[bold yellow]Barème non détecté pour {len(question_ids)} question(s)[/bold yellow]")
        new_scale = {}
        for qid in question_ids:
            if orchestrator.get_max_points(qid) <= 0:
                from rich.prompt import Prompt
                value = Prompt.ask(
                    f"  {qid} - Points max",
                    default="1"
                )
                try:
                    new_scale[qid] = float(value.replace(',', '.'))
                except ValueError:
                    new_scale[qid] = 1.0
        return new_scale

    # Helper to record token usage for current phase (delta from previous)
    _prev_tokens = {'prompt': 0, 'completion': 0}
    _current_sub_phase = WorkflowPhase.GRADING  # Track sub-phase within grading (verification, ultimatum)
    _token_debug_log = []  # Debug log for token tracking

    def record_phase_tokens(current_phase: WorkflowPhase, event_name: str = ""):
        """Record token usage for the completed phase (delta)."""
        nonlocal state, _prev_tokens
        if hasattr(orchestrator.ai, 'get_token_usage'):
            usage = orchestrator.ai.get_token_usage()
            current_prompt = usage.get('prompt_tokens', 0)
            current_completion = usage.get('completion_tokens', 0)

            # Calculate delta from previous phase
            delta_prompt = current_prompt - _prev_tokens['prompt']
            delta_completion = current_completion - _prev_tokens['completion']
            delta_total = delta_prompt + delta_completion

            # Debug log
            _token_debug_log.append({
                'event': event_name,
                'phase': current_phase.value,
                'delta_prompt': delta_prompt,
                'delta_completion': delta_completion,
                'total_prompt': current_prompt,
                'total_completion': current_completion
            })

            # Update state with delta
            if delta_prompt > 0 or delta_completion > 0:
                state = state.with_token_usage(
                    phase=current_phase,
                    prompt_tokens=delta_prompt,
                    completion_tokens=delta_completion
                )

            # Store current for next delta calculation
            _prev_tokens = {'prompt': current_prompt, 'completion': current_completion}

    # ============================================================
    # Phase 2: Grading
    # ============================================================
    state = state.with_phase(WorkflowPhase.GRADING)
    console = cli.console
    is_comparison_mode = hasattr(orchestrator.ai, 'providers')

    # Track LLM completion status for ordered display
    llm_status = {'results': {}, 'total': 2}

    # Track token usage per copy
    prev_tokens = {'total': 0}

    # Collect question results for sorted display at copy_done
    current_copy_questions = {}

    # Progress callback for real-time display
    async def progress_callback(event_type: str, data: dict):
        nonlocal prev_tokens, _current_sub_phase

        if event_type == 'copy_start':
            copy_idx = data['copy_index']
            total = data['total_copies']
            name = data.get('student_name') or '???'
            llm_status['results'] = {}
            current_copy_questions.clear()  # Reset for new copy
            # Update live display instead of printing (handled by LiveProgressDisplay)

        elif event_type == 'question_start':
            q_id = data['question_id']
            q_idx = data['question_index']
            total_q = data['total_questions']
            llm_status['results'] = {}  # Reset for new question

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
            copy_idx = data.get('copy_index', '')
            student_name = data.get('student_name', '???')
            score = data['total_score']
            max_s = data['max_score']
            pct = (score / max_s * 100) if max_s > 0 else 0
            conf = data.get('confidence', 0.5) or 0.5
            final_questions = data.get('final_questions', {})
            feedback = data.get('feedback', '')

            # In batch mode, use final_questions from the event
            # In individual mode, use current_copy_questions populated by question_done
            questions_to_display = final_questions if final_questions else current_copy_questions

            # Header for this copy
            console.print(f"\n  [bold cyan]── Copie {copy_idx}: {student_name} ──[/bold cyan]")

            # Display questions in compact format
            grades_str = "  "
            for qid in sorted(questions_to_display.keys(), key=natural_sort_key):
                q = questions_to_display[qid]
                grade = q.get('grade', 0)
                max_pts = q.get('max_points', q.get('max_pts', 1))
                grades_str += f"{qid}: [bold]{grade:.0f}/{max_pts:.0f}[/bold]  "
            console.print(grades_str)

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

            console.print(f"  [bold {color}]Total: {score:.1f}/{max_s} ({pct:.0f}%)[/bold {color}] [dim]conf: {conf:.0%}[/dim]{tokens_str}")

            # Display feedback/appreciation
            if feedback:
                console.print(f"  [italic dim]{feedback[:150]}{'...' if len(feedback) > 150 else ''}[/italic dim]")

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
            # Skip verbose output - live display shows progress
            pass

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
            # Switch to VERIFICATION phase for token tracking
            record_phase_tokens(_current_sub_phase, event_type)  # Record any pending GRADING tokens
            _current_sub_phase = WorkflowPhase.VERIFICATION

            qid = data.get('question_id')
            reason = data.get('reason', '')
            console.print(f"  [dim]🔄 Vérification {qid}...[/dim]")

        elif event_type == 'batch_comparison_ready':
            # Display structured comparison results for dual LLM batch mode
            providers = data.get('providers', ['LLM1', 'LLM2'])
            copies = data.get('copies', [])

            # Store for later display after verification/ultimatum
            llm_status['comparison_data'] = data

            # Helper to get short provider name
            def get_short_name(provider_str: str, idx: int) -> str:
                # Extract model name from "LLM1: model-name" format
                model = provider_str.replace('LLM1: ', '').replace('LLM2: ', '')
                # Create short readable name
                if 'gemini' in model.lower():
                    if 'flash' in model.lower():
                        return 'Gemini Flash'
                    elif 'pro' in model.lower():
                        return 'Gemini Pro'
                    return 'Gemini'
                elif 'gpt' in model.lower() or 'openai' in model.lower():
                    if '4o' in model.lower():
                        return 'GPT-4o'
                    elif '4' in model:
                        return 'GPT-4'
                    return 'GPT'
                elif 'claude' in model.lower():
                    if 'opus' in model.lower():
                        return 'Claude Opus'
                    elif 'sonnet' in model.lower():
                        return 'Claude Sonnet'
                    return 'Claude'
                # Fallback to LLM1/LLM2
                return f'LLM{idx+1}'

            p1_short = get_short_name(providers[0], 0)
            p2_short = get_short_name(providers[1], 1) if len(providers) > 1 else ''

            for copy_info in copies:
                copy_idx = copy_info.get('copy_index', '')
                student_name = copy_info.get('student_name') or '???'
                questions = copy_info.get('questions', {})

                # Header for this copy
                console.print(f"\n[bold cyan]═══ Copie {copy_idx}: {student_name} ═══[/bold cyan]")

                # Table header with cleaner format
                console.print(f"   {'Q':<4} │ {p1_short:<12} │ {p2_short:<12} │ {'Status'}")
                console.print(f"   {'─'*4}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*12}")

                # Sort questions naturally
                for qid in sorted(questions.keys(), key=natural_sort_key):
                    q = questions[qid]
                    llm1_grade = q.get('llm1_grade')
                    llm1_max = q.get('llm1_max_points', 1)
                    llm2_grade = q.get('llm2_grade')
                    llm2_max = q.get('llm2_max_points', 1)
                    agreement = q.get('agreement', True)

                    # Format grades
                    if llm1_grade is not None:
                        g1_str = f"{llm1_grade:.1f}/{llm1_max:.0f}"
                    else:
                        g1_str = "erreur"

                    if llm2_grade is not None:
                        g2_str = f"{llm2_grade:.1f}/{llm2_max:.0f}"
                    else:
                        g2_str = "erreur"

                    # Status
                    if agreement:
                        status = "[green]✓ accord[/green]"
                    else:
                        status = "[yellow]⚠ désaccord[/yellow]"

                    console.print(f"   {qid:<4} │ {g1_str:<12} │ {g2_str:<12} │ {status}")

                console.print("")

        elif event_type == 'batch_verification_start':
            # Switch to VERIFICATION phase for batch verification
            record_phase_tokens(_current_sub_phase, event_type)  # Record any pending GRADING tokens
            _current_sub_phase = WorkflowPhase.VERIFICATION
            console.print(f"\n  [dim]🔄 Vérification des désaccords...[/dim]")

        elif event_type == 'batch_verification_done':
            # Record VERIFICATION tokens and switch back to GRADING
            record_phase_tokens(WorkflowPhase.VERIFICATION, event_type)
            _current_sub_phase = WorkflowPhase.GRADING

            # Display verification results
            # Show which cases are truly resolved vs going to ultimatum
            questions = data.get('questions', [])
            if questions:
                # Cases that are truly resolved (not going to ultimatum)
                resolved_cases = [q for q in questions if not q.get('goes_to_ultimatum', False)]
                # Cases going to ultimatum
                ultimatum_cases = [q for q in questions if q.get('goes_to_ultimatum', False)]

                if resolved_cases:
                    console.print(f"\n  [bold]📋 Vérification - Consensus atteint:[/bold]")
                    for q in resolved_cases:
                        qid = q.get('question_id')
                        copy_idx = q.get('copy_index', '')
                        final = q.get('final_grade', 0)
                        # Show copy index if available
                        if copy_idx:
                            console.print(f"    Copie {copy_idx} {qid}: [green]{final:.1f}[/green] (consensus)")
                        else:
                            console.print(f"    {qid}: [green]{final:.1f}[/green] (consensus)")

                if ultimatum_cases:
                    console.print(f"\n  [dim]📋 Vérification - Cas nécessitant l'ultimatum ({len(ultimatum_cases)}):[/dim]")
                    for q in ultimatum_cases:
                        qid = q.get('question_id')
                        copy_idx = q.get('copy_index', '')
                        reasons = q.get('ultimatum_reasons', [])
                        reason_str = f" ({', '.join(reasons)})" if reasons else ""
                        # Show copy index if available
                        if copy_idx:
                            console.print(f"    [dim]Copie {copy_idx} {qid} → ultimatum{reason_str}[/dim]")
                        else:
                            console.print(f"    [dim]{qid} → ultimatum{reason_str}[/dim]")

        elif event_type == 'batch_ultimatum_start':
            # Switch to ULTIMATUM phase for ultimatum tokens
            record_phase_tokens(_current_sub_phase, event_type)  # Record any pending tokens
            _current_sub_phase = WorkflowPhase.ULTIMATUM
            console.print(f"\n  [dim]⚖️ Ultimatum (désaccords persistants)...[/dim]")

        elif event_type == 'batch_ultimatum_done':
            # Record ultimatum tokens and switch back to GRADING
            record_phase_tokens(WorkflowPhase.ULTIMATUM, event_type)
            _current_sub_phase = WorkflowPhase.GRADING

            # Display ultimatum results
            questions = data.get('questions', [])
            if questions:
                console.print(f"\n  [bold]📋 Résultats de l'ultimatum:[/bold]")
                for q in questions:
                    qid = q.get('question_id')
                    copy_idx = q.get('copy_index', '')
                    final = q.get('final_grade', 0)
                    method = q.get('method', 'unknown')
                    # Show copy index if available
                    if copy_idx:
                        console.print(f"    Copie {copy_idx} {qid}: [dark_orange]{final:.1f}[/dark_orange] ({method})")
                    else:
                        console.print(f"    {qid}: [dark_orange]{final:.1f}[/dark_orange] ({method})")

        elif event_type == 'ultimatum_parse_warning':
            # Alert user about parsing failure
            warning = data.get('warning', 'Ultimatum parsing failed')
            console.print(f"\n  [bold red]⚠️ {warning}[/bold red]")
            console.print(f"  [dim]Les décisions finales peuvent être imprécises[/dim]")

        elif event_type == 'verification_done':
            # Record VERIFICATION tokens and switch back to GRADING
            record_phase_tokens(WorkflowPhase.VERIFICATION, event_type)
            _current_sub_phase = WorkflowPhase.GRADING

        elif event_type == 'ultimatum_start':
            # Switch to ULTIMATUM phase for ultimatum tokens
            record_phase_tokens(_current_sub_phase, event_type)  # Record any pending tokens
            _current_sub_phase = WorkflowPhase.ULTIMATUM

        elif event_type == 'ultimatum_done':
            # Record ultimatum tokens and switch back to GRADING
            record_phase_tokens(WorkflowPhase.ULTIMATUM, event_type)
            _current_sub_phase = WorkflowPhase.GRADING

    # Run grading with progress updates
    try:
        graded = await orchestrator.grade_all(progress_callback=progress_callback)
    except Exception as e:
        # Handle StudentNameMismatchError specifically
        from core.exceptions import StudentNameMismatchError
        if isinstance(e, StudentNameMismatchError):
            console.print(f"\n[red]{e.message}[/red]")
            console.print("\n[yellow]Arrêt de la correction. Résolvez le problème avec une des options suivantes:[/yellow]")
            console.print("  1. [cyan]--pages-per-copy N[/cyan]      Découpe mécanique du PDF en copies de N pages")
            console.print("  2. [cyan]--auto-detect-structure[/cyan] Pré-analyse la structure avant correction")
            console.print("  3. [cyan]--auto-confirm[/cyan]          Continue malgré le problème (non recommandé)")
            return 1
        raise

    # Record any remaining tokens for the current sub-phase
    record_phase_tokens(_current_sub_phase, "grading_complete")

    # Check if scale was detected, prompt user if not
    if graded:
        # Check for max_points disagreements in dual LLM mode
        if is_comparison_mode:
            max_points_disagreements = []
            name_disagreements = []

            for g in graded:
                llm_comp = g.llm_comparison or {}
                llm_comparison = llm_comp.get('llm_comparison', {})

                # Check for max_points disagreements
                for qid, qdata in llm_comparison.items():
                    if isinstance(qdata, dict) and 'max_points_disagreement' in qdata:
                        mpd = qdata['max_points_disagreement']
                        max_points_disagreements.append({
                            'copy_index': llm_comp.get('student_detection', {}).get('copy_index', '?'),
                            'question_id': qid,
                            'llm1_max_points': mpd.get('llm1_max_points'),
                            'llm2_max_points': mpd.get('llm2_max_points'),
                            'resolved': mpd.get('resolved_max_points'),
                            'persisted_after_ultimatum': mpd.get('persisted_after_ultimatum', False)
                        })

                # Check for name disagreements
                student_detection = llm_comp.get('student_detection', {})
                if 'name_disagreement' in student_detection:
                    nd = student_detection['name_disagreement']
                    name_disagreements.append({
                        'copy_index': student_detection.get('copy_index', '?'),
                        'llm1_name': nd.get('llm1_name'),
                        'llm2_name': nd.get('llm2_name'),
                        'resolved_name': nd.get('resolved_name')
                    })

            # Display max_points disagreements
            if max_points_disagreements:
                console.print(f"\n[bold yellow]⚠ Désaccord sur le barème pour {len(max_points_disagreements)} question(s)[/bold yellow]")
                for mpd in max_points_disagreements[:3]:  # Show max 3
                    persisted = " (non résolu après ultimatum)" if mpd['persisted_after_ultimatum'] else ""
                    console.print(f"  Copie {mpd['copy_index']}, {mpd['question_id']}: LLM1={mpd['llm1_max_points']}pts, LLM2={mpd['llm2_max_points']}pts → résolu à {mpd['resolved']}pts{persisted}")
                if len(max_points_disagreements) > 3:
                    console.print(f"  ... et {len(max_points_disagreements) - 3} autre(s)")
                console.print("[dim]Conseil: Vérifiez le barème dans le sujet et utilisez --bareme pour le spécifier[/dim]")

            # Display name disagreements
            if name_disagreements:
                console.print(f"\n[bold yellow]⚠ Désaccord sur le nom pour {len(name_disagreements)} copie(s)[/bold yellow]")
                for nd in name_disagreements[:3]:  # Show max 3
                    console.print(f"  Copie {nd['copy_index']}: LLM1=\"{nd['llm1_name']}\", LLM2=\"{nd['llm2_name']}\" → résolu à \"{nd['resolved_name']}\"")
                if len(name_disagreements) > 3:
                    console.print(f"  ... et {len(name_disagreements) - 3} autre(s)")
                console.print("[dim]Conseil: Utilisez --pages-per-copy ou --auto-detect-structure pour une meilleure détection[/dim]")

        # Get all question IDs from graded copies
        all_questions = set()
        for g in graded:
            all_questions.update(g.grades.keys())

        # Check which questions have no scale (max_points = 0 or not set)
        missing_scale = [qid for qid in all_questions if orchestrator.get_max_points(qid) <= 0]

        if missing_scale:
            cli.console.print(f"\n[bold yellow]⚠ Barème non détecté pour {len(missing_scale)} question(s)[/bold yellow]")
            new_scale = await prompt_for_missing_scale(missing_scale)
            orchestrator.grading_scale.update(new_scale)
            orchestrator._save_sync()

        # (Re)calculate max_score for all graded copies
        total_max = orchestrator.get_total_max_points()
        if total_max > 0:
            for g in graded:
                g.max_score = total_max

    # ============================================================
    # Phase 3: Verification / Ultimatum
    # ============================================================
    state = state.with_phase(WorkflowPhase.VERIFICATION)

    # Cross-verify names and barème if detected during grading (not pre-detected)
    if args.auto_detect_structure:
        cli.console.print("[dim]Structure pré-détectée, vérification noms/barème skipée[/dim]")
    else:
        verification_results = await orchestrator.verify_detected_parameters()
        if verification_results.get('names_disagreed', 0) > 0:
            cli.console.print(f"[yellow]⚠ {verification_results['names_disagreed']} désaccord(s) sur les noms[/yellow]")

    # Review doubts (low confidence grades)
    if not args.auto_confirm:
        doubts = orchestrator.get_doubts(threshold=0.7)
        if doubts:
            decisions = cli.review_doubts(doubts, language=language)
            if decisions:
                await orchestrator.apply_decisions(decisions)
                cli.show_success(f"Applied {len(decisions)} decision(s)")

    # Record VERIFICATION phase tokens
    record_phase_tokens(WorkflowPhase.VERIFICATION)

    # ============================================================
    # Phase 4: Calibration
    # ============================================================
    state = state.with_phase(WorkflowPhase.CALIBRATION)
    await orchestrator._calibration_phase()

    # Record CALIBRATION phase tokens
    record_phase_tokens(WorkflowPhase.CALIBRATION)

    # ============================================================
    # Phase 5: Export
    # ============================================================
    state = state.with_phase(WorkflowPhase.EXPORT)

    exports = await orchestrator.export()

    # ============================================================
    # Phase 6: Annotation (optionnel)
    # ============================================================
    annotated_files = []
    overlay_files = []
    if args.annotate:
        # Check if annotation model is configured
        annotation_model = get_settings().annotation_model
        if not annotation_model:
            cli.console.print(f"\n[yellow]⚠ Annotation skip: AI_CORRECTION_ANNOTATION_MODEL non configuré[/yellow]")
        else:
            state = state.with_phase(WorkflowPhase.ANNOTATION)
            cli.console.print(f"\n[bold cyan]📄 Annotation des copies...[/bold cyan]")

            from export.pdf_annotator import PDFAnnotator

            # Create output directories
            annotated_dir = Path(args.output) / orchestrator.session_id / "annotated"
            overlay_dir = Path(args.output) / orchestrator.session_id / "overlays"
            annotated_dir.mkdir(parents=True, exist_ok=True)
            overlay_dir.mkdir(parents=True, exist_ok=True)

            annotator = PDFAnnotator(session=orchestrator.session)

            for i, (copy, graded_copy) in enumerate(zip(orchestrator.session.copies, graded), 1):
                student_name = copy.student_name or f"copie_{i}"
                safe_name = student_name.replace(" ", "_").replace("/", "-")

                # Annotated copy (full PDF with annotations)
                annotated_path = annotated_dir / f"{safe_name}_annotated.pdf"
                # Overlay (just annotations, transparent background)
                overlay_path = overlay_dir / f"{safe_name}_overlay.pdf"

                try:
                    cli.console.print(f"  [dim]{student_name}...[/dim]", end="")

                    # Generate annotated copy
                    annotator.annotate_copy(
                        copy=copy,
                        graded=graded_copy,
                        output_path=str(annotated_path),
                        smart_placement=True,
                        language=language
                    )
                    annotated_files.append(str(annotated_path))

                    # Generate overlay
                    annotator.create_annotation_overlay(
                        copy=copy,
                        graded=graded_copy,
                        output_path=str(overlay_path),
                        smart_placement=True,
                        language=language
                    )
                    overlay_files.append(str(overlay_path))

                    cli.console.print(f" [green]✓[/green]")
                except Exception as e:
                    cli.console.print(f" [red]✗ {e}[/red]")

            if annotated_files:
                cli.console.print(f"[green]✓ {len(annotated_files)} copie(s) annotée(s)[/green]")
                exports['annotated_pdfs'] = str(annotated_dir)
            if overlay_files:
                cli.console.print(f"[green]✓ {len(overlay_files)} overlay(s) généré(s)[/green]")
                exports['annotation_overlays'] = str(overlay_dir)

            # Record ANNOTATION phase tokens
            record_phase_tokens(WorkflowPhase.ANNOTATION)

    # Mark complete
    from core.models import SessionStatus
    orchestrator.session.status = SessionStatus.COMPLETE
    orchestrator._save_sync()

    # ============================================================
    # Show Summary
    # ============================================================
    state = state.with_phase(WorkflowPhase.COMPLETE)

    # Gather stats for summary
    scores = [g.total_score for g in graded]
    max_scores = [g.max_score for g in graded]
    actual_max = max_scores[0] if max_scores and max_scores[0] > 0 else 20

    # Calculate duration
    duration = None
    if hasattr(orchestrator.session, 'created_at'):
        from datetime import datetime
        if isinstance(orchestrator.session.created_at, datetime):
            duration = (datetime.now() - orchestrator.session.created_at).total_seconds()

    # Build top performers
    top_performers = []
    for i, g in enumerate(sorted(graded, key=lambda x: x.total_score, reverse=True)[:3], 1):
        copy = next((c for c in orchestrator.session.copies if c.id == g.copy_id), None)
        top_performers.append({
            'name': copy.student_name if copy else f"Élève {i}",
            'score': g.total_score,
            'max': g.max_score
        })

    # Determine mode string for summary
    if is_comparison_mode:
        summary_mode = "comparison"  # Will be displayed as "Double LLM" or "Dual LLM"
    else:
        summary_mode = "single"  # Will be displayed as "LLM Simple" or "Single LLM"

    # Final summary panel (without redundant session_id - already shown in config)
    cli.show_summary(
        copies_count=len(orchestrator.session.copies),
        graded_count=len(graded),
        scores=scores,
        duration=duration,
        mode=summary_mode,
        exports=exports,
        top_performers=top_performers,
        language=language
    )

    # Show token usage by phase
    token_summary = state.get_token_summary()
    if token_summary['total'] > 0:
        cli.console.print(f"\n[bold cyan]📊 Token Usage par Phase:[/bold cyan]")

        # Phase order for display
        phase_order = ['grading', 'verification', 'ultimatum', 'calibration', 'annotation']
        phase_labels = {
            'grading': 'Correction',
            'verification': 'Vérification',
            'ultimatum': 'Ultimatum',
            'calibration': 'Calibration',
            'annotation': 'Annotation'
        }

        for phase_name in phase_order:
            if phase_name in token_summary['by_phase']:
                usage = token_summary['by_phase'][phase_name]
                label = phase_labels.get(phase_name, phase_name)
                cli.console.print(f"  {label}: {usage['total']:,} tokens")

        # Total
        cli.console.print(f"  [bold]Total: {token_summary['total']:,}[/bold] tokens")
        cli.console.print(f"  (Prompt: {token_summary['total_prompt']:,} | Completion: {token_summary['total_completion']:,})")

        # Show by provider if available
        if hasattr(orchestrator.ai, 'get_token_usage'):
            provider_usage = orchestrator.ai.get_token_usage()
            if 'by_provider' in provider_usage:
                cli.console.print(f"\n  [dim]Par provider:[/dim]")
                for provider_name, usage in provider_usage['by_provider'].items():
                    # Use markup=False to avoid Rich interpreting brackets
                    cli.console.print(f"  [{provider_name}] {usage.get('total_tokens', 0):,} tokens ({usage.get('calls', 0)} calls)", markup=False)

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
  # Avec découpage par pages (PDF multi-copies)
  %(prog)s correct dual batch copies.pdf --pages-per-copy 2 --auto-confirm
  %(prog)s correct single individual copies.pdf --pages-per-copy 2

  # Sans découpage (PDF envoyé entier au LLM)
  %(prog)s correct dual batch copies.pdf --auto-confirm
  %(prog)s correct single individual copies.pdf

  # Autres commandes
  %(prog)s status abc123
  %(prog)s export abc123 --format json,csv
  %(prog)s list

Arguments:
  llm_mode        single ou dual (nombre de LLM utilisés)
                  - single: un seul LLM, plus rapide et moins coûteux
                  - dual: deux LLM en parallèle, vérification croisée

  grading_method  individual, batch, ou hybrid
                  - individual: chaque copie corrigée séparément
                  - batch: toutes les copies en un seul appel API
                  - hybrid: LLM1=batch, LLM2=individual (dual uniquement)

Note on --pages-per-copy (--ppc):
  Optionnel. Deux modes de fonctionnement:

  1. AVEC --pages-per-copy N:
     - Le PDF est découpé en copies de N pages chacune
     - Exemple: --pages-per-copy 2 pour un PDF de 8 pages → 4 copies
     - Recommandé pour les PDF multi-élèves avec structure fixe

  2. SANS --pages-per-copy:
     - Le PDF entier est envoyé au LLM sans découpage
     - Le LLM détecte automatiquement les copies et les élèves
     - Recommandé pour:
       * PDF pré-découpé (1 fichier = 1 copie élève)
       * Laisser le LLM analyser la structure du document
       * Documents avec structure variable

Note on --auto-confirm:
  En mode automatique, aucune interaction utilisateur n'est requise.
  - Le barème est détecté automatiquement pendant la correction
  - En cas de désaccord entre les 2 IA (dual): la moyenne est appliquée
  - Sans --auto-confirm: le programme sollicite l'utilisateur pour:
    * Arbitrer les désaccords entre IA
    * Confirmer le barème si non détecté

Note on --second-reading:
  Active la deuxième lecture pour améliorer la qualité de correction.
  - Mode Single LLM: 2 passes - le LLM reçoit ses propres résultats
  - Mode Dual LLM: Ajoute une instruction de relecture dans le prompt
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Correct command
    correct_parser = subparsers.add_parser("correct", help="Grade student copies")
    correct_parser.add_argument(
        "llm_mode",
        choices=["single", "dual"],
        help="Mode LLM: 'single' (un seul LLM) ou 'dual' (2 LLM en comparaison)"
    )
    correct_parser.add_argument(
        "grading_method",
        choices=["individual", "batch", "hybrid"],
        help="Méthode de correction: 'individual' (chaque copie séparément), 'batch' (toutes les copies en un appel), 'hybrid' (LLM1=batch, LLM2=individual)"
    )
    correct_parser.add_argument(
        "pdfs",
        nargs="+",
        help="PDF files or directories to process"
    )
    correct_parser.add_argument(
        "--auto-confirm",
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
        help="Générer des PDFs annotés avec le feedback (nécessite un LLM vision configuré)"
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
        "--skip-reading",
        action="store_true",
        help="Ignorer le consensus de lecture (les LLM notent directement sans valider ce qu'ils lisent)"
    )
    correct_parser.add_argument(
        "--pages-per-copy", "--ppc",
        type=int,
        default=None,
        help="Nombre de pages par copie élève. Si non spécifié, le PDF est envoyé entier au LLM qui détecte les copies automatiquement."
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
    correct_parser.add_argument(
        "--batch-verify",
        choices=["per-question", "grouped"],
        default="grouped",
        help="Mode de vérification post-batch pour les désaccords (dual batch uniquement): 'per-question' (un appel par désaccord) ou 'grouped' (un seul appel groupé)"
    )
    correct_parser.add_argument(
        "--auto-detect-structure",
        action="store_true",
        help="Analyse le PDF entier en pré-phase pour détecter la structure (copies, pages, noms, barème) avec les 2 LLMs. Cross-vérification automatique des désaccords."
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
