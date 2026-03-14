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
from typing import Dict, List

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
from interaction.cli_correct import (
    ProgressHandler,
    create_disagreement_callback,
    create_name_disagreement_callback,
    create_reading_disagreement_callback
)


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
        if not args.pages_per_copy:
            cli.show_error(
                "Mode 'individual' nécessite --pages-per-copy (--ppc).\n"
                "  --ppc N : Découpe mécanique en copies de N pages"
            )
            return 1

    # In batch mode, --parallel has no effect (all copies in one call)
    if args.grading_method == "batch":
        args.parallel = 1  # No parallelism needed in batch mode

    # Initialize workflow state (replaces mutable dicts)
    state = CorrectionState(
        language='fr',
        auto_mode=args.auto_confirm,
        phase=WorkflowPhase.DETECTION
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
        use_chat_continuation=args.chat_continuation,
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

    # Initialize debug capture if --debug flag is set
    if args.debug:
        from utils.debug_capture import init_debug
        debug_dir = Path(args.output) / orchestrator.session_id / "debug"
        init_debug(debug_dir)
        cli.console.print("[yellow]🐛 Debug mode enabled - capturing prompts to debug/debug_log.json[/yellow]")

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
            'detect': args.detect,
            'skip_reading': args.skip_reading,
            'chat_continuation': args.chat_continuation
        }
    )

    # ============================================================
    # Token tracking setup (before any LLM calls)
    # ============================================================
    _prev_tokens = {'prompt': 0, 'completion': 0, 'cached': 0}
    _current_sub_phase = WorkflowPhase.GRADING  # Track sub-phase within grading (verification, ultimatum)
    _token_debug_log = []  # Debug log for token tracking

    def record_phase_tokens(current_phase: WorkflowPhase, event_name: str = ""):
        """Record token usage for the completed phase (delta)."""
        nonlocal state, _prev_tokens
        if hasattr(orchestrator.ai, 'get_token_usage'):
            usage = orchestrator.ai.get_token_usage()
            current_prompt = usage.get('prompt_tokens', 0)
            current_completion = usage.get('completion_tokens', 0)
            current_cached = usage.get('cached_tokens', 0)

            # Calculate delta from previous phase
            delta_prompt = current_prompt - _prev_tokens['prompt']
            delta_completion = current_completion - _prev_tokens['completion']
            delta_cached = current_cached - _prev_tokens['cached']
            delta_total = delta_prompt + delta_completion

            # Debug log
            _token_debug_log.append({
                'event': event_name,
                'phase': current_phase.value,
                'delta_prompt': delta_prompt,
                'delta_completion': delta_completion,
                'delta_cached': delta_cached,
                'total_prompt': current_prompt,
                'total_completion': current_completion,
                'total_cached': current_cached
            })

            # Update state with delta (always record, even if 0, to track all phases)
            state = state.with_token_usage(
                phase=current_phase,
                prompt_tokens=delta_prompt,
                completion_tokens=delta_completion,
                cached_tokens=delta_cached
            )

            # Store current for next delta calculation
            _prev_tokens = {'prompt': current_prompt, 'completion': current_completion, 'cached': current_cached}

    # ============================================================
    # Phase 1: Détection (Initialisation + Diagnostic)
    # - Chargement du PDF
    # - Découpe si --pages-per-copy
    # - Détection du barème (diagnostic)
    # - Confirmation utilisateur du barème
    # ============================================================
    state = state.with_phase(WorkflowPhase.DETECTION)

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

    # ============================================================
    # Phase 1.5: Diagnostic du Barème (Détection + Confirmation)
    # ============================================================
    detected_scale = analysis.get('scale', {})

    # If no scale detected yet, try to detect it
    if not detected_scale and pdf_paths and args.detect:
        cli.console.print(f"\n[bold cyan]🔍 Détection de la structure...[/bold cyan]")

        try:
            from analysis.detection import Detector
            detector = Detector(
                user_id=user_id if hasattr(args, 'user_id') else 'default',
                session_id=orchestrator.session_id,
                language=language,
                provider=orchestrator.ai  # Use orchestrator's provider for token tracking
            )

            # Detect first PDF for barème detection
            detection_result = detector.detect(pdf_paths[0])
            if detection_result:
                # Save detection for reuse by batch_grader
                orchestrator.store.save_detection(detection_result)
                # Display exam name if detected
                if detection_result.exam_name:
                    cli.console.print(f"[bold cyan]📝 Examen: {detection_result.exam_name}[/bold cyan]")

                # Display document structure info
                cli.console.print(f"\n[bold cyan]📋 Structure détectée:[/bold cyan]")

                # Document type
                doc_type_map = {
                    "student_copies": "Copies d'élèves",
                    "subject_only": "Sujet uniquement",
                    "random_document": "Document non reconnu",
                    "unclear": "Non déterminé"
                }
                doc_type = str(detection_result.document_type).replace("DocumentType.", "")
                doc_type_fr = doc_type_map.get(doc_type, doc_type)
                cli.console.print(f"  Type: [bold]{doc_type_fr}[/bold] (conf: {detection_result.confidence_document_type:.0%})")

                # Structure
                struct_map = {
                    "one_pdf_one_student": "1 élève par PDF",
                    "one_pdf_all_students": "Tous les élèves dans 1 PDF",
                    "ambiguous": "Structure ambiguë"
                }
                struct = str(detection_result.structure).replace("PDFStructure.", "")
                struct_fr = struct_map.get(struct, struct)
                cli.console.print(f"  Structure: [bold]{struct_fr}[/bold]")

                # Pages info
                cli.console.print(f"  Pages totales: [bold]{detection_result.page_count}[/bold]")

                # Pages per student (if consistent)
                if detection_result.consistent_pages_per_student and detection_result.pages_per_student:
                    cli.console.print(f"  Pages/élève: [bold]{detection_result.pages_per_student}[/bold] [green](cohérent)[/green]")

                    # Calculate expected students
                    subject_pages = detection_result.subject_page_count or 0
                    student_pages = detection_result.page_count - subject_pages
                    expected_students = student_pages // detection_result.pages_per_student
                    cli.console.print(f"  Élèves estimés: [bold]{expected_students}[/bold] ({student_pages} pages / {detection_result.pages_per_student} pp)")
                elif detection_result.pages_per_student:
                    cli.console.print(f"  Pages/élève: [bold]{detection_result.pages_per_student}[/bold] [yellow](variable)[/yellow]")

                # Subject integration
                if detection_result.subject_page_count and detection_result.subject_page_count > 0:
                    cli.console.print(f"  Pages sujet: [bold]{detection_result.subject_page_count}[/bold]")

                # Students detected directly
                if detection_result.students:
                    cli.console.print(f"  Élèves détectés par LLM: [bold]{len(detection_result.students)}[/bold]")

                # Grading scale
                if detection_result.grading_scale:
                    detected_scale = detection_result.grading_scale
                    cli.console.print(f"[green]✓ Barème détecté automatiquement[/green]")
        except Exception as e:
            cli.console.print(f"[dim]Détection automatique non disponible: {e}[/dim]")
    elif not detected_scale and pdf_paths and not args.detect:
        cli.console.print(f"[dim]Détection automatique désactivée. Saisie manuelle du barème requise.[/dim]")

    # Display detected scale and ask for confirmation
    if detected_scale:
        cli.console.print(f"\n[bold cyan]📊 Barème détecté:[/bold cyan]")
        for qid in sorted(detected_scale.keys(), key=natural_sort_key):
            pts = detected_scale[qid]
            cli.console.print(f"  {qid}: [bold]{pts}[/bold] point(s)")

    # Get user confirmation or modification
    if not args.auto_confirm:
        from rich.prompt import Prompt, Confirm

        if detected_scale:
            cli.console.print(f"\n[yellow]Confirmer ce barème ?[/yellow]")
            confirm = Confirm.ask("Barème correct", default=True)

            if not confirm:
                # Allow user to modify
                cli.console.print("[dim]Modifiez les valeurs (Entrée pour garder la valeur actuelle)[/dim]")
                modified_scale = {}
                for qid in sorted(detected_scale.keys(), key=natural_sort_key):
                    current = detected_scale[qid]
                    value = Prompt.ask(
                        f"  {qid}",
                        default=str(current)
                    )
                    try:
                        modified_scale[qid] = float(value.replace(',', '.'))
                    except ValueError:
                        modified_scale[qid] = current
                detected_scale = modified_scale
        else:
            # No scale detected - ask user to input
            cli.console.print(f"\n[bold yellow]⚠ Barème non détecté automatiquement[/bold yellow]")
            cli.console.print("[dim]Entrez le barème manuellement (laissez vide pour 1 point)[/dim]")

            questions_to_ask = sorted(analysis.get('questions', {}).keys(), key=natural_sort_key) if analysis.get('questions') else ['Q1', 'Q2', 'Q3']
            detected_scale = {}
            for qid in questions_to_ask:
                value = Prompt.ask(
                    f"  {qid} - Points max",
                    default="1"
                )
                try:
                    detected_scale[qid] = float(value.replace(',', '.'))
                except ValueError:
                    detected_scale[qid] = 1.0

            # Ask if more questions
            while True:
                more = Prompt.ask("Ajouter une autre question (ex: Q4)", default="")
                if not more:
                    break
                try:
                    value = Prompt.ask(f"  {more} - Points max", default="1")
                    detected_scale[more] = float(value.replace(',', '.'))
                except ValueError:
                    detected_scale[more] = 1.0
    else:
        # Auto mode: use detected scale or default to 1.0
        if not detected_scale and analysis.get('questions'):
            detected_scale = {qid: 1.0 for qid in analysis['questions'].keys()}
        elif not detected_scale:
            detected_scale = {}

    # Freeze the scale
    cli.console.print(f"\n[green]📌 Barème figé:[/green] {', '.join([f'{qid}: {pts}pts' for qid, pts in sorted(detected_scale.items(), key=lambda x: natural_sort_key(x[0]))])}")
    orchestrator.confirm_scale(detected_scale)

    # Record DETECTION phase tokens (PDF loading, structure detection, barème detection)
    record_phase_tokens(WorkflowPhase.DETECTION, "detection_complete")

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
    progress_handler = ProgressHandler(cli, state, record_phase_tokens)
    
    # Need to update the _current_sub_phase to match what progress_handler tracked 
    # to avoid syntax errors with missing variables down the line
    
    # Run grading with progress updates

    # Run grading with progress updates
    try:
        graded = await orchestrator.grade_all(progress_callback=progress_handler)
    except Exception as e:
        # Handle specific errors with user-friendly messages
        from core.exceptions import StudentNameMismatchError, DualLLMFailureError, OutputTruncatedError
        if isinstance(e, OutputTruncatedError):
            console.print(f"\n[bold red]✗ Réponse tronquée: le modèle n'a pas pu générer la totalité de la réponse.[/bold red]")
            console.print("[yellow]Le nombre de copies est trop élevé pour un seul appel.[/yellow]")
            console.print("[dim]Conseil: Utilisez --pages-per-copy N pour découper le PDF en copies individuelles.[/dim]")
            return 1
        if isinstance(e, StudentNameMismatchError):
            console.print(f"\n[red]{e.message}[/red]")
            console.print("\n[yellow]Arrêt de la correction. Résolvez le problème avec une des options suivantes:[/yellow]")
            console.print("  1. [cyan]--pages-per-copy N[/cyan]      Découpe mécanique du PDF en copies de N pages")
            console.print("  2. [cyan]--auto-confirm[/cyan]          Continue malgré le problème (non recommandé)")
            return 1
        if isinstance(e, DualLLMFailureError):
            console.print(f"\n[bold red]✗ Erreur: {e.message}[/bold red]")
            console.print("\n[yellow]Le mode double LLM nécessite que les deux fournisseurs soient opérationnels.[/yellow]")
            console.print("[dim]Conseil: Vérifiez vos clés API dans le fichier .env ou utilisez un mode single LLM.[/dim]")
            return 1
        raise

    # Record any remaining tokens for the current sub-phase
    record_phase_tokens(progress_handler.get_current_sub_phase(), "grading_complete")

    # Check for name disagreements in dual LLM mode (max_points disagreements no longer exist - barème is frozen)
    if graded and is_comparison_mode:
        name_disagreements = []

        for g in graded:
            audit = g.grading_audit
            if not audit:
                continue

            # Check for name disagreements
            if audit.student_detection:
                sd = audit.student_detection
                # Check if LLMs disagreed on the name
                llm_names = list(sd.llm_results.values())
                if len(llm_names) >= 2 and llm_names[0] != llm_names[1]:
                    llm_ids = list(sd.llm_results.keys())
                    name_disagreements.append({
                        'copy_index': '?',
                        'llm1_name': sd.llm_results.get(llm_ids[0], ''),
                        'llm2_name': sd.llm_results.get(llm_ids[1], ''),
                        'resolved_name': sd.final_name
                    })

        # Display name disagreements
        if name_disagreements:
            console.print(f"\n[bold yellow]⚠ Désaccord sur le nom pour {len(name_disagreements)} copie(s)[/bold yellow]")
            for nd in name_disagreements[:3]:  # Show max 3
                console.print(f"  Copie {nd['copy_index']}: LLM1=\"{nd['llm1_name']}\", LLM2=\"{nd['llm2_name']}\" → résolu à \"{nd['resolved_name']}\"")
            if len(name_disagreements) > 3:
                console.print(f"  ... et {len(name_disagreements) - 3} autre(s)")
            console.print("[dim]Conseil: Utilisez --pages-per-copy pour une meilleure détection des noms[/dim]")

        # (Re)calculate max_score for all graded copies using frozen scale
        total_max = orchestrator.get_total_max_points()
        if total_max > 0:
            for g in graded:
                g.max_score = total_max

    # ============================================================
    # Phase 3: Verification / Ultimatum
    # ============================================================
    state = state.with_phase(WorkflowPhase.VERIFICATION)

    # Cross-verify names and barème if detected during grading
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
        annotation_model = get_settings().annotation_model
        state = state.with_phase(WorkflowPhase.ANNOTATION)
        cli.console.print(f"\n[bold cyan]📄 Annotation des copies...[/bold cyan]")

        if annotation_model:
            cli.console.print(
                f"[dim]Placement intelligent activé via AI_CORRECTION_ANNOTATION_MODEL={annotation_model}[/dim]"
            )
        else:
            cli.console.print(
                "[yellow]⚠ AI_CORRECTION_ANNOTATION_MODEL non configuré: "
                "fallback heuristique pour générer les PDF annotés et overlays[/yellow]"
            )

        from export.annotation_pipeline import AnnotationExportService

        annotation_output_dir = Path(args.output) / orchestrator.session_id
        annotation_service = AnnotationExportService(session=orchestrator.session)

        for i, (copy, graded_copy) in enumerate(zip(orchestrator.session.copies, graded), 1):
            student_name = copy.student_name or f"copie_{i}"

            try:
                cli.console.print(f"  [dim]{student_name}...[/dim]", end="")

                artifact = annotation_service.export_copy_artifacts(
                    copy=copy,
                    graded=graded_copy,
                    output_dir=str(annotation_output_dir),
                    smart_placement=True,
                    language=language,
                    filename_stem=student_name,
                )
                annotated_files.append(artifact.annotated_pdf)
                overlay_files.append(artifact.overlay_pdf)

                cli.console.print(f" [green]✓[/green]")
            except Exception as e:
                cli.console.print(f" [red]✗ {e}[/red]")

        if annotated_files:
            cli.console.print(f"[green]✓ {len(annotated_files)} copie(s) annotée(s)[/green]")
            exports['annotated_pdfs'] = str(annotation_output_dir / "annotated")
        if overlay_files:
            cli.console.print(f"[green]✓ {len(overlay_files)} overlay(s) généré(s)[/green]")
            exports['annotation_overlays'] = str(annotation_output_dir / "overlays")

        # Record ANNOTATION phase tokens
        record_phase_tokens(WorkflowPhase.ANNOTATION)

    # Mark complete
    from core.models import SessionStatus
    orchestrator.session.status = SessionStatus.COMPLETE
    orchestrator._save_sync()

    # Save debug log if debug mode was enabled
    if args.debug:
        from utils.debug_capture import save_debug_log, get_debug_capture
        debug = get_debug_capture()
        if debug:
            debug_path = debug.save()
            cli.console.print(f"[green]🐛 Debug log saved to: {debug_path}[/green]")
            summary = debug.get_summary()
            cli.console.print(f"[dim]   {summary['total_calls']} API calls captured ({summary['vision_calls']} vision, {summary['text_calls']} text)[/dim]")

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
        phase_order = ['detection', 'grading', 'verification', 'ultimatum', 'calibration', 'annotation']
        phase_labels = {
            'detection': 'Détection',
            'grading': 'Correction',
            'verification': 'Vérification',
            'ultimatum': 'Ultimatum',
            'calibration': 'Calibration',
            'annotation': 'Annotation'
        }

        total_cached = 0
        for phase_name in phase_order:
            if phase_name in token_summary['by_phase']:
                usage = token_summary['by_phase'][phase_name]
                label = phase_labels.get(phase_name, phase_name)
                cached = usage.get('cached', 0)
                billable = max(0, usage['total'] - cached)
                total_cached += cached

                if cached > 0:
                    cli.console.print(
                        f"  {label}: {usage['total']:,} bruts "
                        f"({cached:,} cache, {billable:,} facturables)"
                    )
                else:
                    cli.console.print(f"  {label}: {usage['total']:,} facturables")

        # Total
        cli.console.print(f"  [bold]Total brut: {token_summary['total']:,}[/bold] tokens")
        if total_cached > 0:
            cli.console.print(f"  [green]✓ Cache hits: {total_cached:,} tokens[/green]")
        cli.console.print(
            f"  [bold]Total facturable: {token_summary['total_billable']:,}[/bold] tokens"
        )
        cli.console.print(
            f"  (Prompt brut: {token_summary['total_prompt']:,} | "
            f"Prompt facturable: {token_summary['total_billable_prompt']:,} | "
            f"Completion: {token_summary['total_completion']:,})"
        )

        # Show by provider with cost info
        if hasattr(orchestrator.ai, 'get_token_usage'):
            provider_usage = orchestrator.ai.get_token_usage()
            if 'by_provider' in provider_usage:
                cli.console.print(f"\n  [dim]Par LLM:[/dim]")
                for provider_name, usage in provider_usage['by_provider'].items():
                    cached = usage.get('cached_tokens', 0)
                    total = usage.get('total_tokens', 0)
                    billable_total = usage.get('billable_total_tokens', max(0, total - cached))
                    calls = usage.get('calls', 0)
                    if cached > 0:
                        cli.console.print(
                            f"  [{provider_name}] {total:,} bruts "
                            f"({calls} calls, {cached:,} cache, {billable_total:,} facturables)",
                            markup=False
                        )
                    else:
                        cli.console.print(
                            f"  [{provider_name}] {billable_total:,} facturables ({calls} calls)",
                            markup=False
                        )

        # Show estimated cost if available
        if hasattr(orchestrator.ai, 'get_estimated_cost'):
            try:
                cost_info = orchestrator.ai.get_estimated_cost()
                if cost_info and cost_info.get('estimated_cost_usd') is not None:
                    total_cost = cost_info['estimated_cost_usd']
                    cached_savings = cost_info.get('cached_savings_usd', 0) or 0
                    cli.console.print(f"\n  💰 [bold]Coût estimé: ${total_cost:.4f}[/bold]")
                    if cached_savings > 0:
                        cli.console.print(f"     [green]✓ Économie cache: ${cached_savings:.4f}[/green]")
            except Exception:
                pass  # Cost calculation may not be available for all providers

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
    table.add_column("Name", style="cyan", max_width=30)
    table.add_column("ID", style="dim", width=8)
    table.add_column("Created", style="green")
    table.add_column("Status", style="yellow")

    for session_id in sessions:
        session_store = SessionStore(session_id)
        session = session_store.load_session()

        if session:
            # Get session name from policy.subject or use "Unnamed"
            name = session.policy.subject or "Unnamed Session"
            # Truncate name if too long
            if len(name) > 28:
                name = name[:25] + "..."

            # Truncate UUID to first 8 characters for display
            short_id = session_id[:8]

            table.add_row(
                name,
                short_id,
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
        help=f"Nombre de copies traitées en parallèle en mode INDIVIDUAL uniquement (défaut: {DEFAULT_PARALLEL_COPIES}). Ignoré en mode batch."
    )
    correct_parser.add_argument(
        "--batch-verify",
        choices=["per-question", "per-copy", "grouped"],
        default="grouped",
        help="Mode de vérification: 'per-question' (1 appel/désaccord), 'per-copy' (1 appel/copie), 'grouped' (1 appel tout)"
    )
    correct_parser.add_argument(
        "--chat-continuation",
        action="store_true",
        default=True,
        help="Active l'implicit caching pour la vérification et l'ultimatum. Gemini met automatiquement en cache le préfixe commun (prompt + images) pour ~75%% d'économie. Minimum 1024 tokens (Flash) ou 2048 tokens (Pro). (défaut: activé)"
    )
    correct_parser.add_argument(
        "--no-chat-continuation",
        action="store_false",
        dest="chat_continuation",
        help="Désactive l'implicit caching pour la vérification et l'ultimatum."
    )
    correct_parser.add_argument(
        "--detect",
        action="store_true",
        default=True,
        dest="detect",
        help="Active la détection automatique du barème. (défaut: activé)"
    )
    correct_parser.add_argument(
        "--no-detect",
        action="store_false",
        dest="detect",
        help="Désactive la détection automatique du barème."
    )
    correct_parser.add_argument(
        "--debug",
        action="store_true",
        help="Active le mode debug: capture tous les prompts et appels API dans debug/debug_log.json"
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
