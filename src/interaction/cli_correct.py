"""
CLI Correction Handlers for the AI Correction System.

Extracted from main.py to decouple presentation logic from orchestration logic.
"""

from typing import Dict
from rich.table import Table

from core.workflow_state import CorrectionState, WorkflowPhase
from utils.sorting import natural_sort_key
from interaction.cli import CLI


def create_workflow_callbacks(cli: CLI, language: str, auto_mode: bool):
    """
    Create workflow callbacks that integrate with CLI.
    """
    class WorkflowCallbacks:
        def __init__(self, on_disagreement, on_name_disagreement, on_reading_disagreement):
            self.on_disagreement = on_disagreement
            self.on_name_disagreement = on_name_disagreement
            self.on_reading_disagreement = on_reading_disagreement

    async def on_disagreement(question_id: str, question_text: str, llm1_name: str, llm1_result: dict, llm2_name: str, llm2_result: dict, max_points: float) -> tuple[float, str]:
        grade1 = llm1_result.get('grade', 0) or 0
        grade2 = llm2_result.get('grade', 0) or 0
        average_grade = (grade1 + grade2) / 2

        if auto_mode:
            cli.console.print(f"    [yellow]⚠ {llm1_name}: {grade1} vs {llm2_name}: {grade2} → moyenne: {average_grade:.2f}[/yellow]")
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

    async def on_reading_disagreement(llm1_result: dict, llm2_result: dict, question_text: str, image_path) -> str:
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


def create_disagreement_callback(cli: CLI, state: CorrectionState, orchestrator, jurisprudence: Dict):
    async def callback(question_id: str, question_text: str, llm1_name: str, llm1_result: dict, llm2_name: str, llm2_result: dict, max_points: float) -> tuple:
        grade1 = llm1_result.get('grade', 0) or 0
        grade2 = llm2_result.get('grade', 0) or 0
        average_grade = (grade1 + grade2) / 2

        if question_id in jurisprudence:
            past = jurisprudence[question_id]
            cli.console.print(f"    [dim]📜 Jurisprudence: décision passée = {past['decision']:.1f}/{max_points}[/dim]")

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
    async def callback(llm1_result: Dict, llm2_result: Dict, question_text: str, image_path) -> str:
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


class ProgressHandler:
    def __init__(self, cli: CLI, state: CorrectionState, record_phase_tokens_cb):
        self.cli = cli
        self.console = cli.console
        self.state = state
        self.record_phase_tokens = record_phase_tokens_cb

        self.llm_status = {'results': {}, 'total': 2}
        self.prev_tokens = {'total': 0}
        self.current_copy_questions = {}
        self.current_sub_phase = WorkflowPhase.GRADING

    async def __call__(self, event_type: str, data: dict):
        if event_type == 'copy_start':
            self.llm_status['results'] = {}
            self.current_copy_questions.clear()

        elif event_type == 'question_start':
            self.llm_status['results'] = {}

        elif event_type == 'llm_parallel_start':
            pass

        elif event_type == 'llm_complete':
            provider_index = data.get('provider_index', 0)
            provider = data.get('provider', '???')
            grade = data.get('grade')
            all_done = data.get('all_completed', False)

            self.llm_status['results'][provider_index] = {'provider': provider, 'grade': grade}

            if all_done and len(self.llm_status['results']) == self.llm_status['total']:
                first = True
                for idx in sorted(self.llm_status['results'].keys()):
                    r = self.llm_status['results'][idx]
                    if first:
                        prefix = " ▪ "
                        first = False
                    else:
                        prefix = " ┃ "
                    if r['grade'] is not None:
                        self.console.print(f"{prefix}[cyan]{r['provider']}:[/cyan] [bold]{r['grade']:.1f}[/bold]", end="")
                    else:
                        self.console.print(f"{prefix}[red]{r['provider']}: erreur[/red]", end="")
                self.console.print("")

        elif event_type == 'llm_error':
            provider = data.get('provider', '???')
            error = data.get('error', 'Unknown error')
            self.console.print(f"    [red]✗ {provider}: {error}[/red]")

        elif event_type == 'question_done':
            q_id = data['question_id']
            grade = data['grade']
            max_pts = data['max_points']
            final_method = data.get('final_method', 'consensus')

            if final_method in ('consensus', 'single_llm'):
                color, icon = "green", "✓"
            elif final_method == 'verification_consensus':
                color, icon = "yellow", "✓"
            elif final_method == 'ultimatum_consensus':
                color, icon = "dark_orange", "✓"
            else:
                color, icon = "red", "⚠"

            self.current_copy_questions[q_id] = {
                'grade': grade, 'max_pts': max_pts, 'color': color, 'icon': icon, 'note': ''
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

            questions_to_display = final_questions if final_questions else self.current_copy_questions

            self.console.print(f"\n  [bold cyan]── Copie {copy_idx}: {student_name} ──[/bold cyan]")

            grades_str = "  "
            for qid in sorted(questions_to_display.keys(), key=natural_sort_key):
                q = questions_to_display[qid]
                grade = q.get('grade', 0)
                max_pts = q.get('max_points', q.get('max_pts', 1))
                if max_pts == int(max_pts):
                    grades_str += f"{qid}: [bold]{grade:.0f}/{int(max_pts)}[/bold]  "
                else:
                    grades_str += f"{qid}: [bold]{grade:.0f}/{max_pts}[/bold]  "
            self.console.print(grades_str)

            token_usage = data.get('token_usage')
            tokens_str = ""
            if token_usage:
                current_total = token_usage.get('total_tokens', 0)
                tokens_this_copy = current_total - self.prev_tokens['total']
                self.prev_tokens['total'] = current_total
                if tokens_this_copy > 0:
                    tokens_str = f" [dim]│ {tokens_this_copy:,} tokens[/dim]"

            color = "green" if pct >= 50 else "red"
            self.console.print(f"  [bold {color}]Total: {score:.1f}/{max_s} ({pct:.0f}%)[/bold {color}] [dim]conf: {conf:.0%}[/dim]{tokens_str}")

            if feedback:
                self.console.print(f"  [italic dim]{feedback[:150]}{'...' if len(feedback) > 150 else ''}[/italic dim]")

        elif event_type == 'feedback_start':
            self.console.print(f"  [dim]Génération de l'appréciation...[/dim]", end="")

        elif event_type == 'feedback_done':
            feedback = data.get('feedback', '')
            if feedback:
                self.console.print(f" [green]✓[/green]\n  [italic]{feedback}[/italic]")
            else:
                self.console.print(" [green]✓[/green]")

        elif event_type == 'single_pass_complete':
            providers = data.get('providers', [])
            single_pass = data.get('single_pass', {})

            all_qids = set()
            for provider in providers:
                all_qids.update(single_pass.get(provider, {}).get('questions', {}).keys())
            sorted_qids = sorted(all_qids, key=natural_sort_key)

            table = Table(show_header=True, header_style="bold dim", show_lines=False, box=None, padding=(0, 2))
            table.add_column("Question", style="bold", width=10)
            for idx, provider in enumerate(providers):
                if "gemini" in provider.lower():
                    if "3" in provider: short_name = "Gemini 3"
                    elif "2.5" in provider or "flash" in provider.lower(): short_name = "Gemini 2.5"
                    else: short_name = f"Gemini {idx+1}"
                elif "openai" in provider.lower() or "gpt" in provider.lower():
                    short_name = "GPT-4o" if "4o" in provider.lower() else "GPT"
                else: short_name = f"LLM{idx+1}"
                table.add_column(short_name, justify="center", width=12)

            for qid in sorted_qids:
                row = [qid]
                for provider in providers:
                    grade = single_pass.get(provider, {}).get('questions', {}).get(qid, {}).get('grade', 0)
                    row.append(f"{grade:.1f}")
                table.add_row(*row)
            self.console.print(table)

        elif event_type == 'analysis_complete':
            agreed = data.get('agreed', 0)
            total = data.get('total', 0)
            flagged = data.get('flagged', 0)
            flagged_questions = data.get('flagged_questions', [])

            if flagged == 0:
                self.console.print(f"  [green]✓ Analyse: {agreed}/{total} questions en accord[/green]")
            else:
                self.console.print(f"  [yellow]📊 Analyse: {agreed}/{total} accord, {flagged} désaccord(s)[/yellow]")
                for fq in flagged_questions:
                    qid = fq.get('question_id')
                    reason = fq.get('reason', '')
                    llm1_grade = fq.get('llm1', {}).get('grade', 0)
                    llm2_grade = fq.get('llm2', {}).get('grade', 0)
                    llm1_reading = fq.get('llm1', {}).get('reading', '')[:30]
                    llm2_reading = fq.get('llm2', {}).get('reading', '')[:30]
                    if 'Lectures' in reason or 'lecture' in reason:
                        self.console.print(f"    [yellow]⚠ {qid}:[/yellow] {reason}\n        [dim]{llm1_reading}...[/dim] vs [dim]{llm2_reading}...[/dim]")
                    else:
                        self.console.print(f"    [yellow]⚠ {qid}:[/yellow] {reason} ({llm1_grade:.1f} vs {llm2_grade:.1f})")

        elif event_type == 'verification_start':
            self.record_phase_tokens(self.current_sub_phase, event_type)
            self.current_sub_phase = WorkflowPhase.VERIFICATION
            self.console.print(f"  [dim]🔄 Vérification {data.get('question_id')}...[/dim]")

        elif event_type == 'batch_comparison_ready':
            providers = data.get('providers', ['LLM1', 'LLM2'])
            copies = data.get('copies', [])
            self.llm_status['comparison_data'] = data

            def get_short_name(p_str: str, idx: int) -> str:
                model = p_str.replace('LLM1: ', '').replace('LLM2: ', '')
                if 'gemini' in model.lower():
                    if 'flash' in model.lower(): return 'Gemini Flash'
                    elif 'pro' in model.lower(): return 'Gemini Pro'
                    return 'Gemini'
                elif 'gpt' in model.lower() or 'openai' in model.lower():
                    if '4o' in model.lower(): return 'GPT-4o'
                    elif '4' in model: return 'GPT-4'
                    return 'GPT'
                elif 'claude' in model.lower():
                    if 'opus' in model.lower(): return 'Claude Opus'
                    elif 'sonnet' in model.lower(): return 'Claude Sonnet'
                    return 'Claude'
                return f'LLM{idx+1}'

            p1_short = get_short_name(providers[0], 0)
            p2_short = get_short_name(providers[1], 1) if len(providers) > 1 else ''

            for copy_info in copies:
                copy_idx = copy_info.get('copy_index', '')
                student_name = copy_info.get('student_name') or '???'
                questions = copy_info.get('questions', {})

                self.console.print(f"\n[bold cyan]═══ Copie {copy_idx}: {student_name} ═══[/bold cyan]")
                self.console.print(f"   {'Q':<4} │ {p1_short:<12} │ {p2_short:<12} │ {'Status'}")
                self.console.print(f"   {'─'*4}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*12}")

                for qid in sorted(questions.keys(), key=natural_sort_key):
                    q = questions[qid]
                    max_pts = q.get('max_points')
                    def format_grade(g, max_pts):
                        if g is None: return "erreur"
                        return f"{g:.1f}/{int(max_pts)}" if max_pts == int(max_pts) else f"{g:.1f}/{max_pts}"

                    g1_str = format_grade(q.get('llm1_grade'), max_pts)
                    g2_str = format_grade(q.get('llm2_grade'), max_pts)
                    
                    if g1_str == "erreur" or g2_str == "erreur":
                        status = "[red]✗ erreur[/red]"
                    else:
                        status = "[green]✓ accord[/green]" if q.get('agreement', True) else "[yellow]⚠ désaccord[/yellow]"
                        
                    self.console.print(f"   {qid:<4} │ {g1_str:<12} │ {g2_str:<12} │ {status}")
                self.console.print("")

        elif event_type == 'batch_verification_start':
            self.record_phase_tokens(self.current_sub_phase, event_type)
            self.current_sub_phase = WorkflowPhase.VERIFICATION
            self.console.print(f"\n  [dim]🔄 Vérification des désaccords...[/dim]")

        elif event_type == 'batch_verification_done':
            self.record_phase_tokens(WorkflowPhase.VERIFICATION, event_type)
            self.current_sub_phase = WorkflowPhase.GRADING
            questions = data.get('questions', [])
            if questions:
                resolved_cases = [q for q in questions if not q.get('goes_to_ultimatum', False)]
                ultimatum_cases = [q for q in questions if q.get('goes_to_ultimatum', False)]
                if resolved_cases:
                    self.console.print(f"\n  [bold]📋 Vérification - Consensus atteint:[/bold]")
                    for q in resolved_cases:
                        cidx = q.get('copy_index', '')
                        prefix = f"    Copie {cidx} " if cidx else "    "
                        self.console.print(f"{prefix}{q.get('question_id')}: [green]{q.get('final_grade', 0):.1f}[/green] (consensus)")
                if ultimatum_cases:
                    self.console.print(f"\n  [dim]📋 Vérification - Cas nécessitant l'ultimatum ({len(ultimatum_cases)}):[/dim]")
                    for q in ultimatum_cases:
                        cidx = q.get('copy_index', '')
                        prefix = f"    [dim]Copie {cidx} " if cidx else "    [dim]"
                        reasons = q.get('ultimatum_reasons', [])
                        reason_str = f" ({', '.join(reasons)})" if reasons else ""
                        self.console.print(f"{prefix}{q.get('question_id')} → ultimatum{reason_str}[/dim]")

        elif event_type == 'batch_ultimatum_start':
            self.record_phase_tokens(self.current_sub_phase, event_type)
            self.current_sub_phase = WorkflowPhase.ULTIMATUM
            self.console.print(f"\n  [dim]⚖️ Ultimatum (désaccords persistants)...[/dim]")

        elif event_type == 'batch_ultimatum_done':
            self.record_phase_tokens(WorkflowPhase.ULTIMATUM, event_type)
            self.current_sub_phase = WorkflowPhase.GRADING
            questions = data.get('questions', [])
            if questions:
                self.console.print(f"\n  [bold]📋 Résultats de l'ultimatum:[/bold]")
                for q in questions:
                    cidx = q.get('copy_index', '')
                    prefix = f"    Copie {cidx} " if cidx else "    "
                    self.console.print(f"{prefix}{q.get('question_id')}: [dark_orange]{q.get('final_grade', 0):.1f}[/dark_orange] ({q.get('method', 'unknown')})")

        elif event_type == 'ultimatum_parse_warning':
            self.console.print(f"\n  [bold red]⚠️ {data.get('warning', 'Ultimatum parsing failed')}[/bold red]\n  [dim]Les décisions finales peuvent être imprécises[/dim]")

        elif event_type == 'verification_done':
            self.record_phase_tokens(WorkflowPhase.VERIFICATION, event_type)
            self.current_sub_phase = WorkflowPhase.GRADING

        elif event_type == 'ultimatum_start':
            self.record_phase_tokens(self.current_sub_phase, event_type)
            self.current_sub_phase = WorkflowPhase.ULTIMATUM

        elif event_type == 'ultimatum_done':
            self.record_phase_tokens(WorkflowPhase.ULTIMATUM, event_type)
            self.current_sub_phase = WorkflowPhase.GRADING

    def get_current_sub_phase(self):
        return self.current_sub_phase
