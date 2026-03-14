import asyncio
import logging
from typing import List, Optional

from core.models import CopyDocument, GradedCopy, SessionStatus
from core.services.graders.base import BaseGrader, GradingContext
from audit.builder import build_audit_from_llm_comparison, extract_final_question_outputs
from utils.sorting import question_sort_key

logger = logging.getLogger(__name__)


class DualLLMGrader(BaseGrader):
    """Grade copies one-by-one using two LLM providers with comparison."""

    async def grade_all(self, progress_callback=None) -> List[GradedCopy]:
        ctx = self.ctx

        if not hasattr(ctx.ai, 'grade_copy'):
            raise RuntimeError("AI provider does not support stateless grading")

        ctx.session.transition_to(SessionStatus.CORRECTION)

        if hasattr(ctx.ai, 'set_progress_callback') and progress_callback:
            ctx.ai.set_progress_callback(progress_callback)

        total_copies = len(ctx.session.copies)
        provider_names = [name for name, _ in ctx.ai.providers] if hasattr(ctx.ai, 'providers') else []
        questions = self._build_questions_list()

        semaphore = asyncio.Semaphore(ctx.parallel)

        # Get callbacks from orchestrator
        disagreement_callback = ctx.get_orchestrator_attr('_disagreement_callback')
        reading_disagreement_callback = ctx.get_orchestrator_attr('_reading_disagreement_callback')

        async def grade_one_copy(i: int, copy: CopyDocument) -> Optional[GradedCopy]:
            async with semaphore:
                if progress_callback:
                    await self._call_callback(progress_callback, 'copy_start', {
                        'copy_index': i + 1,
                        'total_copies': total_copies,
                        'copy_id': copy.id,
                        'student_name': copy.student_name,
                        'questions': list(ctx.grading_scale.keys())
                    })

                image_paths = [str(p) for p in copy.page_images] if copy.page_images else []
                if not image_paths:
                    return None

                if progress_callback:
                    await self._call_callback(progress_callback, 'single_pass_start', {
                        'num_questions': len(questions),
                        'providers': provider_names
                    })

                result = await ctx.ai.grade_copy(
                    questions=questions,
                    image_paths=image_paths,
                    language="fr",
                    disagreement_callback=disagreement_callback,
                    reading_disagreement_callback=reading_disagreement_callback,
                    second_reading=ctx.second_reading
                )

                # Extract audit info
                audit = result.get("audit", {})
                single_pass = audit.get("single_pass", {})
                disagreement_report = audit.get("disagreement_report", {})
                verification = audit.get("verification", {})

                if progress_callback:
                    await self._call_callback(progress_callback, 'single_pass_complete', {
                        'providers': provider_names,
                        'single_pass': single_pass
                    })

                if progress_callback:
                    await self._call_callback(progress_callback, 'analysis_complete', {
                        'agreed': disagreement_report.get('agreed', 0),
                        'flagged': disagreement_report.get('flagged', 0),
                        'total': disagreement_report.get('total_questions', 0),
                        'flagged_questions': disagreement_report.get('flagged_questions', [])
                    })

                for flagged in disagreement_report.get('flagged_questions', []):
                    qid = flagged.get('question_id')
                    if progress_callback:
                        await self._call_callback(progress_callback, 'verification_start', {
                            'question_id': qid,
                            'reason': flagged.get('reason'),
                            'llm1_grade': flagged.get('llm1', {}).get('grade'),
                            'llm2_grade': flagged.get('llm2', {}).get('grade')
                        })

                # Notify final results per question
                results_data = result.get("results", {})
                for q_id in sorted(results_data.keys(), key=question_sort_key):
                    q_result = results_data[q_id]
                    q_audit = verification.get(q_id, {})
                    method = q_audit.get('method', 'unknown')
                    agreement = q_audit.get('agreement', True)
                    detected_max_points = q_result.get("max_points", ctx.grading_scale.get(q_id, 1.0))

                    if progress_callback:
                        await self._call_callback(progress_callback, 'question_done', {
                            'copy_index': i + 1,
                            'question_id': q_id,
                            'grade': q_result.get("grade", 0),
                            'max_points': detected_max_points,
                            'method': method,
                            'agreement': agreement
                        })

                # Convert to GradedCopy
                graded = GradedCopy(
                    copy_id=copy.id,
                    policy_version=ctx.session.policy.version
                )

                consensus_name = result.get("student_name")
                if consensus_name:
                    copy.student_name = consensus_name

                for q_id in sorted(results_data.keys(), key=question_sort_key):
                    q_result = results_data[q_id]
                    graded.grades[q_id] = q_result.get("grade", 0)
                    graded.max_points_by_question[q_id] = q_result.get("max_points", 1.0)

                llm_comparison = result.get("llm_comparison", {})

                graded.total_score = sum(g or 0 for g in graded.grades.values())
                graded.max_score = (
                    sum(graded.max_points_by_question.values())
                    if graded.max_points_by_question
                    else sum(ctx.grading_scale.values())
                )
                graded.confidence = 0.5

                llm_comp_data = {
                    "options": result.get("options", {}),
                    "llm_comparison": llm_comparison,
                    "student_name_info": result.get("student_name_info", {}),
                    "summary": result.get("summary", {}),
                    "timing": result.get("timing", {})
                }
                graded.grading_audit = build_audit_from_llm_comparison(
                    llm_comp_data,
                    mode="dual",
                    grading_method="individual",
                    verification_mode="none",
                    provider_names=provider_names,
                    grading_scale=ctx.grading_scale
                )

                final_outputs = extract_final_question_outputs(graded.grading_audit)
                graded.confidence_by_question = {
                    q_id: data["confidence"]
                    for q_id, data in final_outputs.items()
                    if data["confidence"] is not None
                }
                graded.reasoning = {
                    q_id: data["reasoning"]
                    for q_id, data in final_outputs.items()
                    if data["reasoning"]
                }
                graded.student_feedback = {
                    q_id: data["feedback"]
                    for q_id, data in final_outputs.items()
                    if data["feedback"]
                }
                if graded.confidence_by_question:
                    graded.confidence = (
                        sum(c for c in graded.confidence_by_question.values()) / len(graded.confidence_by_question)
                    )

                if progress_callback:
                    token_usage = None
                    if hasattr(ctx.ai, 'get_token_usage'):
                        token_usage = ctx.ai.get_token_usage()

                    await self._call_callback(progress_callback, 'copy_done', {
                        'copy_index': i + 1,
                        'total_copies': total_copies,
                        'copy_id': copy.id,
                        'student_name': copy.student_name,
                        'total_score': graded.total_score,
                        'max_score': graded.max_score,
                        'confidence': graded.confidence,
                        'token_usage': token_usage,
                        'grading_summary': result.get("summary", {})
                    })

                return graded

        # Execute all in parallel (limited by semaphore)
        tasks = [grade_one_copy(i, copy) for i, copy in enumerate(ctx.session.copies)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error grading copy {i+1}: {result}")
                if progress_callback:
                    copy = ctx.session.copies[i] if i < len(ctx.session.copies) else None
                    await self._call_callback(progress_callback, 'copy_error', {
                        'copy_index': i + 1,
                        'total_copies': total_copies,
                        'copy_id': copy.id if copy else None,
                        'student_name': copy.student_name if copy else None,
                        'error': str(result)
                    })
            elif result is not None:
                ctx.session.graded_copies.append(result)
                self._save_sync()

        return ctx.session.graded_copies
