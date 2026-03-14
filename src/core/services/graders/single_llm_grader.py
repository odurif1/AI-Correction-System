import asyncio
import logging
from typing import List, Optional

from core.models import CopyDocument, GradedCopy, SessionStatus
from core.services.graders.base import BaseGrader, GradingContext
from audit.builder import build_audit_from_llm_comparison, extract_final_question_outputs
from utils.sorting import question_sort_key

logger = logging.getLogger(__name__)


class SingleLLMGrader(BaseGrader):
    """Grade copies one-by-one with a single LLM provider."""

    async def grade_all(self, progress_callback=None) -> List[GradedCopy]:
        from ai.single_pass_grader import SinglePassGrader

        ctx = self.ctx
        ctx.session.transition_to(SessionStatus.CORRECTION)
        total_copies = len(ctx.session.copies)
        questions = self._build_questions_list()

        semaphore = asyncio.Semaphore(ctx.parallel)

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
                        'providers': ['single_llm']
                    })

                grader = SinglePassGrader(ctx.ai)

                if ctx.second_reading:
                    result, verification_audit = await grader.grade_with_self_verification(
                        questions, image_paths, "fr"
                    )
                else:
                    result = await grader.grade_all_questions(
                        questions, image_paths, "fr"
                    )
                    verification_audit = None

                # Convert to GradedCopy
                graded = GradedCopy(
                    copy_id=copy.id,
                    policy_version=ctx.session.policy.version
                )

                for q_id in sorted(result.questions.keys(), key=question_sort_key):
                    q_result = result.questions[q_id]
                    graded.grades[q_id] = q_result.grade
                    graded.max_points_by_question[q_id] = q_result.max_points

                if result.student_name:
                    copy.student_name = result.student_name

                graded.total_score = sum(graded.grades.values())
                graded.max_score = sum(graded.max_points_by_question.values())
                graded.confidence = 0.5

                # Build unified grading audit
                provider_name = getattr(ctx.ai, 'model_name', 'single_llm')
                llm_comparison_for_audit = {
                    "options": {
                        "mode": "individual",
                        "providers": [provider_name],
                        "second_reading": ctx.second_reading,
                        "duration_ms": result.duration_ms
                    },
                    "llm_comparison": {
                        "copy_1": {
                            "questions": {
                                qid: {
                                    provider_name: {
                                        "grade": q_result.grade,
                                        "max_points": q_result.max_points,
                                        "question_text": q_result.question_text,
                                        "reading": q_result.student_answer_read,
                                        "reasoning": q_result.reasoning,
                                        "feedback": q_result.feedback,
                                        "confidence": q_result.confidence
                                    },
                                    "final": {
                                        "grade": q_result.grade,
                                        "max_points": q_result.max_points,
                                        "confidence": q_result.confidence,
                                        "reasoning": q_result.reasoning,
                                        "feedback": q_result.feedback,
                                        "method": "single_llm",
                                        "agreement": None
                                    }
                                }
                                for qid, q_result in result.questions.items()
                            }
                        }
                    }
                }
                if result.student_name:
                    llm_comparison_for_audit["llm_comparison"]["copy_1"]["student_detection"] = {
                        "final_resolved_name": result.student_name,
                        "llm1_student_name": result.student_name
                    }
                graded.grading_audit = build_audit_from_llm_comparison(
                    llm_comparison_for_audit,
                    mode="single",
                    grading_method="individual",
                    verification_mode="none",
                    provider_names=[provider_name],
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

                # Notify per question
                for q_id in sorted(graded.grades.keys(), key=question_sort_key):
                    if progress_callback:
                        await self._call_callback(progress_callback, 'question_done', {
                            'copy_index': i + 1,
                            'question_id': q_id,
                            'grade': graded.grades[q_id],
                            'max_points': graded.max_points_by_question.get(q_id, 1.0),
                            'method': 'single_llm_second_reading' if ctx.second_reading else 'single_llm',
                            'agreement': True
                        })

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
                        'token_usage': token_usage
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
                self._save_sync(last_graded=result)

        return ctx.session.graded_copies
