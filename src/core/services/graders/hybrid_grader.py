import asyncio
import logging
from pathlib import Path
from typing import List

from core.models import CopyDocument, GradedCopy, SessionStatus
from core.services.graders.base import BaseGrader, GradingContext
from audit.builder import build_audit_from_llm_comparison, extract_final_question_outputs
from config.settings import get_settings
from utils.sorting import question_sort_key

logger = logging.getLogger(__name__)


class HybridGrader(BaseGrader):
    """Grade copies with LLM1=batch + LLM2=individual, then compare."""

    async def grade_all(self, progress_callback=None) -> List[GradedCopy]:
        from ai.batch_grader import BatchGrader, BatchResult
        from ai.single_pass_grader import SinglePassGrader

        ctx = self.ctx
        total_copies = len(ctx.session.copies)

        if not hasattr(ctx.ai, 'providers') or len(ctx.ai.providers) < 2:
            raise RuntimeError("Hybrid mode requires dual LLM with 2 providers")

        provider_names = [name for name, _ in ctx.ai.providers]
        llm1_name, llm1_provider = ctx.ai.providers[0]
        llm2_name, llm2_provider = ctx.ai.providers[1]

        if progress_callback:
            await self._call_callback(progress_callback, 'hybrid_start', {
                'total_copies': total_copies,
                'llm1': llm1_name,
                'llm2': llm2_name,
                'llm1_mode': 'batch',
                'llm2_mode': 'individual'
            })

        # Prepare copies data
        copies_data = []
        for i, copy in enumerate(ctx.session.copies):
            image_paths = []
            if copy.page_images:
                image_paths = copy.page_images
            elif copy.pdf_path:
                from vision.pdf_reader import PDFReader
                reader = PDFReader(copy.pdf_path)
                temp_dir = Path(ctx.store.session_dir) / "temp"
                temp_dir.mkdir(exist_ok=True)

                for page_num in range(copy.page_count):
                    image_path = str(temp_dir / f"hybrid_copy_{i+1}_page_{page_num}.png")
                    image_bytes = reader.get_page_image_bytes(page_num)
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                    image_paths.append(image_path)
                reader.close()

            copies_data.append({
                'copy_index': i + 1,
                'image_paths': image_paths,
                'student_name': copy.student_name,
                'start_page': copy.start_page,
                'end_page': copy.end_page
            })

        # Build questions dict
        questions = {}
        for qid, max_pts in ctx.grading_scale.items():
            questions[qid] = {
                'text': '',
                'criteria': '',
                'max_points': max_pts
            }

        language = 'fr'

        # ===== PARALLEL: LLM1 (batch) + LLM2 (individual) =====
        async def run_llm1_batch():
            grader = BatchGrader(llm1_provider)
            return await grader.grade_batch(copies_data, questions, language)

        async def run_llm2_individual():
            results = []
            semaphore = asyncio.Semaphore(ctx.parallel)

            async def grade_one_copy(i: int, copy):
                async with semaphore:
                    grader = SinglePassGrader(llm2_provider)
                    copy_data = copies_data[i]
                    q_list = [{'id': qid, 'text': '', 'criteria': '', 'max_points': qdata['max_points']}
                              for qid, qdata in questions.items()]
                    result = await grader.grade_all_questions(
                        q_list, copy_data['image_paths'], language
                    )
                    return (i, result)

            tasks = [grade_one_copy(i, copy) for i, copy in enumerate(ctx.session.copies)]
            individual_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in individual_results:
                if not isinstance(result, Exception):
                    results.append(result)
            return results

        if progress_callback:
            await self._call_callback(progress_callback, 'hybrid_grading', {'status': 'running_both'})

        batch_result, individual_results = await asyncio.gather(
            run_llm1_batch(),
            run_llm2_individual(),
            return_exceptions=True
        )

        if isinstance(batch_result, Exception):
            logger.error(f"LLM1 batch failed: {batch_result}")
            batch_result = None
        if isinstance(individual_results, Exception):
            logger.error(f"LLM2 individual failed: {individual_results}")
            individual_results = []

        # ===== MERGE AND COMPARE =====
        graded_copies = []

        for i, original_copy in enumerate(ctx.session.copies):
            # Get LLM1 (batch) result
            llm1_grades = {}
            llm1_name_detected = None
            if batch_result and batch_result.parse_success:
                for copy_result in batch_result.copies:
                    if copy_result.copy_index == i + 1:
                        llm1_grades = {qid: qdata['grade'] for qid, qdata in copy_result.questions.items()}
                        llm1_name_detected = copy_result.student_name
                        break

            # Get LLM2 (individual) result
            llm2_grades = {}
            llm2_name_detected = None
            for idx, result in individual_results:
                if idx == i and result and result.parse_success:
                    llm2_grades = {qid: qdata.get('grade', 0) for qid, qdata in result.questions.items()}
                    llm2_name_detected = result.student_name
                    break

            # Compare and merge
            final_grades = {}
            disagreements = []
            llm_comparison_data = {}

            for qid in ctx.grading_scale.keys():
                g1 = llm1_grades.get(qid, 0)
                g2 = llm2_grades.get(qid, 0)
                max_pts = ctx.grading_scale.get(qid, 1)
                relative_threshold = max_pts * get_settings().grade_agreement_threshold

                if abs(g1 - g2) >= relative_threshold:
                    disagreements.append({
                        'question_id': qid,
                        'llm1_grade': g1,
                        'llm2_grade': g2,
                        'max_points': max_pts
                    })
                    final_grades[qid] = (g1 + g2) / 2
                else:
                    final_grades[qid] = g1

                llm_comparison_data[qid] = {
                    llm1_name: {'grade': g1, 'max_points': max_pts, 'mode': 'batch'},
                    llm2_name: {'grade': g2, 'max_points': max_pts, 'mode': 'individual'},
                    'final': {
                        'grade': final_grades[qid],
                        'agreement': abs(g1 - g2) < relative_threshold
                    }
                }

            student_name = original_copy.student_name or llm1_name_detected or llm2_name_detected

            copy_llm_comparison = {
                "options": {
                    "mode": "hybrid",
                    "providers": [llm1_name, llm2_name],
                    "llm1_mode": "batch",
                    "llm2_mode": "individual"
                },
                "llm_comparison": llm_comparison_data
            }

            graded = GradedCopy(
                copy_id=original_copy.id,
                grades=final_grades,
                total_score=sum(final_grades.values()),
                max_score=sum(ctx.grading_scale.values()),
                confidence=0.85 if not disagreements else 0.70,
                feedback=f"Hybrid mode. Disagreements: {len(disagreements)}",
                grading_audit=build_audit_from_llm_comparison(
                    copy_llm_comparison,
                    mode="dual",
                    grading_method="hybrid",
                    verification_mode="none",
                    provider_names=[llm1_name, llm2_name],
                    grading_scale=ctx.grading_scale
                )
            )

            if graded.grading_audit:
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

            if student_name:
                original_copy.student_name = student_name

            graded_copies.append(graded)
            ctx.session.graded_copies.append(graded)

            if progress_callback:
                await self._call_callback(progress_callback, 'copy_done', {
                    'copy_index': i + 1,
                    'total_copies': total_copies,
                    'copy_id': original_copy.id,
                    'student_name': student_name,
                    'total_score': graded.total_score,
                    'max_score': graded.max_score,
                    'disagreements': [d['question_id'] for d in disagreements]
                })

        self._save_sync()

        if progress_callback:
            await self._call_callback(progress_callback, 'hybrid_done', {
                'total_copies': total_copies,
                'graded_copies': len(graded_copies),
                'patterns': batch_result.patterns if batch_result else {}
            })

        return graded_copies
