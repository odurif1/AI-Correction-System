import asyncio
import logging
import traceback
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

from audit.builder import build_audit_from_llm_comparison, extract_final_question_outputs
from config.settings import get_settings
from core.models import CopyDocument, GradedCopy, SessionStatus
from core.services.graders.base import BaseGrader, GradingContext
from utils.sorting import question_sort_key

logger = logging.getLogger(__name__)


class BatchGrader(BaseGrader):
    """Grade all copies in one API call (batch mode)."""

    async def grade_all(self, progress_callback=None) -> List[GradedCopy]:
        from ai.batch_grader import grade_all_copies_in_batches, BatchResult

        ctx = self.ctx
        total_copies = len(ctx.session.copies)
        batch_verify = ctx.get_orchestrator_attr('_batch_verify', 'grouped')
        use_chat_continuation = ctx.get_orchestrator_attr('_use_chat_continuation', False)

        # ===== PRE-DETECTION PHASE =====
        pre_detected_students = None
        detection_hints = None

        # Check stored detection first
        stored_detection = ctx.store.load_detection()
        if stored_detection:
            if stored_detection.students and len(stored_detection.students) >= 5:
                # Many LLM-confirmed students — trust individual boundaries
                pre_detected_students = stored_detection.students
                logger.info(f"Reusing stored detection: {len(pre_detected_students)} LLM-confirmed students")
            elif stored_detection.consistent_pages_per_student and stored_detection.pages_per_student:
                # Structural info only — pass as hints, let LLM detect actual boundaries
                first_copy = ctx.session.copies[0] if ctx.session.copies else None
                if first_copy:
                    subject_pages = stored_detection.subject_page_count or 0
                    expected = (first_copy.page_count - subject_pages) // stored_detection.pages_per_student
                    detection_hints = {
                        'expected_students': expected,
                        'pages_per_student': stored_detection.pages_per_student,
                        'subject_pages': subject_pages,
                    }
                    logger.info(f"Structural hints: ~{expected} students, ~{stored_detection.pages_per_student}pp (LLM will detect actual boundaries)")
            elif stored_detection.students:
                pre_detected_students = stored_detection.students
                logger.info(f"Reusing stored detection: {len(pre_detected_students)} students")
        elif total_copies == 1 and not ctx.pages_per_copy and ctx.comparison_mode:
            logger.info("Running pre-detection to ensure consistent student detection between LLMs")
            if progress_callback:
                await self._call_callback(progress_callback, 'detection_start', {
                    'reason': 'pre_detection_for_dual_llm'
                })

            try:
                from analysis.detection import Detector
                first_copy = ctx.session.copies[0]

                _, provider = ctx.ai.providers[0]
                detector = Detector(
                    user_id=ctx.session.user_id,
                    session_id=ctx.session.session_id,
                    language='fr',
                    provider=provider
                )

                detection_result = detector.detect(first_copy.pdf_path, mode="auto")

                from rich.console import Console
                console = Console()
                console.print(f"\n[bold cyan]🔍 Pré-détection pour mode double LLM:[/bold cyan]")

                doc_type = str(detection_result.document_type).replace("DocumentType.", "")
                console.print(f"  Type document: [bold]{doc_type}[/bold]")
                console.print(f"  Pages totales: [bold]{detection_result.page_count}[/bold]")

                if detection_result.consistent_pages_per_student and detection_result.pages_per_student:
                    console.print(f"  Pages/élève: [bold]{detection_result.pages_per_student}[/bold] [green](structure cohérente)[/green]")
                elif detection_result.pages_per_student:
                    console.print(f"  Pages/élève: [bold]{detection_result.pages_per_student}[/bold] [yellow](variable)[/yellow]")

                if detection_result.subject_page_count and detection_result.subject_page_count > 0:
                    console.print(f"  Pages sujet: [bold]{detection_result.subject_page_count}[/bold]")

                if detection_result.consistent_pages_per_student and detection_result.pages_per_student:
                    pages_per_student = detection_result.pages_per_student
                    subject_pages = detection_result.subject_page_count or 0
                    total_pages = first_copy.page_count
                    student_pages = total_pages - subject_pages
                    num_students = student_pages // pages_per_student

                    console.print(f"  [green]✓ Structure cohérente → calcul mathématique:[/green]")
                    console.print(f"    Élèves: [bold]{num_students}[/bold] ({student_pages} pages ÷ {pages_per_student} pp)")

                    from core.models import StudentInfo
                    pre_detected_students = []
                    for i in range(num_students):
                        start_page = subject_pages + (i * pages_per_student) + 1
                        end_page = start_page + pages_per_student - 1
                        pre_detected_students.append(StudentInfo(
                            index=i + 1,
                            name=None,
                            start_page=start_page,
                            end_page=end_page,
                            confidence=0.9
                        ))

                    logger.info(f"Calculated {len(pre_detected_students)} students from structure")
                elif detection_result.students:
                    pre_detected_students = detection_result.students
                    console.print(f"  Élèves détectés par LLM: [bold]{len(pre_detected_students)}[/bold]")
                    logger.info(f"Pre-detected {len(pre_detected_students)} students from LLM")

                if pre_detected_students and progress_callback:
                    await self._call_callback(progress_callback, 'detection_done', {
                        'students_count': len(pre_detected_students),
                        'students': [
                            {'name': s.name, 'start_page': s.start_page, 'end_page': s.end_page}
                            for s in pre_detected_students
                        ],
                        'calculated': detection_result.consistent_pages_per_student
                    })
            except Exception as e:
                logger.warning(f"Pre-detection failed, falling back to LLM detection during grading: {e}")
                pre_detected_students = None

        # Notify start
        if progress_callback:
            await self._call_callback(progress_callback, 'batch_start', {
                'total_copies': total_copies,
                'mode': 'batch',
                'pre_detected': pre_detected_students is not None
            })

        # Prepare copies data
        copies_data = []

        if pre_detected_students:
            logger.info(f"Using {len(pre_detected_students)} pre-detected students for grading")

            first_copy = ctx.session.copies[0]
            all_page_images = []
            if first_copy.pdf_path:
                from vision.pdf_reader import PDFReader
                reader = PDFReader(first_copy.pdf_path)
                temp_dir = Path(ctx.store.session_dir) / "temp"
                temp_dir.mkdir(exist_ok=True)

                for page_num in range(first_copy.page_count):
                    image_path = str(temp_dir / f"page_{page_num}.png")
                    image_bytes = reader.get_page_image_bytes(page_num)
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                    all_page_images.append(image_path)
                reader.close()

            for i, student in enumerate(pre_detected_students):
                start_page = max(0, (student.start_page or 1) - 1)
                end_page = min(len(all_page_images), student.end_page or len(all_page_images))
                student_images = all_page_images[start_page:end_page]

                copies_data.append({
                    'copy_index': i + 1,
                    'image_paths': student_images,
                    'student_name': student.name,
                    'start_page': student.start_page,
                    'end_page': student.end_page,
                    'pre_detected': True
                })

                total_copies = len(pre_detected_students)

                if progress_callback:
                    await self._call_callback(progress_callback, 'copy_start', {
                        'copy_index': i + 1,
                        'total_copies': total_copies,
                        'copy_id': f"pre_detected_{i+1}",
                        'student_name': student.name or '???',
                        'questions': []
                    })
        else:
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
                        image_path = str(temp_dir / f"batch_copy_{i+1}_page_{page_num}.png")
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

                if progress_callback:
                    await self._call_callback(progress_callback, 'copy_start', {
                        'copy_index': i + 1,
                        'total_copies': total_copies,
                        'copy_id': copy.id,
                        'student_name': copy.student_name or '???',
                        'questions': []
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

        graded_copies = []
        llm_comparison_data = {}

        # ===== SETUP IMPLICIT CACHING =====
        chat_manager = None
        common_prefix = None

        logger.info(
            f"Cache config: comparison_mode={ctx.comparison_mode}, "
            f"chat_continuation={use_chat_continuation}, "
            f"questions_count={len(questions)}"
        )

        if ctx.comparison_mode and use_chat_continuation:
            from ai.batch_grader import CacheManager
            from prompts.batch import build_common_prefix

            _, provider1 = ctx.ai.providers[0]
            _, provider2 = ctx.ai.providers[1]

            caching_supported = (
                provider1.supports_context_caching() or
                provider2.supports_context_caching()
            )

            if caching_supported:
                try:
                    logger.info("Creating CacheManager for implicit caching (sessions will be set up after detection)")
                    chat_manager = CacheManager(ctx.ai.providers, cache_mode="shared")
                    common_prefix = build_common_prefix(questions, language)
                    logger.info(f"Built common prefix: {len(common_prefix)} chars (~{len(common_prefix)//4} tokens)")
                except Exception as e:
                    logger.warning(f"Cache setup failed: {e}. Continuing without cache.")
                    chat_manager = None
                    common_prefix = None
            else:
                logger.warning("Context caching not supported by providers. Using regular calls.")

        if ctx.comparison_mode:
            # Dual LLM batch
            provider_names = [name for name, _ in ctx.ai.providers]
            llm1_name, llm2_name = provider_names[0], provider_names[1]

            if progress_callback:
                await self._call_callback(progress_callback, 'batch_llm_start', {
                    'providers': provider_names
                })

            async def grade_with_provider(provider, name):
                from ai.batch_grader import BatchGrader as AIBatchGrader
                grader = AIBatchGrader(provider)
                return await grader.grade_batch(
                    copies_data, questions, language,
                    detect_students=pre_detected_students is None,
                    common_prefix=common_prefix,
                    detection_hints=detection_hints
                )

            tasks = [
                grade_with_provider(provider, name)
                for name, provider in ctx.ai.providers
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            llm1_result = results[0] if not isinstance(results[0], Exception) else None
            llm2_result = results[1] if not isinstance(results[1], Exception) else None

            if isinstance(results[0], Exception):
                logger.error(f"LLM1 exception: {results[0]}")
                logger.debug("".join(traceback.format_exception(type(results[0]), results[0], results[0].__traceback__)))
            elif llm1_result:
                logger.debug(f"LLM1 result: parse_success={llm1_result.parse_success}, errors={llm1_result.parse_errors}")
            if isinstance(results[1], Exception):
                logger.error(f"LLM2 exception: {results[1]}")
                logger.debug("".join(traceback.format_exception(type(results[1]), results[1], results[1].__traceback__)))
            elif llm2_result:
                logger.debug(f"LLM2 result: parse_success={llm2_result.parse_success}, errors={llm2_result.parse_errors}")

            if progress_callback:
                await self._call_callback(progress_callback, 'batch_llm_done', {
                    'llm1_success': llm1_result is not None and llm1_result.parse_success,
                    'llm2_success': llm2_result is not None and llm2_result.parse_success
                })

            llm1_success = llm1_result is not None and llm1_result.parse_success
            llm2_success = llm2_result is not None and llm2_result.parse_success

            if not llm1_success and not llm2_success:
                from core.exceptions import DualLLMFailureError
                raise DualLLMFailureError(
                    f"Échec du mode double LLM: {llm1_name}, {llm2_name} n'ont pas retourné de résultats valides. "
                    f"Vérifiez vos clés API et réessayez.",
                    llm1_success=llm1_success,
                    llm2_success=llm2_success
                )

            # If exactly one LLM succeeded, keep the valid result and continue in
            # degraded single-LLM mode instead of failing the whole correction.
            if llm1_success != llm2_success:
                successful_name = llm1_name if llm1_success else llm2_name
                batch_result = llm1_result if llm1_success else llm2_result
                logger.warning(
                    "Batch dual-LLM degraded to single provider for this run: "
                    f"{successful_name} succeeded, "
                    f"{llm2_name if llm1_success else llm1_name} failed"
                )

                llm_comparison_data = {
                    "options": {
                        "mode": "batch_single_fallback",
                        "providers": [successful_name],
                        "total_copies": total_copies,
                        "fallback_reason": "dual_llm_partial_failure",
                    },
                    "llm_comparison": {}
                }

                for copy_result in batch_result.copies:
                    copy_idx = copy_result.copy_index
                    llm_comparison_data["llm_comparison"][f"copy_{copy_idx}"] = {
                        "student_name": copy_result.student_name,
                        "questions": {}
                    }

                    for qid, qdata in copy_result.questions.items():
                        llm_comparison_data["llm_comparison"][f"copy_{copy_idx}"]["questions"][qid] = {
                            successful_name: {
                                "grade": qdata.get('grade', 0),
                                "max_points": qdata.get('max_points', 1),
                                "question_text": qdata.get('question_text', ''),
                                "reading": qdata.get('student_answer_read', ''),
                                "reasoning": qdata.get('reasoning', ''),
                                "feedback": qdata.get('feedback', ''),
                                "confidence": qdata.get('confidence', 0.8)
                            },
                            "final": {
                                "grade": qdata.get('grade', 0),
                                "max_points": qdata.get('max_points', 1),
                                "confidence": qdata.get('confidence', 0.8),
                                "reasoning": qdata.get('reasoning', ''),
                                "feedback": qdata.get('feedback', ''),
                                "method": "single_llm_fallback",
                                "agreement": None
                            }
                        }
            else:
                # ===== CROSS-VERIFY STUDENT NAMES =====
                from utils.name_matching import cross_verify_student_names, format_name_mismatch_message

                name_result = cross_verify_student_names(
                    llm1_result.copies if llm1_result else [],
                    llm2_result.copies if llm2_result else []
                )

                llm_comparison_data = {
                    "options": {
                        "mode": "batch",
                        "providers": provider_names,
                        "total_copies": total_copies
                    },
                    "name_verification": {
                        "matches": name_result.matches,
                        "mismatches": name_result.mismatches,
                        "llm1_only": name_result.llm1_only,
                        "llm2_only": name_result.llm2_only,
                        "all_matched": name_result.all_matched
                    },
                    "llm_comparison": {}
                }

                if name_result.requires_user_action:
                    from core.exceptions import StudentNameMismatchError

                    msg = format_name_mismatch_message(name_result, language)

                    workflow_state = ctx.get_orchestrator_attr('workflow_state')
                    is_auto_mode = workflow_state.auto_mode if workflow_state else False

                    if is_auto_mode:
                        print(f"\n{msg}")
                        print("\n[WARNING] Continuing despite name mismatch (--auto-confirm mode)")
                        llm_comparison_data["name_verification_warning"] = True
                    else:
                        raise StudentNameMismatchError(
                            msg,
                            mismatches=name_result.mismatches,
                            llm1_only=name_result.llm1_only,
                            llm2_only=name_result.llm2_only
                        )

                # Build merged results with comparison data
                batch_result = None
                name_verification = llm_comparison_data.get("name_verification", {})
                llm_comparison_data = {
                    "options": {
                        "mode": "batch",
                        "providers": provider_names,
                        "total_copies": total_copies
                    },
                    "name_verification": name_verification,
                    "llm_comparison": {}
                }

                if llm1_result and llm1_result.parse_success:
                    batch_result = llm1_result
                else:
                    batch_result = llm2_result

                # For each copy, build comparison data
                for copy_result in batch_result.copies:
                    copy_idx = copy_result.copy_index

                    llm1_student_name = copy_result.student_name
                    llm2_student_name = None
                    if llm2_result and llm2_result.parse_success:
                        for c in llm2_result.copies:
                            if c.copy_index == copy_idx:
                                llm2_student_name = c.student_name
                                break

                    llm_comparison_data["llm_comparison"][f"copy_{copy_idx}"] = {
                        "student_name": copy_result.student_name,
                        "student_name_initial": {
                            "llm1_name": llm1_student_name,
                            "llm2_name": llm2_student_name
                        },
                        "questions": {}
                    }

                    llm2_copy = None
                    if llm2_result and llm2_result.parse_success:
                        for c in llm2_result.copies:
                            if c.copy_index == copy_idx:
                                llm2_copy = c
                                break

                    for qid, qdata in copy_result.questions.items():
                        llm1_qdata = {
                            "grade": qdata.get('grade', 0),
                            "max_points": qdata.get('max_points', 1),
                            "question_text": qdata.get('question_text', ''),
                            "reading": qdata.get('student_answer_read', ''),
                            "reasoning": qdata.get('reasoning', ''),
                            "feedback": qdata.get('feedback', ''),
                            "confidence": qdata.get('confidence', 0.8)
                        }

                        llm2_qdata = {}
                        if llm2_copy and qid in llm2_copy.questions:
                            q2 = llm2_copy.questions[qid]
                            llm2_qdata = {
                                "grade": q2.get('grade', 0),
                                "max_points": q2.get('max_points', 1),
                                "question_text": q2.get('question_text', ''),
                                "reading": q2.get('student_answer_read', ''),
                                "reasoning": q2.get('reasoning', ''),
                                "feedback": q2.get('feedback', ''),
                                "confidence": q2.get('confidence', 0.8)
                            }

                        if llm2_qdata:
                            final_grade = (llm1_qdata["grade"] + llm2_qdata["grade"]) / 2
                            max_pts = max(llm1_qdata.get("max_points", 1.0), llm2_qdata.get("max_points", 1.0))
                            threshold = max_pts * get_settings().grade_agreement_threshold
                            grade_agreement = abs(llm1_qdata["grade"] - llm2_qdata["grade"]) < threshold
                            max_points_agreement = abs(llm1_qdata.get("max_points", 1.0) - llm2_qdata.get("max_points", 1.0)) < 0.01
                            agreement = grade_agreement and max_points_agreement
                            r1 = (llm1_qdata.get("reading") or "").lower().strip()
                            r2 = (llm2_qdata.get("reading") or "").lower().strip()
                            initial_reading_similarity = SequenceMatcher(None, r1, r2).ratio() if r1 and r2 else None
                        else:
                            final_grade = llm1_qdata["grade"]
                            max_pts = llm1_qdata.get("max_points", 1.0)
                            agreement = True
                            initial_reading_similarity = None

                        question_data = {
                            "max_points": max_pts,
                            llm1_name: llm1_qdata,
                        }
                        if llm2_qdata:
                            question_data[llm2_name] = llm2_qdata

                    question_data["_initial_final"] = {
                        "grade": final_grade,
                        "max_points": max_pts,
                        "confidence": min(llm1_qdata.get("confidence", 0.8), llm2_qdata.get("confidence", 0.8)) if llm2_qdata else llm1_qdata.get("confidence", 0.8),
                        "reasoning": llm1_qdata.get("reasoning", ""),
                        "feedback": llm1_qdata.get("feedback", ""),
                        "method": "average" if llm2_qdata else "single_llm",
                        "agreement": agreement,
                        "reading_similarity": initial_reading_similarity
                    }

                    llm_comparison_data["llm_comparison"][f"copy_{copy_idx}"]["questions"][qid] = question_data

            # Notify CLI that comparison data is ready
            if progress_callback:
                comparison_summary = {
                    'providers': [llm1_name, llm2_name],
                    'copies': []
                }
                for copy_idx_key, copy_data in llm_comparison_data.get("llm_comparison", {}).items():
                    copy_summary = {
                        'copy_index': copy_idx_key,
                        'student_name': copy_data.get('student_name'),
                        'questions': {}
                    }
                    for qid, qdata in copy_data.get('questions', {}).items():
                        llm1_q = qdata.get(llm1_name, {})
                        llm2_q = qdata.get(llm2_name, {})
                        frozen_max_pts = ctx.session.policy.question_weights.get(qid, 1.0)
                        copy_summary['questions'][qid] = {
                            'llm1_grade': llm1_q.get('grade'),
                            'llm2_grade': llm2_q.get('grade'),
                            'max_points': frozen_max_pts,
                            'agreement': qdata.get('_initial_final', {}).get('agreement', True)
                        }
                    comparison_summary['copies'].append(copy_summary)

                await self._call_callback(progress_callback, 'batch_comparison_ready', comparison_summary)

            # ═══════════════════════════════════════════════════════════════
            # POST-BATCH VERIFICATION
            # ═══════════════════════════════════════════════════════════════
            from ai.batch_grader import (
                detect_disagreements, run_dual_llm_verification,
                run_per_question_dual_verification, run_dual_llm_ultimatum,
                run_per_question_dual_ultimatum,
                Disagreement
            )

            pdf_page_count = None
            end_pages = [c.get('end_page') for c in copies_data if c.get('end_page')]
            if end_pages:
                pdf_page_count = max(end_pages)

            # ===== SETUP CACHE AFTER DETECTION =====
            logger.debug(f"After detection: chat_manager={chat_manager is not None}, use_chat_continuation={use_chat_continuation}")
            if chat_manager and use_chat_continuation:
                try:
                    detected_copies_data = []
                    temp_dir = Path(ctx.store.session_dir) / "temp"

                    for copy_result in batch_result.copies:
                        copy_images = getattr(copy_result, 'image_paths', [])
                        if not copy_images:
                            copy_idx = copy_result.copy_index
                            copy_images = [
                                str(temp_dir / f"batch_copy_{copy_idx}_page_{p}.png")
                                for p in range(10)
                                if (temp_dir / f"batch_copy_{copy_idx}_page_{p}.png").exists()
                            ]

                        detected_copies_data.append({
                            'copy_index': copy_result.copy_index,
                            'image_paths': copy_images,
                            'student_name': copy_result.student_name
                        })

                    if detected_copies_data:
                        await chat_manager.create_sessions(
                            detected_copies_data,
                            questions=questions,
                            language=language
                        )
                        logger.info(f"Cache ready for {len(detected_copies_data)} detected copies")
                except Exception as e:
                    logger.warning(f"Post-detection cache setup failed: {e}")
                    chat_manager = None

            # ===== SETUP EXPLICIT CACHE (Gemini CachedContent) =====
            explicit_cache_mgr = None
            use_explicit_cache = get_settings().use_explicit_cache

            if use_explicit_cache and ctx.comparison_mode:
                # Check if any provider supports explicit caching (Gemini)
                has_gemini = any(
                    hasattr(p, 'create_cached_context')
                    for _, p in ctx.ai.providers
                )

                if has_gemini:
                    try:
                        from ai.batch_grader import ExplicitCacheManager
                        from prompts.batch import build_common_prefix

                        # Collect ALL images from all copies
                        all_images = []
                        for copy_result in batch_result.copies:
                            copy_images = getattr(copy_result, 'image_paths', [])
                            if not copy_images:
                                temp_dir = Path(ctx.store.session_dir) / "temp"
                                copy_idx = copy_result.copy_index
                                copy_images = [
                                    str(temp_dir / f"batch_copy_{copy_idx}_page_{p}.png")
                                    for p in range(10)
                                    if (temp_dir / f"batch_copy_{copy_idx}_page_{p}.png").exists()
                                ]
                            all_images.extend(copy_images)

                        # Build system prompt with barème
                        system_prompt = build_common_prefix(questions, language)

                        if all_images:
                            explicit_cache_mgr = ExplicitCacheManager(ctx.ai.providers)

                            # Create explicit cache
                            cache_created = await explicit_cache_mgr.create_cache(
                                images=all_images,
                                system_prompt=system_prompt,
                                ttl_seconds=1800  # 30 minutes
                            )

                            if cache_created:
                                logger.info(f"Explicit cache created with {len(all_images)} images for verification/ultimatum")
                            else:
                                logger.info("Explicit cache creation failed, falling back to implicit caching")
                                explicit_cache_mgr = None
                    except Exception as e:
                        logger.warning(f"Explicit cache setup failed: {e}. Using implicit caching.")
                        explicit_cache_mgr = None

            # Detect grade disagreements
            disagreements = detect_disagreements(
                llm1_result, llm2_result,
                llm1_name, llm2_name,
                copies_data,
                pdf_page_count=pdf_page_count
            )

            # Detect student name disagreements
            name_disagreements = []
            if ctx.comparison_mode and llm1_result and llm2_result:
                for copy_result in batch_result.copies:
                    llm1_student_name = None
                    llm2_student_name = None

                    for c in llm1_result.copies:
                        if c.copy_index == copy_result.copy_index:
                            llm1_student_name = c.student_name
                            break
                    for c in llm2_result.copies:
                        if c.copy_index == copy_result.copy_index:
                            llm2_student_name = c.student_name
                            break

                    if llm1_student_name and llm2_student_name:
                        n1_normalized = llm1_student_name.lower().strip()
                        n2_normalized = llm2_student_name.lower().strip()
                        if n1_normalized != n2_normalized:
                            name_disagreements.append({
                                'copy_index': copy_result.copy_index,
                                'llm1_name': llm1_student_name,
                                'llm2_name': llm2_student_name
                            })

            if disagreements or (batch_verify == "grouped" and name_disagreements):
                logger.info(f"Detected {len(disagreements)} disagreements, running dual LLM verification ({batch_verify})")

                if progress_callback:
                    await self._call_callback(progress_callback, 'batch_verification_start', {
                        'disagreements_count': len(disagreements),
                        'mode': batch_verify
                    })

                # Determine which cache to use (prefer explicit over implicit)
                effective_cache_manager = explicit_cache_mgr or chat_manager
                if explicit_cache_mgr:
                    logger.info("Using EXPLICIT cache for verification (Gemini CachedContent API)")
                elif chat_manager:
                    logger.info("Using IMPLICIT cache for verification (common prefix caching)")
                else:
                    logger.info("No cache available, using regular API calls for verification")

                # ===== PHASE 1: Dual LLM Verification =====
                if batch_verify == "per-question":
                    verification_results = await run_per_question_dual_verification(
                        ctx.ai.providers, disagreements, language,
                        chat_manager=effective_cache_manager
                    )
                elif batch_verify == "per-copy":
                    from ai.batch_grader import run_per_copy_dual_verification
                    verification_results = await run_per_copy_dual_verification(
                        ctx.ai.providers, disagreements, language,
                        name_disagreements=name_disagreements if name_disagreements else None,
                        chat_manager=effective_cache_manager
                    )
                else:
                    extra_images = []
                    if name_disagreements and not disagreements:
                        for nd in name_disagreements:
                            for copy_result in batch_result.copies:
                                if copy_result.copy_index == nd['copy_index']:
                                    if hasattr(copy_result, 'image_paths'):
                                        extra_images.extend(copy_result.image_paths)
                                    break

                    verification_results = await run_dual_llm_verification(
                        ctx.ai.providers, disagreements, language,
                        name_disagreements=name_disagreements if name_disagreements else None,
                        extra_images=extra_images if extra_images else None,
                        chat_manager=effective_cache_manager
                    )

                # Apply verification results
                for copy_result in batch_result.copies:
                    for qid, qdata in copy_result.questions.items():
                        key = f"copy_{copy_result.copy_index}_{qid}"
                        if key in verification_results:
                            verified = verification_results[key]
                            qdata['grade'] = verified['final_grade']
                            qdata['reasoning'] = verified.get('llm1_reasoning', qdata.get('reasoning', ''))
                            if verified.get('final_feedback'):
                                qdata['feedback'] = verified['final_feedback']

                            comp_key = f"copy_{copy_result.copy_index}"
                            if comp_key in llm_comparison_data.get("llm_comparison", {}):
                                if qid in llm_comparison_data["llm_comparison"][comp_key].get("questions", {}):
                                    question_comp_data = llm_comparison_data["llm_comparison"][comp_key]["questions"][qid]
                                    llm1_data = question_comp_data.get(llm1_name, {})
                                    llm2_data = question_comp_data.get(llm2_name, {})

                                    resolved_max_points = verified.get('resolved_max_points', qdata.get('max_points', 1.0))
                                    initial_mp_disagreement = abs(llm1_data.get('max_points', 1.0) - llm2_data.get('max_points', 1.0)) > 0.01 if llm2_data else False

                                    qdata['max_points'] = resolved_max_points

                                    verification_data = {
                                        "llm1_new_grade": verified.get('llm1_new_grade'),
                                        "llm2_new_grade": verified.get('llm2_new_grade'),
                                        "llm1_reasoning": verified.get('llm1_reasoning', ''),
                                        "llm2_reasoning": verified.get('llm2_reasoning', ''),
                                        "llm1_feedback": verified.get('llm1_feedback', ''),
                                        "llm2_feedback": verified.get('llm2_feedback', ''),
                                        "final_feedback": verified.get('final_feedback', ''),
                                        "confidence": verified.get('confidence', 0.8),
                                        "method": verified.get('method', 'grouped')
                                    }

                                    llm1_mp = verified.get('llm1_new_max_points')
                                    llm2_mp = verified.get('llm2_new_max_points')
                                    mp_values_changed = (llm1_mp is not None and abs(llm1_mp - llm1_data.get('max_points', 1.0)) > 0.01) or \
                                                       (llm2_mp is not None and abs(llm2_mp - llm2_data.get('max_points', 1.0)) > 0.01)

                                    if initial_mp_disagreement or mp_values_changed or (llm1_mp is not None and llm2_mp is not None and abs(llm1_mp - llm2_mp) > 0.01):
                                        verification_data["llm1_new_max_points"] = llm1_mp if llm1_mp is not None else llm1_data.get('max_points', 1.0)
                                        verification_data["llm2_new_max_points"] = llm2_mp if llm2_mp is not None else llm2_data.get('max_points', 1.0)
                                        verification_data["resolved_max_points"] = resolved_max_points

                                    llm1_reading = verified.get('llm1_new_reading')
                                    llm2_reading = verified.get('llm2_new_reading')
                                    if llm1_reading or llm2_reading:
                                        verification_data["llm1_new_reading"] = llm1_reading
                                        verification_data["llm2_new_reading"] = llm2_reading

                                        if llm1_reading and llm2_reading:
                                            reading_similarity = SequenceMatcher(
                                                None,
                                                llm1_reading.lower().strip(),
                                                llm2_reading.lower().strip()
                                            ).ratio()
                                            verification_data["reading_similarity"] = round(reading_similarity, 2)
                                            verification_data["reading_disagreement"] = reading_similarity < 0.8

                                    llm_comparison_data["llm_comparison"][comp_key]["questions"][qid]["verification"] = verification_data

                                    llm_comparison_data["llm_comparison"][comp_key]["questions"][qid]["_pending_final"] = {
                                        "grade": verified['final_grade'],
                                        "max_points": resolved_max_points,
                                        "confidence": verified.get('confidence', 0.8),
                                        "reasoning": verified.get('llm1_reasoning', ''),
                                        "feedback": verified.get('final_feedback', ''),
                                        "method": f"verified_{verified.get('method', 'grouped')}"
                                    }

                # ===== NAME VERIFICATION RESULTS =====
                name_verification_results = {}
                persistent_name_disagreements = []
                if batch_verify in ("grouped", "per-copy") and name_disagreements:
                    for nd in name_disagreements:
                        copy_idx = nd['copy_index']
                        name_key = f"name_{copy_idx}"

                        if name_key in verification_results:
                            name_result = verification_results[name_key]
                            name_verification_results[copy_idx] = name_result

                            if name_result.get('agreement'):
                                for copy_result in batch_result.copies:
                                    if copy_result.copy_index == copy_idx:
                                        copy_result.student_name = name_result['resolved_name']
                                        break

                                comp_key = f"copy_{copy_idx}"
                                if comp_key not in llm_comparison_data.get("llm_comparison", {}):
                                    llm_comparison_data["llm_comparison"][comp_key] = {}
                                llm_comparison_data["llm_comparison"][comp_key]["student_name_resolution"] = {
                                    "verification": {
                                        "llm1_new_name": name_result.get('llm1_new_name'),
                                        "llm2_new_name": name_result.get('llm2_new_name'),
                                        "agreement": True
                                    },
                                    "final_resolved_name": name_result['resolved_name']
                                }
                            else:
                                persistent_name_disagreements.append({
                                    'copy_index': copy_idx,
                                    'llm1_name': name_result.get('llm1_new_name', nd['llm1_name']),
                                    'llm2_name': name_result.get('llm2_new_name', nd['llm2_name'])
                                })

                # ===== PHASE 2: Ultimatum Round =====
                persistent_disagreements = []
                for d in disagreements:
                    key = f"copy_{d.copy_index}_{d.question_id}"

                    if key in verification_results:
                        v = verification_results[key]
                        llm1_new = v.get('llm1_new_grade', d.llm1_grade)
                        llm2_new = v.get('llm2_new_grade', d.llm2_grade)
                        max_pts = v.get('max_points', d.max_points)
                        relative_threshold = max_pts * get_settings().grade_agreement_threshold
                        gap = abs(llm1_new - llm2_new)

                        llm1_initial = d.llm1_grade
                        llm2_initial = d.llm2_grade
                        grade_flip_flop = (llm1_new != llm1_initial and llm2_new != llm2_initial and
                                          abs(llm1_new - llm2_initial) < 0.01 and abs(llm2_new - llm1_initial) < 0.01)

                        frozen_max_pts = d.max_points

                        original_has_reading = hasattr(d, 'disagreement_type') and 'reading' in getattr(d, 'disagreement_type', '')

                        llm1_new_reading = v.get('llm1_new_reading', d.llm1_reading if hasattr(d, 'llm1_reading') else '')
                        llm2_new_reading = v.get('llm2_new_reading', d.llm2_reading if hasattr(d, 'llm2_reading') else '')
                        reading_still_differs = False
                        if llm1_new_reading and llm2_new_reading:
                            reading_similarity = SequenceMatcher(None, llm1_new_reading.lower(), llm2_new_reading.lower()).ratio()
                            reading_still_differs = reading_similarity < 0.8

                        is_persistent = (gap >= relative_threshold or
                                        grade_flip_flop or
                                        (original_has_reading and reading_still_differs))

                        if is_persistent:
                            persistent_disagreements.append({
                                'copy_index': d.copy_index,
                                'question_id': d.question_id,
                                'llm1_grade': llm1_new,
                                'llm2_grade': llm2_new,
                                'max_points': frozen_max_pts,
                                'llm1_initial_grade': llm1_initial,
                                'llm2_initial_grade': llm2_initial,
                                'llm1_reasoning': v.get('llm1_reasoning', d.llm1_reasoning),
                                'llm2_reasoning': v.get('llm2_reasoning', d.llm2_reasoning),
                                'image_paths': d.image_paths,
                                'flip_flop_detected': grade_flip_flop,
                                'reading_disagreement': (original_has_reading and reading_still_differs)
                            })

                # Build verification summary
                ultimatum_keys = set()
                ultimatum_reasons = {}
                for pd in persistent_disagreements:
                    key = f"copy_{pd['copy_index']}_{pd['question_id']}"
                    ultimatum_keys.add(key)
                    reasons = []
                    if pd.get('max_points_disagreement'):
                        reasons.append('barème')
                    if pd.get('flip_flop_detected'):
                        reasons.append('flip-flop')
                    if pd.get('reading_disagreement'):
                        reasons.append('lecture')
                    if pd.get('llm1_grade') is not None and pd.get('llm2_grade') is not None:
                        gap = abs(pd['llm1_grade'] - pd['llm2_grade'])
                        if gap >= pd.get('max_points', 1) * get_settings().grade_agreement_threshold:
                            reasons.append('notes')
                    ultimatum_reasons[key] = reasons if reasons else ['autre']

                verification_summary = {
                    'resolved_count': len([k for k in verification_results if k.startswith('copy_')]),
                    'questions': []
                }
                for key, verified in verification_results.items():
                    if key.startswith('copy_') and not key.startswith('name_'):
                        parts = key.split('_')
                        if len(parts) >= 3:
                            copy_idx = int(parts[1])
                            qid = '_'.join(parts[2:])
                            goes_to_ultimatum = key in ultimatum_keys
                            verification_summary['questions'].append({
                                'copy_index': copy_idx,
                                'question_id': qid,
                                'final_grade': verified.get('final_grade'),
                                'method': verified.get('method'),
                                'goes_to_ultimatum': goes_to_ultimatum,
                                'ultimatum_reasons': ultimatum_reasons.get(key, [])
                            })

                if progress_callback:
                    await self._call_callback(progress_callback, 'batch_verification_done', verification_summary)

                if persistent_disagreements:
                    logger.info(f"Persisting {len(persistent_disagreements)} disagreements after verification, running ultimatum round")

                    if progress_callback:
                        await self._call_callback(progress_callback, 'batch_ultimatum_start', {
                            'persistent_count': len(persistent_disagreements)
                        })

                    # Use explicit cache for ultimatum if available
                    if explicit_cache_mgr:
                        logger.info("Using EXPLICIT cache for ultimatum")
                    elif chat_manager:
                        logger.info("Using IMPLICIT cache for ultimatum")

                    if batch_verify == "per-question":
                        ultimatum_results = await run_per_question_dual_ultimatum(
                            ctx.ai.providers, persistent_disagreements, language,
                            chat_manager=effective_cache_manager
                        )
                    elif batch_verify == "per-copy":
                        from ai.batch_grader import run_per_copy_dual_ultimatum
                        ultimatum_results = await run_per_copy_dual_ultimatum(
                            ctx.ai.providers, persistent_disagreements, language,
                            chat_manager=effective_cache_manager
                        )
                    else:
                        ultimatum_results = await run_dual_llm_ultimatum(
                            ctx.ai.providers, persistent_disagreements, language,
                            chat_manager=effective_cache_manager
                        )

                    parse_status = ultimatum_results.pop('_parse_status', {})
                    llm1_parse_failed = parse_status.get('llm1_parse_failed', False)
                    llm2_parse_failed = parse_status.get('llm2_parse_failed', False)

                    if llm1_parse_failed or llm2_parse_failed:
                        parse_warning = []
                        if llm1_parse_failed:
                            parse_warning.append(llm1_name)
                        if llm2_parse_failed:
                            parse_warning.append(llm2_name)
                        warning_msg = f"Ultimatum parsing failed for: {', '.join(parse_warning)}"
                        logger.warning(warning_msg)

                        if progress_callback:
                            await self._call_callback(progress_callback, 'ultimatum_parse_warning', {
                                'warning': warning_msg,
                                'llm1_failed': llm1_parse_failed,
                                'llm2_failed': llm2_parse_failed
                            })

                        llm_comparison_data['ultimatum_parse_warning'] = {
                            'llm1_parse_failed': llm1_parse_failed,
                            'llm2_parse_failed': llm2_parse_failed,
                            'warning': warning_msg
                        }

                    # Apply ultimatum results
                    for copy_result in batch_result.copies:
                        for qid, qdata in copy_result.questions.items():
                            key = f"copy_{copy_result.copy_index}_{qid}"
                            if key in ultimatum_results:
                                ultimate = ultimatum_results[key]
                                qdata['grade'] = ultimate['final_grade']
                                qdata['reasoning'] = ultimate.get('llm1_reasoning', qdata.get('reasoning', ''))
                                if ultimate.get('final_feedback'):
                                    qdata['feedback'] = ultimate['final_feedback']
                                if ultimate.get('llm1_final_reading'):
                                    qdata['student_answer_read'] = ultimate['llm1_final_reading']

                                comp_key = f"copy_{copy_result.copy_index}"
                                if comp_key in llm_comparison_data.get("llm_comparison", {}):
                                    if qid in llm_comparison_data["llm_comparison"][comp_key].get("questions", {}):
                                        q_max_points = qdata.get('max_points', 1.0)

                                        grade_flip_flop = False
                                        max_points_flip_flop = False
                                        max_points_disagreement = False
                                        for pd in persistent_disagreements:
                                            if pd.get('copy_index') == copy_result.copy_index and pd.get('question_id') == qid:
                                                grade_flip_flop = pd.get('flip_flop_detected', False)
                                                max_points_flip_flop = pd.get('max_points_flip_flop', False)
                                                max_points_disagreement = pd.get('max_points_disagreement', False)
                                                break

                                        ultimatum_data = {
                                            "llm1_final_grade": ultimate.get('llm1_final_grade'),
                                            "llm2_final_grade": ultimate.get('llm2_final_grade'),
                                            "llm1_final_reading": ultimate.get('llm1_final_reading'),
                                            "llm2_final_reading": ultimate.get('llm2_final_reading'),
                                            "llm1_decision": ultimate.get('llm1_decision'),
                                            "llm2_decision": ultimate.get('llm2_decision'),
                                            "llm1_reasoning": ultimate.get('llm1_reasoning', ''),
                                            "llm2_reasoning": ultimate.get('llm2_reasoning', ''),
                                            "llm1_feedback": ultimate.get('llm1_feedback', ''),
                                            "llm2_feedback": ultimate.get('llm2_feedback', ''),
                                            "final_feedback": ultimate.get('final_feedback', ''),
                                            "llm1_parse_success": ultimate.get('llm1_parse_success', True),
                                            "llm2_parse_success": ultimate.get('llm2_parse_success', True),
                                            "grade_flip_flop": grade_flip_flop,
                                            "max_points_flip_flop": max_points_flip_flop,
                                            "max_points_disagreement": max_points_disagreement,
                                            "confidence": ultimate.get('confidence', 0.8),
                                            "method": ultimate.get('method', 'ultimatum_average')
                                        }

                                        if ultimate.get('max_points_disagreement'):
                                            ultimatum_data["llm1_final_max_points"] = ultimate.get('llm1_final_max_points')
                                            ultimatum_data["llm2_final_max_points"] = ultimate.get('llm2_final_max_points')
                                            ultimatum_data["resolved_max_points"] = ultimate.get('resolved_max_points')

                                        llm_comparison_data["llm_comparison"][comp_key]["questions"][qid]["ultimatum"] = ultimatum_data

                                        ultimatum_resolved_mp = ultimate.get('resolved_max_points', q_max_points)
                                        llm_comparison_data["llm_comparison"][comp_key]["questions"][qid]["_pending_final"] = {
                                            "grade": ultimate['final_grade'],
                                            "max_points": ultimatum_resolved_mp,
                                            "method": ultimate.get('method', 'ultimatum_average')
                                        }
                                        qdata['max_points'] = ultimatum_resolved_mp

                    # Build ultimatum summary
                    ultimatum_summary = {
                        'resolved_count': len(ultimatum_results),
                        'questions': []
                    }
                    for key, ultimate in ultimatum_results.items():
                        if key.startswith('copy_'):
                            parts = key.split('_')
                            if len(parts) >= 3:
                                copy_idx = int(parts[1])
                                qid = '_'.join(parts[2:])
                                ultimatum_summary['questions'].append({
                                    'copy_index': copy_idx,
                                    'question_id': qid,
                                    'llm1_final': ultimate.get('llm1_final_grade'),
                                    'llm2_final': ultimate.get('llm2_final_grade'),
                                    'final_grade': ultimate.get('final_grade'),
                                    'method': ultimate.get('method')
                                })

                    if progress_callback:
                        await self._call_callback(progress_callback, 'batch_ultimatum_done', ultimatum_summary)

                # ===== CLEANUP =====
                if chat_manager:
                    chat_manager.clear()
                    logger.info("Cleared implicit cache manager after ultimatum")

                # Cleanup explicit cache
                if explicit_cache_mgr:
                    await explicit_cache_mgr.cleanup()
                    logger.info("Cleared explicit cache manager after ultimatum")

                # ===== NAME ULTIMATUM (grouped mode) =====
                if batch_verify == "grouped" and persistent_name_disagreements:
                    from ai.batch_grader import run_student_name_ultimatum

                    logger.info(f"{len(persistent_name_disagreements)} name disagreements persist after verification, running ultimatum")

                    if progress_callback:
                        await self._call_callback(progress_callback, 'name_ultimatum_start', {
                            'persistent_count': len(persistent_name_disagreements)
                        })

                    name_disagreement_images = []
                    for copy_result in batch_result.copies:
                        for d in persistent_name_disagreements:
                            if d['copy_index'] == copy_result.copy_index:
                                if hasattr(copy_result, 'image_paths'):
                                    name_disagreement_images.extend(copy_result.image_paths)
                                break

                    name_ultimatum_results = await run_student_name_ultimatum(
                        ctx.ai.providers, persistent_name_disagreements, name_disagreement_images, language
                    )

                    if progress_callback:
                        await self._call_callback(progress_callback, 'name_ultimatum_done', {
                            'resolved_count': len(name_ultimatum_results)
                        })

                    for copy_idx, result in name_ultimatum_results.items():
                        resolved_name = result.get('resolved_name')

                        for copy_result in batch_result.copies:
                            if copy_result.copy_index == copy_idx:
                                copy_result.student_name = resolved_name
                                break

                        comp_key = f"copy_{copy_idx}"
                        if comp_key in llm_comparison_data.get("llm_comparison", {}):
                            if "student_name_resolution" not in llm_comparison_data["llm_comparison"][comp_key]:
                                llm_comparison_data["llm_comparison"][comp_key]["student_name_resolution"] = {}
                            llm_comparison_data["llm_comparison"][comp_key]["student_name_resolution"]["ultimatum"] = result
                            llm_comparison_data["llm_comparison"][comp_key]["student_name_resolution"]["final_resolved_name"] = resolved_name

                    workflow_state = ctx.get_orchestrator_attr('workflow_state')
                    is_auto_mode = workflow_state.auto_mode if workflow_state else False

                    for copy_idx, result in name_ultimatum_results.items():
                        if result.get('needs_user_resolution'):
                            if is_auto_mode:
                                llm1_conf = result.get('llm1_confidence', 0.5)
                                llm2_conf = result.get('llm2_confidence', 0.5)
                                llm1_final = result.get('llm1_final_name', '')
                                llm2_final = result.get('llm2_final_name', '')

                                if llm1_conf >= llm2_conf:
                                    resolved_name = llm1_final
                                else:
                                    resolved_name = llm2_final

                                logger.info(f"Auto-resolved name for copy {copy_idx}: {resolved_name} (auto-confirm mode)")
                            else:
                                resolved_name = await self._ask_student_name_resolution(result, llm1_name, llm2_name)

                            for copy_result in batch_result.copies:
                                if copy_result.copy_index == copy_idx:
                                    copy_result.student_name = resolved_name
                                    break

                            comp_key = f"copy_{copy_idx}"
                            if comp_key in llm_comparison_data.get("llm_comparison", {}):
                                llm_comparison_data["llm_comparison"][comp_key]["student_name_resolution"]["user_resolved_name"] = resolved_name
                                llm_comparison_data["llm_comparison"][comp_key]["student_name_resolution"]["final_resolved_name"] = resolved_name

            # ===== STUDENT NAME VERIFICATION (per-question mode only) =====
            if batch_verify == "per-question" and name_disagreements:
                from ai.batch_grader import run_student_name_verification, run_student_name_ultimatum

                logger.info(f"{len(name_disagreements)} student name disagreements detected, running verification")

                if progress_callback:
                    await self._call_callback(progress_callback, 'name_verification_start', {
                        'disagreements_count': len(name_disagreements)
                    })

                name_disagreement_images = []
                for copy_result in batch_result.copies:
                    for d in name_disagreements:
                        if d['copy_index'] == copy_result.copy_index:
                            if hasattr(copy_result, 'image_paths'):
                                name_disagreement_images.extend(copy_result.image_paths)
                            break

                name_verification_results = await run_student_name_verification(
                    ctx.ai.providers, name_disagreements, name_disagreement_images, language
                )

                persistent_name_disagreements = []
                for copy_idx, result in name_verification_results.items():
                    if not result.get('agreement'):
                        orig = next((d for d in name_disagreements if d['copy_index'] == copy_idx), None)
                        if orig:
                            persistent_name_disagreements.append({
                                'copy_index': copy_idx,
                                'llm1_name': result['llm1_new_name'],
                                'llm2_name': result['llm2_new_name']
                            })

                name_ultimatum_results = {}
                if persistent_name_disagreements:
                    logger.info(f"{len(persistent_name_disagreements)} name disagreements persist, running ultimatum")

                    if progress_callback:
                        await self._call_callback(progress_callback, 'name_ultimatum_start', {
                            'persistent_count': len(persistent_name_disagreements)
                        })

                    name_ultimatum_results = await run_student_name_ultimatum(
                        ctx.ai.providers, persistent_name_disagreements, name_disagreement_images, language
                    )

                    if progress_callback:
                        await self._call_callback(progress_callback, 'name_ultimatum_done', {
                            'resolved_count': len(name_ultimatum_results)
                        })

                all_name_results = {**name_verification_results, **name_ultimatum_results}
                for copy_idx, result in all_name_results.items():
                    resolved_name = result.get('resolved_name')

                    for copy_result in batch_result.copies:
                        if copy_result.copy_index == copy_idx:
                            copy_result.student_name = resolved_name
                            break

                    comp_key = f"copy_{copy_idx}"
                    if comp_key not in llm_comparison_data.get("llm_comparison", {}):
                        llm_comparison_data["llm_comparison"][comp_key] = {}

                    llm_comparison_data["llm_comparison"][comp_key]["student_name_resolution"] = {
                        "verification": {
                            "llm1_new_name": name_verification_results.get(copy_idx, {}).get('llm1_new_name'),
                            "llm2_new_name": name_verification_results.get(copy_idx, {}).get('llm2_new_name'),
                            "agreement": name_verification_results.get(copy_idx, {}).get('agreement')
                        },
                        "ultimatum": name_ultimatum_results.get(copy_idx) if copy_idx in name_ultimatum_results else None,
                        "final_resolved_name": resolved_name,
                        "needs_user_resolution": result.get('needs_user_resolution', False)
                    }

                workflow_state = ctx.get_orchestrator_attr('workflow_state')
                is_auto_mode = workflow_state.auto_mode if workflow_state else False

                for copy_idx, result in all_name_results.items():
                    if result.get('needs_user_resolution'):
                        if is_auto_mode:
                            llm1_conf = result.get('llm1_confidence', 0.5)
                            llm2_conf = result.get('llm2_confidence', 0.5)
                            llm1_final = result.get('llm1_final_name', '')
                            llm2_final = result.get('llm2_final_name', '')

                            if llm1_conf >= llm2_conf:
                                resolved_name = llm1_final
                            else:
                                resolved_name = llm2_final

                            logger.info(f"Auto-resolved name for copy {copy_idx}: {resolved_name} (auto-confirm mode)")
                        else:
                            resolved_name = await self._ask_student_name_resolution(result, llm1_name, llm2_name)

                        for copy_result in batch_result.copies:
                            if copy_result.copy_index == copy_idx:
                                copy_result.student_name = resolved_name
                                break

                        comp_key = f"copy_{copy_idx}"
                        if comp_key in llm_comparison_data.get("llm_comparison", {}):
                            llm_comparison_data["llm_comparison"][comp_key]["student_name_resolution"]["user_resolved_name"] = resolved_name

                if progress_callback:
                    await self._call_callback(progress_callback, 'name_verification_done', {
                        'resolved_count': len(name_verification_results)
                    })

            # ===== FINALIZE: Rebuild question data in correct order =====
            for copy_idx in range(len(batch_result.copies)):
                comp_key = f"copy_{copy_idx + 1}"
                if comp_key in llm_comparison_data.get("llm_comparison", {}):
                    questions_data = llm_comparison_data["llm_comparison"][comp_key].get("questions", {})
                    for qid, qdata in questions_data.items():
                        llm1_data = qdata.get(llm1_name)
                        llm2_data = qdata.get(llm2_name)
                        verification_data = qdata.get("verification")
                        ultimatum_data = qdata.get("ultimatum")

                        if "_pending_final" in qdata:
                            final_data = qdata["_pending_final"].copy()
                        elif "_initial_final" in qdata:
                            final_data = qdata["_initial_final"].copy()
                        else:
                            llm1_mp = llm1_data.get("max_points", 1.0) if llm1_data else 1.0
                            llm2_mp = llm2_data.get("max_points", 1.0) if llm2_data else 1.0
                            resolved_mp = qdata.get("max_points")
                            if resolved_mp is None:
                                resolved_mp = llm1_mp if llm1_mp == llm2_mp else max(llm1_mp, llm2_mp)
                            final_data = {
                                "grade": 0,
                                "max_points": resolved_mp,
                                "method": "unknown"
                            }

                        if "max_points" not in final_data:
                            final_data["max_points"] = qdata.get("max_points", 1.0)

                        rebuilt = {}
                        if llm1_data:
                            rebuilt[llm1_name] = llm1_data
                        if llm2_data:
                            rebuilt[llm2_name] = llm2_data
                        if verification_data:
                            rebuilt["verification"] = verification_data
                        if ultimatum_data:
                            rebuilt["ultimatum"] = ultimatum_data
                        rebuilt["final"] = final_data

                        llm_comparison_data["llm_comparison"][comp_key]["questions"][qid] = rebuilt

            else:
                # No disagreements - apply averaged grades now
                for copy_result in batch_result.copies:
                    for qid, qdata in copy_result.questions.items():
                        comp_key = f"copy_{copy_result.copy_index}"
                        if comp_key in llm_comparison_data.get("llm_comparison", {}):
                            if qid in llm_comparison_data["llm_comparison"][comp_key].get("questions", {}):
                                q_data = llm_comparison_data["llm_comparison"][comp_key]["questions"][qid]
                                final_data = q_data.get("_initial_final", {})
                                qdata['grade'] = final_data.get('grade', qdata.get('grade', 0))

        else:
            # Single LLM batch
            from ai.batch_grader import BatchGrader as AIBatchGrader
            grader = AIBatchGrader(ctx.ai)
            batch_result = await grader.grade_batch(copies_data, questions, language)

            if not batch_result.parse_success:
                logger.error(f"Batch grading failed: {batch_result.parse_errors}")
                return []

            provider_name = getattr(ctx.ai, 'model_name', 'single_llm')
            llm_comparison_data = {
                "options": {
                    "mode": "batch",
                    "providers": [provider_name],
                    "total_copies": total_copies
                },
                "llm_comparison": {}
            }

            for copy_result in batch_result.copies:
                copy_idx = copy_result.copy_index
                llm_comparison_data["llm_comparison"][f"copy_{copy_idx}"] = {
                    "student_name": copy_result.student_name,
                    "questions": {}
                }

                for qid, qdata in copy_result.questions.items():
                    llm_comparison_data["llm_comparison"][f"copy_{copy_idx}"]["questions"][qid] = {
                        provider_name: {
                            "grade": qdata.get('grade', 0),
                            "max_points": qdata.get('max_points', 1),
                            "question_text": qdata.get('question_text', ''),
                            "reading": qdata.get('student_answer_read', ''),
                            "reasoning": qdata.get('reasoning', ''),
                            "feedback": qdata.get('feedback', ''),
                            "confidence": qdata.get('confidence', 0.8)
                        },
                        "final": {
                            "grade": qdata.get('grade', 0),
                            "max_points": qdata.get('max_points', 1),
                            "method": "single_llm",
                            "agreement": None
                        }
                    }

        # Track student detection for audit
        student_detection_info = {
            'mode': 'multi_student_detection' if len(batch_result.copies) > len(ctx.session.copies) else 'standard',
            'input_copies': len(ctx.session.copies),
            'detected_copies': len(batch_result.copies),
            'students': [],
            'llm_detections': {}
        }

        if ctx.comparison_mode and llm1_result and llm2_result:
            llm1_name, llm2_name = provider_names[0], provider_names[1]
            student_detection_info['llm_detections'] = {
                llm1_name: [
                    {'copy_index': c.copy_index, 'student_name': c.student_name}
                    for c in llm1_result.copies
                ],
                llm2_name: [
                    {'copy_index': c.copy_index, 'student_name': c.student_name}
                    for c in llm2_result.copies
                ]
            }

        # If LLM detected more students than we have copies, create new CopyDocuments
        if len(batch_result.copies) > len(ctx.session.copies):
            logger.info(f"LLM detected {len(batch_result.copies)} students, creating additional copies")
            first_copy = ctx.session.copies[0] if ctx.session.copies else None
            for i in range(len(ctx.session.copies), len(batch_result.copies)):
                new_copy = CopyDocument(
                    pdf_path=first_copy.pdf_path if first_copy else None,
                    page_count=first_copy.page_count if first_copy else 0,
                    student_name=None,
                    language=first_copy.language if first_copy else 'fr'
                )
                ctx.session.copies.append(new_copy)
                logger.info(f"Created new copy {i+1} for detected student")
            total_copies = len(ctx.session.copies)

        # Add student detection info
        if llm_comparison_data:
            llm_comparison_data['student_detection'] = student_detection_info
        else:
            llm_comparison_data = {
                "options": {
                    "mode": "batch",
                    "providers": [getattr(ctx.ai, 'model_name', 'unknown')],
                    "total_copies": total_copies
                },
                "student_detection": student_detection_info,
                "llm_comparison": {}
            }

        # Convert batch results to GradedCopy objects
        for copy_result in batch_result.copies:
            copy_idx = copy_result.copy_index - 1
            if copy_idx < 0 or copy_idx >= len(ctx.session.copies):
                logger.warning(f"Copy index {copy_result.copy_index} out of bounds, skipping")
                continue

            original_copy = ctx.session.copies[copy_idx]

            detected_name = copy_result.student_name
            if detected_name and not original_copy.student_name:
                original_copy.student_name = detected_name

            copy_key = f"copy_{copy_result.copy_index}"
            if copy_key not in llm_comparison_data.get("llm_comparison", {}):
                llm_comparison_data["llm_comparison"][copy_key] = {"questions": {}}

            llm1_student_name = None
            llm2_student_name = None
            if ctx.comparison_mode and llm1_result and llm2_result:
                for c in llm1_result.copies:
                    if c.copy_index == copy_result.copy_index:
                        llm1_student_name = c.student_name
                        break
                for c in llm2_result.copies:
                    if c.copy_index == copy_result.copy_index:
                        llm2_student_name = c.student_name
                        break

            final_resolved_name = detected_name

            llm_comparison_data["llm_comparison"][copy_key]["student_detection"] = {
                'copy_index': copy_result.copy_index,
                'student_name': detected_name,
                'llm1_student_name': llm1_student_name,
                'llm2_student_name': llm2_student_name,
                'final_resolved_name': final_resolved_name,
                'pages': getattr(copy_result, 'pages', None)
            }

            if llm1_student_name and llm2_student_name:
                n1_normalized = llm1_student_name.lower().strip()
                n2_normalized = llm2_student_name.lower().strip()
                if n1_normalized != n2_normalized:
                    llm_comparison_data["llm_comparison"][copy_key]["student_detection"]["name_disagreement"] = {
                        "llm1_name": llm1_student_name,
                        "llm2_name": llm2_student_name,
                        "resolved_name": final_resolved_name,
                        "resolution_method": "llm1_as_base"
                    }

            student_detection_info['students'].append({
                'copy_index': copy_result.copy_index,
                'student_name': detected_name,
                'pages': getattr(copy_result, 'pages', None)
            })

            if progress_callback:
                await self._call_callback(progress_callback, 'copy_start', {
                    'copy_index': copy_result.copy_index,
                    'total_copies': total_copies,
                    'copy_id': original_copy.id,
                    'student_name': detected_name or original_copy.student_name,
                    'questions': list(copy_result.questions.keys())
                })

            # Build grades dict
            grades = {}
            detected_scale = {}

            for qid, qdata in copy_result.questions.items():
                grades[qid] = qdata.get('grade', 0)
                max_pts = float(qdata.get('max_points', 0))
                if max_pts > 0:
                    detected_scale[qid] = max_pts

            # Update grading scale from detection (via orchestrator)
            ctx.orchestrator.update_scale_from_detection(detected_scale)

            # Calculate max_score
            batch_max_score = sum(ctx.grading_scale.values()) if ctx.grading_scale else 0.0

            max_points_per_question = {}
            for qid in copy_result.questions.keys():
                max_points_per_question[qid] = ctx.grading_scale.get(qid, 0)

            # Get comparison data for this copy
            copy_comparison = None
            if llm_comparison_data and copy_key in llm_comparison_data.get("llm_comparison", {}):
                copy_data = llm_comparison_data["llm_comparison"][copy_key]

                student_detection = copy_data.get("student_detection", {})
                student_name_resolution = copy_data.get("student_name_resolution", {})
                student_name_initial = copy_data.get("student_name_initial", {})

                student_name_section = {}

                llm1_initial = student_name_initial.get("llm1_name")
                llm2_initial = student_name_initial.get("llm2_name")
                if llm1_initial or llm2_initial:
                    initial_agreement = False
                    if llm1_initial and llm2_initial:
                        initial_agreement = llm1_initial.lower().strip() == llm2_initial.lower().strip()
                    student_name_section["initial"] = {
                        "llm1_name": llm1_initial,
                        "llm2_name": llm2_initial,
                        "agreement": initial_agreement
                    }

                if student_name_resolution.get("verification"):
                    v = student_name_resolution["verification"]
                    student_name_section["verification"] = {
                        "llm1_new_name": v.get("llm1_new_name"),
                        "llm2_new_name": v.get("llm2_new_name"),
                        "agreement": v.get("agreement"),
                        "method": v.get("method", "verification")
                    }

                if student_name_resolution.get("ultimatum"):
                    u = student_name_resolution["ultimatum"]
                    student_name_section["ultimatum"] = {
                        "llm1_final_name": u.get("llm1_final_name"),
                        "llm2_final_name": u.get("llm2_final_name"),
                        "resolved_name": u.get("resolved_name"),
                        "agreement": u.get("agreement"),
                        "llm1_confidence": u.get("llm1_confidence"),
                        "llm2_confidence": u.get("llm2_confidence")
                    }

                if student_name_resolution.get("user_resolved_name"):
                    student_name_section["user_resolution"] = {
                        "resolved_name": student_name_resolution["user_resolved_name"]
                    }

                final_resolved = student_name_resolution.get("final_resolved_name") or student_detection.get("final_resolved_name")
                if final_resolved:
                    if student_name_resolution.get("user_resolved_name"):
                        method = "user_resolution"
                    elif student_name_resolution.get("ultimatum"):
                        method = "ultimatum"
                    elif student_name_resolution.get("verification", {}).get("agreement"):
                        method = "verification_consensus"
                    elif student_name_resolution.get("verification"):
                        method = "verification_average"
                    else:
                        method = "initial_consensus" if initial_agreement else "llm1_as_base"

                    student_name_section["final"] = {
                        "resolved_name": final_resolved,
                        "method": method
                    }

                if student_detection.get("copy_index"):
                    student_name_section["copy_index"] = student_detection["copy_index"]

                copy_comparison = {
                    "options": llm_comparison_data["options"],
                    "llm_comparison": {
                        copy_key: {
                            "questions": copy_data.get("questions", {})
                        }
                    },
                    "student_name": student_name_section
                }
            elif llm_comparison_data and "student_detection" in llm_comparison_data:
                copy_comparison = {
                    "options": llm_comparison_data.get("options", {}),
                    "student_detection": llm_comparison_data["student_detection"]
                }

            audit_provider_names = (
                llm_comparison_data.get("options", {}).get("providers")
                if llm_comparison_data
                else None
            )
            audit_mode = "dual" if audit_provider_names and len(audit_provider_names) > 1 else "single"

            graded = GradedCopy(
                copy_id=original_copy.id,
                grades=grades,
                total_score=sum(grades.values()),
                max_score=batch_max_score,
                confidence=0.5,
                feedback=copy_result.overall_feedback or "",
                max_points_by_question=max_points_per_question,
                grading_audit=build_audit_from_llm_comparison(
                    copy_comparison or {},
                    mode=audit_mode,
                    grading_method=ctx.grading_mode or "batch",
                    verification_mode=batch_verify or "none",
                    provider_names=audit_provider_names,
                    grading_scale=ctx.grading_scale
                ) if copy_comparison else None
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

            graded_copies.append(graded)
            ctx.session.graded_copies.append(graded)

            if progress_callback:
                final_questions = {}
                for qid in sorted(grades.keys(), key=lambda x: (int(x.replace('Q', '')) if x.replace('Q', '').isdigit() else 999)):
                    final_questions[qid] = {
                        'grade': grades[qid],
                        'max_points': max_points_per_question.get(qid, 1.0),
                        'confidence': graded.confidence_by_question.get(qid, 0.8)
                    }

                await self._call_callback(progress_callback, 'copy_done', {
                    'copy_index': copy_result.copy_index,
                    'total_copies': total_copies,
                    'copy_id': original_copy.id,
                    'student_name': copy_result.student_name,
                    'total_score': graded.total_score,
                    'max_score': graded.max_score,
                    'confidence': graded.confidence,
                    'final_questions': final_questions
                })

        # Save and notify completion
        self._save_sync()

        if progress_callback:
            await self._call_callback(progress_callback, 'batch_done', {
                'total_copies': total_copies,
                'graded_copies': len(graded_copies),
                'patterns': batch_result.patterns
            })

        return graded_copies

    # ── Helper methods ──

    async def _ask_max_points_resolution(self, disagreement: dict, llm1_name: str, llm2_name: str) -> float:
        """Resolve max_points disagreement (kept for backward compatibility)."""
        return disagreement.get('max_points', 1.0)

    async def _ask_student_name_resolution(self, disagreement: dict, llm1_name: str, llm2_name: str) -> str:
        """Ask user to resolve a student name disagreement after ultimatum."""
        from interaction.cli import console
        from rich.prompt import Prompt
        from rich.panel import Panel

        llm1_student = disagreement.get('llm1_final_name', '')
        llm2_student = disagreement.get('llm2_final_name', '')

        console.print(Panel(
            f"[bold yellow]Désaccord sur le nom de l'étudiant après ultimatum[/bold yellow]\n\n"
            f"[cyan]{llm1_name}[/cyan]: {llm1_student}\n"
            f"[magenta]{llm2_name}[/magenta]: {llm2_student}",
            title="Résolution requise",
            border_style="yellow"
        ))

        console.print(f"\nOptions:")
        console.print(f"  1. {llm1_name}: {llm1_student}")
        console.print(f"  2. {llm2_name}: {llm2_student}")
        console.print(f"  3. Personnalisé")

        choice = Prompt.ask(
            "Choisir le nom de l'étudiant",
            choices=["1", "2", "3"],
            default="1"
        )

        if choice == "1":
            return llm1_student
        elif choice == "2":
            return llm2_student
        else:
            custom = Prompt.ask("Entrer le nom", default=llm1_student)
            return custom
