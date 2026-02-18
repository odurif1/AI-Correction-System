"""
Comparison Provider for dual-LLM grading.

Runs two LLMs in parallel, compares results, and performs cross-verification
if there's a disagreement.

Audit structure provides complete traceability:
- Initial results from each LLM
- Verification prompts and responses
- Timing and token usage
- Confidence evolution
- Decision path taken

Stateless Architecture:
- Each call is independent (no state between calls)
- Context is explicit in prompts
- Images are re-sent for verification (fresh look)
- Previous reasoning is passed explicitly (no memorization)
"""

import asyncio
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class DisagreementRecord:
    """Record of a disagreement between two LLMs."""
    question_id: str
    question_text: str
    llm1_name: str
    llm1_result: Dict[str, Any]
    llm2_name: str
    llm2_result: Dict[str, Any]
    initial_difference: float
    after_cross_verification: Optional[Dict[str, float]] = None
    resolved: bool = False


@dataclass
class ReadingDisagreementRecord:
    """Record of a reading disagreement between two LLMs."""
    question_id: str
    question_text: str
    llm1_name: str
    llm1_reading: str
    llm2_name: str
    llm2_reading: str
    user_resolved: bool = False
    final_reading: str = ""


@dataclass
class GradingContext:
    """Context for a grading operation with all parameters."""
    question_text: str
    criteria: str
    image_path: Any
    max_points: float
    class_context: str = ""
    language: str = "fr"
    question_id: str = ""
    skip_reading_consensus: bool = False

    # Derived/computed fields
    effective_criteria: str = ""
    jurisprudence_context: str = ""

    # Timing
    start_time: float = 0.0


def build_llm_audit_info(
    result: Dict,
    provider_name: str,
    duration_ms: float = None,
    tokens: Dict = None,
    prompt_sent: str = None
) -> Dict:
    """
    Build complete audit info for an LLM result.

    Args:
        result: The LLM result dict
        provider_name: Name of the provider
        duration_ms: Time taken for the call
        tokens: Token usage {"prompt": int, "completion": int}
        prompt_sent: The prompt that was sent (for verification phases)

    Returns:
        Complete audit info dict
    """
    info = {
        "provider": provider_name,
        "grade": result.get("grade"),
        "confidence": result.get("confidence"),
        "internal_reasoning": result.get("internal_reasoning", ""),
        "student_feedback": result.get("student_feedback", ""),
        "student_answer_read": result.get("student_answer_read", "")
    }

    if duration_ms is not None:
        info["duration_ms"] = round(duration_ms, 1)

    if tokens:
        info["tokens"] = tokens

    if prompt_sent:
        info["prompt_sent"] = prompt_sent

    return info


class ComparisonProvider:
    """
    Wrapper provider that runs two LLMs in parallel and compares results.

    Workflow:
    1. Grade with both providers in parallel
    2. Compare grades
    3. If difference -> cross-verification
    4. If still different -> call disagreement_callback or average

    Usage:
        provider1 = GeminiProvider(...)
        provider2 = OpenAIProvider(...)
        comparison = ComparisonProvider([
            ("gemini", provider1),
            ("openai", provider2)
        ],
        disagreement_callback=my_callback  # async fn(...) -> tuple[float, str]
        )
        result = await comparison.grade_with_vision(...)
    """

    def __init__(
        self,
        providers: List[Tuple[str, Any]],
        disagreement_callback: callable = None,
        progress_callback: callable = None
    ):
        """
        Initialize comparison provider.

        Args:
            providers: List of (name, provider) tuples
                       Example: [("gemini", GeminiProvider()), ("openai", OpenAIProvider())]
            disagreement_callback: Optional async callback for disagreements
                                   async def callback(question_id, question_text, llm1_name, llm1_result, llm2_name, llm2_result, max_points) -> float
                                   Returns the chosen grade
            progress_callback: Optional callback for progress updates
                               async def callback(event_type, data)
        """
        if len(providers) < 2:
            raise ValueError("ComparisonProvider requires at least 2 providers")

        self.providers = providers
        self.disagreement_callback = disagreement_callback
        self.progress_callback = progress_callback
        self.disagreements: List[DisagreementRecord] = []
        self.comparison_data: Dict[str, Any] = {}
        self.jurisprudence: Dict[str, Any] = {}  # Store past decisions

    def set_progress_callback(self, callback: callable):
        """Set progress callback (can be set after initialization)."""
        self.progress_callback = callback

    def set_jurisprudence(self, jurisprudence: Dict[str, Any]):
        """Set jurisprudence data (past user decisions)."""
        self.jurisprudence = jurisprudence

    def get_jurisprudence(self) -> Dict[str, Any]:
        """Get current jurisprudence data."""
        return self.jurisprudence

    def get_token_usage(self) -> Dict[str, Any]:
        """Get total token usage from all providers."""
        total_prompt = 0
        total_completion = 0
        provider_usage = {}

        for name, provider in self.providers:
            if hasattr(provider, 'get_token_usage'):
                usage = provider.get_token_usage()
                provider_usage[name] = usage
                total_prompt += usage.get('prompt_tokens', 0)
                total_completion += usage.get('completion_tokens', 0)

        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "by_provider": provider_usage
        }

    @property
    def primary_provider(self) -> Any:
        """Return the first provider as primary."""
        return self.providers[0][1]

    def has_disagreements(self) -> bool:
        """Check if there are unresolved disagreements."""
        return any(not d.resolved for d in self.disagreements)

    def get_disagreements(self) -> List[DisagreementRecord]:
        """Get all unresolved disagreements."""
        return [d for d in self.disagreements if not d.resolved]

    async def _notify_progress(self, event_type: str, data: dict):
        """Safely call progress callback."""
        if self.progress_callback is None:
            return
        try:
            result = self.progress_callback(event_type, data)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass

    # ==================== GRADING HELPER METHODS ====================

    def _build_jurisprudence_context(
        self,
        question_id: str,
        question_text: str,
        max_points: float,
        language: str
    ) -> str:
        """Build jurisprudence context string from past decisions."""
        if not question_id or question_id not in self.jurisprudence:
            return ""

        past = self.jurisprudence[question_id]
        past_question = past.get('question_text', '')

        # Only use jurisprudence if it's for the SAME question text
        if not past_question or past_question.strip() != question_text.strip():
            return ""

        reasoning_hint = ""
        if past.get('reasoning_llm1'):
            reasoning_hint = f"\n- Raisonnement IA 1: {past['reasoning_llm1'][:150]}..."

        if language == "fr":
            return f"""

INFORMATION - Décision passée (à titre indicatif):
Pour cette même question "{question_id}", l'enseignant a précédemment décidé:
- Note attribuée: {past['decision']:.1f}/{past.get('max_points', max_points)}
- Notes proposées par les IA: {past.get('llm1_grade', '?')} vs {past.get('llm2_grade', '?')}
{reasoning_hint}
Cette information est fournie à titre de référence pour t'aider. Tu reste libre de ta notation.
"""
        else:
            return f"""

INFORMATION - Past decision (for reference only):
For this same question "{question_id}", the teacher previously decided:
- Grade given: {past['decision']:.1f}/{past.get('max_points', max_points)}
- AI proposed grades: {past.get('llm1_grade', '?')} vs {past.get('llm2_grade', '?')}
{reasoning_hint}
This information is provided as a reference to help you. You remain free in your grading.
"""

    async def _run_parallel_grading(
        self,
        question_text: str,
        criteria: str,
        image_path: Any,
        max_points: float,
        class_context: str,
        language: str,
        phase_timings: Dict[str, Dict]
    ) -> Tuple[List[Dict], List[float]]:
        """
        Run grading with both providers in parallel.

        Returns:
            Tuple of (results list, durations list)
        """
        completed_count = 0
        total_providers = len(self.providers)

        async def call_provider(index: int, name: str, provider):
            """Call a provider, handling both sync and async methods."""
            nonlocal completed_count
            call_start = time.time()
            try:
                result = provider.grade_with_vision(
                    question_text=question_text,
                    criteria=criteria,
                    image_path=image_path,
                    max_points=max_points,
                    class_context=class_context,
                    language=language
                )
                if asyncio.iscoroutine(result):
                    result = await result

                call_duration = (time.time() - call_start) * 1000
                completed_count += 1

                await self._notify_progress('llm_complete', {
                    'provider': name,
                    'provider_index': index,
                    'grade': result.get('grade'),
                    'confidence': result.get('confidence'),
                    'all_completed': completed_count == total_providers
                })

                phase_timings["initial"][f"llm{index+1}"] = round(call_duration, 1)
                return (index, result, call_duration)

            except Exception as e:
                call_duration = (time.time() - call_start) * 1000
                error_result = {
                    "grade": None,
                    "confidence": 0.0,
                    "internal_reasoning": f"Error: {str(e)}",
                    "student_feedback": "",
                    "error": str(e)
                }
                completed_count += 1
                await self._notify_progress('llm_error', {
                    'provider': name,
                    'provider_index': index,
                    'error': str(e)
                })
                phase_timings["initial"][f"llm{index+1}"] = round(call_duration, 1)
                return (index, error_result, call_duration)

        # Run all providers in parallel
        tasks = [call_provider(i, name, provider) for i, (name, provider) in enumerate(self.providers)]
        indexed_results = await asyncio.gather(*tasks)

        # Sort by index to maintain original order
        indexed_results.sort(key=lambda x: x[0])
        results = [r for _, r, _ in indexed_results]
        durations = [d for _, _, d in indexed_results]

        return results, durations

    def _build_reading_comparison(
        self,
        reading1: str,
        reading2: str
    ) -> Dict[str, Any]:
        """Build comparison dict for readings from both LLMs."""
        comparison = {
            "llm1_read": reading1,
            "llm2_read": reading2,
            "identical": reading1.strip().lower() == reading2.strip().lower() if reading1 and reading2 else False,
            "difference_type": None
        }

        if reading1 and reading2 and not comparison["identical"]:
            r1_lower, r2_lower = reading1.lower().strip(), reading2.lower().strip()
            if r1_lower.replace("é", "e").replace("è", "e") == r2_lower.replace("é", "e").replace("è", "e"):
                comparison["difference_type"] = "accent"
            elif r1_lower in r2_lower or r2_lower in r1_lower:
                comparison["difference_type"] = "partial"
            else:
                comparison["difference_type"] = "substantial"

        return comparison

    async def _handle_reading_reverification(
        self,
        results: List[Dict],
        image_path: Any,
        question_text: str,
        language: str,
        max_points: float,
        criteria: str,
        reading_comparison: Dict
    ) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Handle reading re-verification when readings differ substantially.

        Returns:
            Tuple of (reverification_info dict or None, updated results)
        """
        if reading_comparison.get("difference_type") != "substantial":
            return None, results

        reading1 = results[0].get("student_answer_read", "")
        reading2 = results[1].get("student_answer_read", "")

        reading_results = [
            {"reading": reading1, "grade": results[0].get("grade"), "confidence": results[0].get("confidence")},
            {"reading": reading2, "grade": results[1].get("grade"), "confidence": results[1].get("confidence")}
        ]

        start_time = time.time()
        verified_readings, reading_prompts = await self._cross_verify_reading(
            reading_results,
            image_path=image_path,
            question_text=question_text,
            language=language,
            max_points=max_points,
            grading_context=criteria
        )
        duration = (time.time() - start_time) * 1000

        new_reading1 = verified_readings[0].get("reading", reading1)
        new_reading2 = verified_readings[1].get("reading", reading2)
        new_grade1 = verified_readings[0].get("grade")
        new_grade2 = verified_readings[1].get("grade")

        reverification_info = {
            "llm1": {
                "initial_reading": reading1,
                "final_reading": new_reading1,
                "reading_changed": new_reading1.strip().lower() != reading1.strip().lower(),
                "initial_grade": results[0].get("grade"),
                "final_grade": new_grade1,
                "grade_changed": new_grade1 is not None and abs(new_grade1 - results[0].get("grade", 0)) > 0.01,
                "justification": verified_readings[0].get("justification", ""),
                "prompt_sent": reading_prompts.get("llm1"),
                "raw_response": verified_readings[0].get("raw_response", "")
            },
            "llm2": {
                "initial_reading": reading2,
                "final_reading": new_reading2,
                "reading_changed": new_reading2.strip().lower() != reading2.strip().lower(),
                "initial_grade": results[1].get("grade"),
                "final_grade": new_grade2,
                "grade_changed": new_grade2 is not None and abs(new_grade2 - results[1].get("grade", 0)) > 0.01,
                "justification": verified_readings[1].get("justification", ""),
                "prompt_sent": reading_prompts.get("llm2"),
                "raw_response": verified_readings[1].get("raw_response", "")
            },
            "duration_ms": round(duration, 1)
        }

        # Update results with new readings and grades
        updated_results = list(results)  # Copy
        if new_grade1 is not None:
            updated_results[0] = {**updated_results[0], "grade": new_grade1, "student_answer_read": new_reading1}
        if new_grade2 is not None:
            updated_results[1] = {**updated_results[1], "grade": new_grade2, "student_answer_read": new_reading2}

        return reverification_info, updated_results

    def _build_comparison_info(
        self,
        results: List[Dict],
        initial_durations: List[float],
        initial_grades: List[float],
        reading_comparison: Dict,
        pre_reading_consensus_result: Optional[Dict],
        reading_reverification: Optional[Dict],
        confidence_evolution: Dict,
        phase_timings: Dict,
        image_path: Any
    ) -> Dict[str, Any]:
        """Build the comprehensive comparison_info structure."""
        return {
            "initial": {
                "llm1": build_llm_audit_info(
                    results[0], self.providers[0][0],
                    duration_ms=initial_durations[0] if initial_durations else None
                ),
                "llm2": build_llm_audit_info(
                    results[1], self.providers[1][0],
                    duration_ms=initial_durations[1] if len(initial_durations) > 1 else None
                ),
                "difference": abs(initial_grades[0] - initial_grades[1]) if len(initial_grades) == 2 else None
            },
            "reading_comparison": reading_comparison,
            "pre_reading_consensus": pre_reading_consensus_result,
            "reading_reverification": reading_reverification,
            "confidence_evolution": confidence_evolution,
            "timing": {
                "initial": phase_timings["initial"],
                "reading_reverification": reading_reverification.get("duration_ms") if reading_reverification else None,
                "verification": None,
                "ultimatum": None,
                "total_ms": None
            },
            "decision_path": {
                "initial_agreement": len(initial_grades) == 2 and initial_grades[0] == initial_grades[1],
                "reading_reverification_triggered": reading_reverification is not None,
                "verification_triggered": False,
                "ultimatum_triggered": False,
                "final_method": None
            },
            "images": {
                "count": len(image_path) if isinstance(image_path, list) else 1 if image_path else 0,
                "paths": image_path if isinstance(image_path, list) else [image_path] if image_path else []
            },
            "after_cross_verification": None,
            "after_ultimatum": None,
            "final": None
        }

    async def grade_with_vision(
        self,
        question_text: str,
        criteria: str,
        image_path: Any,
        max_points: float,
        class_context: str = "",
        language: str = "fr",
        question_id: str = "",
        reading_disagreement_callback: callable = None,
        skip_reading_consensus: bool = False
    ) -> Dict[str, Any]:
        """
        Grade with both providers and compare results.

        Two-phase approach:
        - Phase 1 (optional): Establish reading consensus
        - Phase 2: Grade with both LLMs in parallel
        - Phase 3: Cross-verification if disagreement
        - Phase 4: Ultimatum round if still disagreeing

        Args:
            question_text: The question being graded
            criteria: Grading criteria
            image_path: Path(s) to image(s)
            max_points: Maximum points
            class_context: Context about class patterns
            language: Language for prompts
            question_id: Question identifier (for jurisprudence lookup)
            reading_disagreement_callback: Callback for reading disagreements
            skip_reading_consensus: If True, skip reading consensus phase

        Returns:
            Merged result with comparison data
        """
        total_start_time = time.time()
        phase_timings = {"initial": {}, "verification": {}, "ultimatum": {}}

        # ===== PHASE 1: Pre-reading consensus (optional) =====
        established_reading = None
        pre_reading_consensus_result = None

        if not skip_reading_consensus and reading_disagreement_callback is not None:
            reading_result = await self.read_student_answer_with_consensus(
                image_path=image_path,
                question_text=question_text,
                language=language,
                reading_disagreement_callback=reading_disagreement_callback
            )
            established_reading = reading_result.get("reading")
            pre_reading_consensus_result = reading_result.get("comparison")

            if reading_result.get("user_validated") and established_reading:
                if language == "fr":
                    criteria += f"\n\n─── LECTURE VALIDÉE PAR L'ENSEIGNANT ───\nL'élève a écrit: {established_reading}"
                else:
                    criteria += f"\n\n─── TEACHER-VALIDATED READING ───\nThe student wrote: {established_reading}"

        # ===== PHASE 2: Build context and run parallel grading =====
        jurisprudence_context = self._build_jurisprudence_context(
            question_id, question_text, max_points, language
        )
        effective_criteria = criteria + jurisprudence_context

        await self._notify_progress('llm_parallel_start', {
            'providers': [name for name, _ in self.providers],
            'question_text': question_text[:50] + '...' if len(question_text) > 50 else question_text
        })

        results, initial_durations = await self._run_parallel_grading(
            question_text=question_text,
            criteria=effective_criteria,
            image_path=image_path,
            max_points=max_points,
            class_context=class_context,
            language=language,
            phase_timings=phase_timings
        )

        # ===== PHASE 3: Analyze results =====
        grades = [r.get("grade", 0) for r in results if r.get("grade") is not None]
        initial_grades = list(grades)

        reading1 = results[0].get("student_answer_read", "")
        reading2 = results[1].get("student_answer_read", "")
        reading_comparison = self._build_reading_comparison(reading1, reading2)

        confidence_evolution = {
            "initial": {
                "llm1": results[0].get("confidence"),
                "llm2": results[1].get("confidence")
            }
        }

        # ===== PHASE 4: Reading re-verification (if needed) =====
        reading_reverification, results = await self._handle_reading_reverification(
            results=results,
            image_path=image_path,
            question_text=question_text,
            language=language,
            max_points=max_points,
            criteria=criteria,
            reading_comparison=reading_comparison
        )

        # Recalculate grades after potential re-verification
        grades = [r.get("grade", 0) for r in results if r.get("grade") is not None]

        # ===== BUILD AUDIT STRUCTURE =====
        comparison_info = self._build_comparison_info(
            results=results,
            initial_durations=initial_durations,
            initial_grades=initial_grades,
            reading_comparison=reading_comparison,
            pre_reading_consensus_result=pre_reading_consensus_result,
            reading_reverification=reading_reverification,
            confidence_evolution=confidence_evolution,
            phase_timings=phase_timings,
            image_path=image_path
        )

        # ===== PHASE 5: Check for agreement =====
        if len(grades) == 2 and grades[0] == grades[1]:
            comparison_info["decision_path"]["final_method"] = "consensus"
            comparison_info["timing"]["total_ms"] = round((time.time() - total_start_time) * 1000, 1)
            comparison_info["final"] = {"grade": grades[0], "agreement": True, "method": "consensus"}
            return self._merge_results(results, comparison_info)

        # ===== PHASE 6: Cross-verification =====
        comparison_info["decision_path"]["verification_triggered"] = True
        verification_start = time.time()

        if len(grades) == 2 and grades[0] != grades[1]:
            verified_results, verification_prompts = await self._run_cross_verification(
                results,
                question_text=question_text,
                criteria=criteria,
                image_path=image_path,
                max_points=max_points,
                class_context=class_context,
                language=language
            )
            verification_duration = (time.time() - verification_start) * 1000
            verified_grades = [r.get("grade", 0) for r in verified_results if r.get("grade") is not None]

            confidence_evolution["after_cross_verification"] = {
                "llm1": verified_results[0].get("confidence"),
                "llm2": verified_results[1].get("confidence")
            }

            comparison_info["after_cross_verification"] = {
                "llm1": build_llm_audit_info(verified_results[0], self.providers[0][0],
                                             prompt_sent=verification_prompts.get("llm1")),
                "llm2": build_llm_audit_info(verified_results[1], self.providers[1][0],
                                             prompt_sent=verification_prompts.get("llm2")),
                "difference": abs(verified_grades[0] - verified_grades[1]) if len(verified_grades) == 2 else None
            }
            comparison_info["timing"]["verification"] = {"total_ms": round(verification_duration, 1)}

            # ===== PHASE 7: Ultimatum round (if still disagreeing) =====
            if len(verified_grades) == 2 and verified_grades[0] != verified_grades[1]:
                comparison_info["decision_path"]["ultimatum_triggered"] = True
                ultimatum_start = time.time()

                final_results, ultimatum_prompts = await self._run_ultimatum_round(
                    verified_results, results,
                    question_text=question_text,
                    criteria=criteria,
                    image_path=image_path,
                    max_points=max_points,
                    class_context=class_context,
                    language=language
                )
                ultimatum_duration = (time.time() - ultimatum_start) * 1000
                final_grades = [r.get("grade", 0) for r in final_results if r.get("grade") is not None]

                confidence_evolution["after_ultimatum"] = {
                    "llm1": final_results[0].get("confidence"),
                    "llm2": final_results[1].get("confidence")
                }

                comparison_info["after_ultimatum"] = {
                    "llm1": build_llm_audit_info(final_results[0], self.providers[0][0],
                                                 prompt_sent=ultimatum_prompts.get("llm1")),
                    "llm2": build_llm_audit_info(final_results[1], self.providers[1][0],
                                                 prompt_sent=ultimatum_prompts.get("llm2")),
                    "difference": abs(final_grades[0] - final_grades[1]) if len(final_grades) == 2 else None
                }
                comparison_info["timing"]["ultimatum"] = {"total_ms": round(ultimatum_duration, 1)}

                verified_results = final_results
                verified_grades = final_grades

            # ===== PHASE 8: Final decision =====
            if len(verified_grades) == 2 and verified_grades[0] != verified_grades[1]:
                # Persistent disagreement
                comparison_info["decision_path"]["final_method"] = "average"
                comparison_info["timing"]["total_ms"] = round((time.time() - total_start_time) * 1000, 1)
                comparison_info["final"] = {
                    "grade": sum(verified_grades) / len(verified_grades),
                    "agreement": False,
                    "method": "average"
                }

                disagreement = DisagreementRecord(
                    question_id=question_id,
                    question_text=question_text,
                    llm1_name=self.providers[0][0],
                    llm1_result=verified_results[0],
                    llm2_name=self.providers[1][0],
                    llm2_result=verified_results[1],
                    initial_difference=comparison_info["initial"]["difference"],
                    after_cross_verification={"llm1": verified_grades[0], "llm2": verified_grades[1]},
                    resolved=False
                )
                self.disagreements.append(disagreement)

                # Call user callback if provided
                if self.disagreement_callback:
                    callback_result = await self.disagreement_callback(
                        question_id=question_id,
                        question_text=question_text,
                        llm1_name=self.providers[0][0],
                        llm1_result=verified_results[0],
                        llm2_name=self.providers[1][0],
                        llm2_result=verified_results[1],
                        max_points=max_points
                    )
                    chosen_grade, feedback_source = callback_result
                    disagreement.resolved = True
                    comparison_info["user_choice"] = chosen_grade
                    comparison_info["feedback_source"] = feedback_source
                    comparison_info["final"] = {
                        "grade": chosen_grade,
                        "agreement": False,
                        "method": "user_choice",
                        "feedback_source": feedback_source
                    }
                    return self._merge_results_with_user_choice(
                        verified_results, comparison_info, chosen_grade, feedback_source
                    )
            else:
                # Agreement reached after verification
                comparison_info["decision_path"]["final_method"] = "verification_consensus"
                comparison_info["timing"]["total_ms"] = round((time.time() - total_start_time) * 1000, 1)
                comparison_info["final"] = {
                    "grade": verified_grades[0] if len(verified_grades) > 0 else 0,
                    "agreement": True,
                    "method": "verification_consensus"
                }

            return self._merge_results(verified_results, comparison_info)

        # ===== FALLBACK =====
        comparison_info["timing"]["total_ms"] = round((time.time() - total_start_time) * 1000, 1)

        if comparison_info.get("final") is None:
            valid_result = next((r for r in results if r.get("grade") is not None), results[0])
            comparison_info["final"] = {
                "grade": valid_result.get("grade", 0),
                "agreement": True,
                "method": "single_valid" if len(grades) < 2 else "consensus"
            }
            if len(grades) < 2:
                comparison_info["decision_path"]["final_method"] = "single_valid"

        return self._merge_results(results, comparison_info)

    async def _run_cross_verification(
        self,
        results: List[Dict],
        **kwargs
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        First verification round: each LLM sees the other's reasoning.

        Returns both the verified results AND the prompts that were sent.

        Args:
            results: Initial grading results from both LLMs
            **kwargs: Original grading arguments

        Returns:
            Tuple of (verified_results, prompts_sent)
        """
        verified = []
        prompts_sent = {"llm1": None, "llm2": None}

        for i, (name, provider) in enumerate(self.providers):
            other_result = results[1 - i]
            other_reasoning = other_result.get("internal_reasoning", "")
            other_grade = other_result.get("grade", 0)
            other_answer_read = other_result.get("student_answer_read", "")
            my_grade = results[i].get("grade", 0)
            my_answer_read = results[i].get("student_answer_read", "")

            language = kwargs.get("language", "fr")

            # Build verification prompt - force independent verification
            if language == "fr":
                verify_prompt = f"""─── CONTESTATION ───
Un autre correcteur a noté {other_grade}/{kwargs.get('max_points', 5)} (tu as mis {my_grade}/{kwargs.get('max_points', 5)}).

Son raisonnement: {other_reasoning[:500]}

─── RÉEXAMEN INDÉPENDANT ───
1. RÉEXAMINE la réponse de l'élève - ne te fie PAS à son avis
2. Analyse OBJECTIVEMENT ce qui est correct et ce qui ne l'est pas
3. Décide TOI-MÊME de ta note finale

RÈGLES:
- Ne change ta note QUE si tu identifies toi-même une erreur dans ton analyse
- Si tu maintiens ta note: justifie par des arguments précis
- Si tu changes: explique ce que tu as constaté de nouveau
- SI INCERTAIN: abaisse ta CONFIANCE (< 0.5) pour signaler le doute

INTERDICTION: Ne change pas ta note juste parce que l'autre dit autre chose.

Question originale: {kwargs.get('question_text', '')}"""
            else:
                verify_prompt = f"""─── DISAGREEMENT ───
Another grader gave {other_grade}/{kwargs.get('max_points', 5)} (you gave {my_grade}/{kwargs.get('max_points', 5)}).

Their reasoning: {other_reasoning[:500]}

─── INDEPENDENT RE-EXAMINATION ───
1. RE-EXAMINE the student's answer - do NOT trust their opinion
2. Analyze OBJECTIVELY what is correct and what is not
3. Decide YOURSELF on your final grade

RULES:
- Only change your grade if YOU identify an error in your own analysis
- If you maintain: justify with precise arguments
- If you change: explain what you have now observed
- IF UNCERTAIN: lower your CONFIDENCE (< 0.5) to signal doubt

FORBIDDEN: Do not change your grade just because the other says so.

Original question: {kwargs.get('question_text', '')}"""

            # Store the prompt
            prompts_sent[f"llm{i+1}"] = verify_prompt.strip()

            try:
                new_result = provider.grade_with_vision(
                    question_text=kwargs.get("question_text", ""),
                    criteria=verify_prompt,
                    image_path=kwargs.get("image_path"),
                    max_points=kwargs.get("max_points", 5),
                    class_context=kwargs.get("class_context", ""),
                    language=language
                )
                if asyncio.iscoroutine(new_result):
                    new_result = await new_result
                verified.append(new_result)
            except Exception as e:
                verified.append(results[i])

        return verified, prompts_sent

    # ==================== UNIFIED VERIFICATION METHODS ====================

    async def _run_unified_verification(
        self,
        questions: List[Dict[str, Any]],
        flagged_questions: List[Any],  # QuestionDisagreement objects
        name_disagreement: Optional[Any],  # NameDisagreement object
        single_pass_results: Dict[str, Any],  # SinglePassResult by provider name
        image_paths: List[str],
        language: str
    ) -> Tuple[Dict[str, Dict], Optional[str], Dict[str, Any]]:
        """
        Run unified verification for ALL disagreements in ONE call per LLM.

        This replaces the per-question verification loop with a single unified
        call that handles all disagreements (name + questions) at once.

        Args:
            questions: List of all question definitions
            flagged_questions: List of QuestionDisagreement objects from analyzer
            name_disagreement: Optional NameDisagreement object
            single_pass_results: Dict mapping provider_name -> SinglePassResult
            image_paths: List of image paths to RE-SEND
            language: Language for prompts

        Returns:
            Tuple of:
            - Dict of question_id -> verified result
            - Optional consensus student name (or None if unchanged)
            - Audit info dict
        """
        import logging
        from config.prompts import build_unified_verification_prompt

        provider_names = [name for name, _ in self.providers]

        # Build disagreements list for prompt
        disagreements = []
        for d in flagged_questions:
            qid = d.question_id
            sp_llm1 = single_pass_results[provider_names[0]].questions.get(qid)
            sp_llm2 = single_pass_results[provider_names[1]].questions.get(qid)

            disagreements.append({
                "question_id": qid,
                "llm1": {
                    "grade": sp_llm1.grade if sp_llm1 else 0,
                    "reading": sp_llm1.student_answer_read if sp_llm1 else "",
                    "confidence": sp_llm1.confidence if sp_llm1 else 0.5,
                    "max_points": sp_llm1.max_points if sp_llm1 else 1.0
                },
                "llm2": {
                    "grade": sp_llm2.grade if sp_llm2 else 0,
                    "reading": sp_llm2.student_answer_read if sp_llm2 else "",
                    "confidence": sp_llm2.confidence if sp_llm2 else 0.5,
                    "max_points": sp_llm2.max_points if sp_llm2 else 1.0
                },
                "type": d.disagreement_type.value,
                "reason": d.reason
            })

        # Build name disagreement dict if present
        name_dis_dict = None
        if name_disagreement:
            name_dis_dict = {
                "llm1_name": name_disagreement.llm1_name,
                "llm2_name": name_disagreement.llm2_name,
                "similarity": name_disagreement.similarity
            }

        # Build unified prompt
        unified_prompt = build_unified_verification_prompt(
            questions=questions,
            disagreements=disagreements,
            name_disagreement=name_dis_dict,
            language=language
        )

        # Call both providers in parallel
        results_per_provider = {}
        prompts_sent = {}

        async def call_provider(idx: int, name: str, provider):
            try:
                # Use the provider's grade_with_vision method with the unified prompt
                result = provider.grade_with_vision(
                    question_text="Vérification unifiée de tous les désaccords",
                    criteria=unified_prompt,
                    image_path=image_paths,
                    max_points=10,  # Not used for unified prompt
                    language=language
                )
                if asyncio.iscoroutine(result):
                    result = await result
                return (idx, name, result)
            except Exception as e:
                logging.error(f"Unified verification failed for {name}: {e}")
                return (idx, name, None)

        # Run both providers in parallel
        tasks = [call_provider(i, name, provider) for i, (name, provider) in enumerate(self.providers)]
        provider_results = await asyncio.gather(*tasks)

        for idx, name, result in provider_results:
            results_per_provider[name] = result
            prompts_sent[f"llm{idx+1}"] = unified_prompt

        # Parse JSON responses
        verified_results = {}
        consensus_name = None
        remaining_disagreements = []

        for qid in [d["question_id"] for d in disagreements]:
            llm1_result = results_per_provider.get(provider_names[0], {})
            llm2_result = results_per_provider.get(provider_names[1], {})

            # Extract question results from JSON response
            llm1_questions = llm1_result.get("questions", {}) if isinstance(llm1_result, dict) else {}
            llm2_questions = llm2_result.get("questions", {}) if isinstance(llm2_result, dict) else {}

            q1 = llm1_questions.get(qid, {})
            q2 = llm2_questions.get(qid, {})

            # Get single pass results for fallback
            sp_llm1 = single_pass_results[provider_names[0]].questions.get(qid)
            sp_llm2 = single_pass_results[provider_names[1]].questions.get(qid)

            # Use parsed results or fallback to single pass
            grade1 = q1.get("grade") if q1.get("grade") is not None else (sp_llm1.grade if sp_llm1 else 0)
            grade2 = q2.get("grade") if q2.get("grade") is not None else (sp_llm2.grade if sp_llm2 else 0)

            reading1 = q1.get("student_answer_read", "") or (sp_llm1.student_answer_read if sp_llm1 else "")
            reading2 = q2.get("student_answer_read", "") or (sp_llm2.student_answer_read if sp_llm2 else "")

            confidence1 = q1.get("confidence", 0.5) or (sp_llm1.confidence if sp_llm1 else 0.5)
            confidence2 = q2.get("confidence", 0.5) or (sp_llm2.confidence if sp_llm2 else 0.5)

            max_pts = q1.get("max_points") or (sp_llm1.max_points if sp_llm1 else 1.0)

            # Check for agreement
            if abs(grade1 - grade2) < 0.1:
                # Agreement reached
                verified_results[qid] = {
                    "grade": grade1,
                    "max_points": max_pts,
                    "confidence": (confidence1 + confidence2) / 2,
                    "student_answer_read": reading1,
                    "student_feedback": q1.get("feedback", ""),
                    "internal_reasoning": q1.get("reasoning", ""),
                    "method": "unified_consensus"
                }
            else:
                # Still disagreeing - store for ultimatum
                remaining_disagreements.append({
                    "question_id": qid,
                    "llm1": {"grade": grade1, "reading": reading1, "confidence": confidence1, "max_points": max_pts},
                    "llm2": {"grade": grade2, "reading": reading2, "confidence": confidence2, "max_points": max_pts}
                })
                # Use average for now (will be updated by ultimatum if needed)
                verified_results[qid] = {
                    "grade": (grade1 + grade2) / 2,
                    "max_points": max_pts,
                    "confidence": min(confidence1, confidence2),
                    "student_answer_read": reading1,
                    "student_feedback": q1.get("feedback", ""),
                    "internal_reasoning": q1.get("reasoning", ""),
                    "method": "unified_averaged",
                    "_needs_ultimatum": True
                }

        # Handle name consensus
        if name_disagreement:
            llm1_result = results_per_provider.get(provider_names[0], {})
            llm2_result = results_per_provider.get(provider_names[1], {})

            name1 = llm1_result.get("student_name") if isinstance(llm1_result, dict) else None
            name2 = llm2_result.get("student_name") if isinstance(llm2_result, dict) else None

            if name1 and name2:
                # Normalize and compare
                n1_norm = name1.strip().lower()
                n2_norm = name2.strip().lower()
                if n1_norm == n2_norm:
                    consensus_name = name1.strip()
                else:
                    # Still disagree - keep the first one
                    consensus_name = name_disagreement.llm1_name
            else:
                consensus_name = name_disagreement.llm1_name

        # Build audit info
        audit_info = {
            "type": "unified",
            "prompts_sent": prompts_sent,
            "questions_verified": list(verified_results.keys()),
            "name_verified": name_disagreement is not None,
            "remaining_disagreements": [d["question_id"] for d in remaining_disagreements],
            "phases": 1
        }

        # Store remaining disagreements for ultimatum
        self._unified_remaining = remaining_disagreements
        self._unified_results = results_per_provider
        self._unified_evolution = {d["question_id"]: [(d["llm1"]["grade"], d["llm2"]["grade"])] for d in disagreements}

        return verified_results, consensus_name, audit_info

    async def _run_unified_ultimatum(
        self,
        questions: List[Dict[str, Any]],
        remaining_disagreements: List[Dict[str, Any]],
        evolution: Dict[str, List[Tuple[float, float]]],
        name_disagreement: Optional[Any],
        verification_results: Dict[str, Dict],
        image_paths: List[str],
        language: str
    ) -> Tuple[Dict[str, Dict], Optional[str], Dict[str, Any]]:
        """
        Run unified ultimatum for remaining disagreements after verification.

        This is the final phase when disagreement persists after unified verification.

        Args:
            questions: List of all question definitions
            remaining_disagreements: List of disagreements still unresolved
            evolution: Dict mapping question_id -> list of (llm1_grade, llm2_grade) tuples
            name_disagreement: Optional NameDisagreement object (if still unresolved)
            verification_results: Results from unified verification phase
            image_paths: List of image paths to RE-SEND
            language: Language for prompts

        Returns:
            Tuple of:
            - Dict of question_id -> final result
            - Optional final student name
            - Audit info dict
        """
        import logging
        from config.prompts import build_unified_ultimatum_prompt

        provider_names = [name for name, _ in self.providers]

        # Build disagreements list for ultimatum prompt
        disagreements = []
        for d in remaining_disagreements:
            qid = d["question_id"]
            q_evolution = evolution.get(qid, [])

            # Get the latest grades (from verification phase)
            if len(q_evolution) >= 1:
                latest = q_evolution[-1]
            else:
                latest = (d["llm1"]["grade"], d["llm2"]["grade"])

            disagreements.append({
                "question_id": qid,
                "llm1": {
                    "grade": latest[0],
                    "reading": d["llm1"]["reading"],
                    "confidence": d["llm1"]["confidence"],
                    "max_points": d["llm1"]["max_points"]
                },
                "llm2": {
                    "grade": latest[1],
                    "reading": d["llm2"]["reading"],
                    "confidence": d["llm2"]["confidence"],
                    "max_points": d["llm2"]["max_points"]
                }
            })

        # Build name disagreement dict if present
        name_dis_dict = None
        if name_disagreement:
            name_dis_dict = {
                "llm1_name": name_disagreement.llm1_name,
                "llm2_name": name_disagreement.llm2_name
            }

        # Build evolution dict for prompt
        evolution_dict = {}
        for d in remaining_disagreements:
            qid = d["question_id"]
            evo = evolution.get(qid, [])
            evolution_dict[qid] = evo

        # Build unified ultimatum prompt
        ultimatum_prompt = build_unified_ultimatum_prompt(
            questions=questions,
            disagreements=disagreements,
            evolution=evolution_dict,
            name_disagreement=name_dis_dict,
            language=language
        )

        # Call both providers in parallel
        results_per_provider = {}
        prompts_sent = {}

        async def call_provider(idx: int, name: str, provider):
            try:
                result = provider.grade_with_vision(
                    question_text="Ultimatum unifié - décision finale",
                    criteria=ultimatum_prompt,
                    image_path=image_paths,
                    max_points=10,
                    language=language
                )
                if asyncio.iscoroutine(result):
                    result = await result
                return (idx, name, result)
            except Exception as e:
                logging.error(f"Unified ultimatum failed for {name}: {e}")
                return (idx, name, None)

        tasks = [call_provider(i, name, provider) for i, (name, provider) in enumerate(self.providers)]
        provider_results = await asyncio.gather(*tasks)

        for idx, name, result in provider_results:
            results_per_provider[name] = result
            prompts_sent[f"llm{idx+1}"] = ultimatum_prompt

        # Parse results and determine final outcomes
        final_results = {}
        final_name = None

        for d in remaining_disagreements:
            qid = d["question_id"]
            llm1_result = results_per_provider.get(provider_names[0], {})
            llm2_result = results_per_provider.get(provider_names[1], {})

            llm1_questions = llm1_result.get("questions", {}) if isinstance(llm1_result, dict) else {}
            llm2_questions = llm2_result.get("questions", {}) if isinstance(llm2_result, dict) else {}

            q1 = llm1_questions.get(qid, {})
            q2 = llm2_questions.get(qid, {})

            grade1 = q1.get("grade") if q1.get("grade") is not None else d["llm1"]["grade"]
            grade2 = q2.get("grade") if q2.get("grade") is not None else d["llm2"]["grade"]

            confidence1 = q1.get("confidence", 0.5) or d["llm1"]["confidence"]
            confidence2 = q2.get("confidence", 0.5) or d["llm2"]["confidence"]

            max_pts = d["llm1"]["max_points"]

            if abs(grade1 - grade2) < 0.1:
                # Consensus reached in ultimatum
                final_results[qid] = {
                    "grade": grade1,
                    "max_points": max_pts,
                    "confidence": (confidence1 + confidence2) / 2,
                    "student_answer_read": q1.get("student_answer_read", d["llm1"]["reading"]),
                    "student_feedback": q1.get("feedback", ""),
                    "internal_reasoning": q1.get("reasoning", ""),
                    "method": "ultimatum_consensus"
                }
            else:
                # Final disagreement - average
                final_results[qid] = {
                    "grade": (grade1 + grade2) / 2,
                    "max_points": max_pts,
                    "confidence": min(confidence1, confidence2),
                    "student_answer_read": q1.get("student_answer_read", d["llm1"]["reading"]),
                    "student_feedback": q1.get("feedback", ""),
                    "internal_reasoning": q1.get("reasoning", ""),
                    "method": "ultimatum_averaged"
                }

        # Handle final name
        if name_disagreement:
            llm1_result = results_per_provider.get(provider_names[0], {})
            llm2_result = results_per_provider.get(provider_names[1], {})

            name1 = llm1_result.get("student_name") if isinstance(llm1_result, dict) else None
            name2 = llm2_result.get("student_name") if isinstance(llm2_result, dict) else None

            if name1 and name2 and name1.strip().lower() == name2.strip().lower():
                final_name = name1.strip()
            else:
                # Keep original disagreement name
                final_name = name_disagreement.llm1_name

        audit_info = {
            "type": "unified_ultimatum",
            "prompts_sent": prompts_sent,
            "questions_verified": list(final_results.keys()),
            "phases": 2
        }

        return final_results, final_name, audit_info

    # ==================== FALLBACK ULTIMATUM (for grade_with_vision) ====================

    async def _run_ultimatum_round(
        self,
        round1_results: List[Dict],
        original_results: List[Dict],
        **kwargs
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Ultimatum round for fallback per-question grading.

        This is used by grade_with_vision when single-pass fails and we need
        per-question verification. For the main flow, use unified verification.

        Args:
            round1_results: Results from verification round
            original_results: Original grading results (before any verification)
            **kwargs: Original grading arguments

        Returns:
            Tuple of (final_results, prompts_sent)
        """
        verified = []
        prompts_sent = {"llm1": None, "llm2": None}

        for i, (name, provider) in enumerate(self.providers):
            other_result = round1_results[1 - i]
            other_reasoning = other_result.get("internal_reasoning", "")
            other_grade = other_result.get("grade", 0)
            my_grade = round1_results[i].get("grade", 0)

            language = kwargs.get("language", "fr")
            max_points = kwargs.get("max_points", 5)

            if language == "fr":
                ultimatum_prompt = f"""
─── ULTIMATUM - DÉCISION FINALE ───
Question: {kwargs.get('question_text', '')}
Note maximale: {max_points} points

DÉSACCORD PERSISTANT après vérification:
- Ta note: {my_grade}/{max_points}
- Autre note: {other_grade}/{max_points}

Son raisonnement: {other_reasoning[:300]}

─── RÈGLES ───
- Option A: Accepter l'autre note → explique pourquoi
- Option B: Maintenir ta note → arguments précis
- SI INCERTAIN: abaisse ta CONFIANCE (< 0.5)

INTERDICTION: Ne choisis pas au hasard.

FORMAT DE RÉPONSE:
GRADE: [note]/{max_points}
CONFIDENCE: [0.0-1.0]
STUDENT_ANSWER_READ: [lecture]
INTERNAL_REASONING: [analyse]
STUDENT_FEEDBACK: [feedback]
"""
            else:
                ultimatum_prompt = f"""
─── ULTIMATUM - FINAL DECISION ───
Question: {kwargs.get('question_text', '')}
Max points: {max_points}

PERSISTENT DISAGREEMENT after verification:
- Your grade: {my_grade}/{max_points}
- Other grade: {other_grade}/{max_points}

Their reasoning: {other_reasoning[:300]}

─── RULES ───
- Option A: Accept the other grade → explain why
- Option B: Maintain your grade → precise arguments
- IF UNCERTAIN: lower your CONFIDENCE (< 0.5)

FORBIDDEN: Don't choose randomly.

RESPONSE FORMAT:
GRADE: [grade]/{max_points}
CONFIDENCE: [0.0-1.0]
STUDENT_ANSWER_READ: [reading]
INTERNAL_REASONING: [analysis]
STUDENT_FEEDBACK: [feedback]
"""

            prompts_sent[f"llm{i+1}"] = ultimatum_prompt.strip()

            try:
                new_result = provider.grade_with_vision(
                    question_text=kwargs.get("question_text", ""),
                    criteria=ultimatum_prompt,
                    image_path=kwargs.get("image_path"),
                    max_points=max_points,
                    class_context=kwargs.get("class_context", ""),
                    language=language
                )
                if asyncio.iscoroutine(new_result):
                    new_result = await new_result

                if new_result.get("grade") is None:
                    new_result["grade"] = my_grade
                    new_result["_parse_failed"] = True

                verified.append(new_result)
            except Exception as e:
                import logging
                logging.error(f"Ultimatum round failed for {name}: {e}")
                verified.append(round1_results[i])

        return verified, prompts_sent

    def _merge_results(
        self,
        results: List[Dict],
        comparison_info: Dict = None
    ) -> Dict[str, Any]:
        """
        Merge results from multiple providers.

        Strategy:
        - If agreement: use first result
        - If disagreement after verification: average the grades

        Args:
            results: List of results from providers
            comparison_info: Comparison metadata (NEW STRUCTURE with initial/after_cross_verification/final)

        Returns:
            Merged result
        """
        if not results:
            return {
                "grade": 0,
                "confidence": 0,
                "student_feedback": ""
            }

        # Use first result as base, but exclude internal_reasoning (kept per-LLM in comparison)
        merged = {k: v for k, v in results[0].items() if k != "internal_reasoning"}

        # Add comparison info
        if comparison_info:
            merged["comparison"] = comparison_info

        # Check if final decision indicates disagreement (new structure)
        # Handle case where comparison_info["final"] might be None
        final_info = (comparison_info.get("final") or {}) if comparison_info else {}
        is_agreement = final_info.get("agreement", True)

        # If disagreement, average the grades
        grades = [r.get("grade", 0) for r in results if r.get("grade") is not None]
        if len(grades) > 1 and not is_agreement:
            merged["grade"] = sum(grades) / len(grades)
            merged["confidence"] = min(r.get("confidence", 0) for r in results)
            merged["is_averaged"] = True

        return merged

    def _merge_results_with_user_choice(
        self,
        results: List[Dict],
        comparison_info: Dict,
        chosen_grade: float,
        feedback_source: str = "llm1"
    ) -> Dict[str, Any]:
        """
        Merge results with user's chosen grade and feedback source.

        Args:
            results: List of results from providers
            comparison_info: Comparison metadata
            chosen_grade: Grade chosen by user
            feedback_source: Which feedback to use ("llm1", "llm2", or "merge")

        Returns:
            Merged result with user's choice
        """
        if not results:
            return {
                "grade": chosen_grade,
                "confidence": 0.5,
                "student_feedback": ""
            }

        # Choose base result based on feedback source, excluding internal_reasoning
        if feedback_source == "llm2" and len(results) > 1:
            merged = {k: v for k, v in results[1].items() if k != "internal_reasoning"}
        elif feedback_source == "merge" and len(results) > 1:
            # Merge: use LLM1 as base but combine feedbacks
            merged = {k: v for k, v in results[0].items() if k != "internal_reasoning"}
            fb1 = results[0].get("student_feedback", "")
            fb2 = results[1].get("student_feedback", "")
            if fb1 and fb2:
                merged["student_feedback"] = f"{fb1} / {fb2}"
            elif fb2:
                merged["student_feedback"] = fb2
        else:
            # Default: use LLM1
            merged = {k: v for k, v in results[0].items() if k != "internal_reasoning"}

        # Override with user's chosen grade
        merged["grade"] = chosen_grade
        merged["confidence"] = 1.0  # High confidence since user decided
        merged["user_decided"] = True

        # Add comparison info
        if comparison_info:
            merged["comparison"] = comparison_info

        return merged

    # Delegate other methods to primary provider
    def call_text(self, prompt: str, system_prompt: str = None, response_format: str = None) -> str:
        """Delegate to primary provider."""
        return self.primary_provider.call_text(prompt, system_prompt, response_format)

    def call_vision(self, prompt: str, image_path: str = None, image_bytes: bytes = None,
                    pil_image=None, response_format: str = None) -> str:
        """Delegate to primary provider."""
        return self.primary_provider.call_vision(prompt, image_path, image_bytes, pil_image, response_format)

    def get_embedding(self, text: str) -> List[float]:
        """Delegate to primary provider."""
        return self.primary_provider.get_embedding(text)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Delegate to primary provider."""
        return self.primary_provider.get_embeddings(texts)

    def _parse_grading_response(self, response: str) -> Dict[str, Any]:
        """Delegate to primary provider."""
        return self.primary_provider._parse_grading_response(response)

    async def detect_student_name_with_consensus(
        self,
        image_path,
        language: str = "fr",
        name_disagreement_callback: callable = None
    ) -> Dict[str, Any]:
        """
        Detect student name using both LLMs with consensus.

        Args:
            image_path: Path to first page image
            language: Language for prompts
            name_disagreement_callback: Optional async callback for name disagreements
                                        Signature: async def callback(llm1_result, llm2_result) -> str

        Returns:
            Dict with:
            - name: final detected name
            - confidence: final confidence
            - llm1_result: result from LLM1
            - llm2_result: result from LLM2
            - consensus: True if agreement, False if resolved by callback/averaging
        """
        # Notify start
        await self._notify_progress('name_detection_start', {
            'providers': [name for name, _ in self.providers]
        })

        # Step 1: Detect with both providers in parallel
        async def detect_with_provider(index: int, name: str, provider):
            try:
                result = provider.detect_student_name(image_path, language)
                await self._notify_progress('name_detection_complete', {
                    'provider': name,
                    'provider_index': index,
                    'name': result.get('name'),
                    'confidence': result.get('confidence')
                })
                return (index, result)
            except Exception as e:
                error_result = {
                    "name": None,
                    "confidence": 0.0,
                    "reasoning": f"Error: {str(e)}"
                }
                return (index, error_result)

        tasks = [detect_with_provider(i, name, provider) for i, (name, provider) in enumerate(self.providers)]
        indexed_results = await asyncio.gather(*tasks)
        indexed_results.sort(key=lambda x: x[0])
        results = [r for _, r in indexed_results]

        name1 = results[0].get("name")
        name2 = results[1].get("name")
        conf1 = results[0].get("confidence", 0)
        conf2 = results[1].get("confidence", 0)

        comparison_info = {
            "llm1": {
                "provider": self.providers[0][0],
                "name": name1,
                "confidence": conf1,
                "reasoning": results[0].get("reasoning", "")
            },
            "llm2": {
                "provider": self.providers[1][0],
                "name": name2,
                "confidence": conf2,
                "reasoning": results[1].get("reasoning", "")
            }
        }

        # Step 2: Check for agreement
        def normalize_name(name: str) -> str:
            """Normalize name for comparison."""
            if not name:
                return ""
            # Lowercase, remove extra spaces, common variations
            return " ".join(name.lower().strip().split())

        norm1 = normalize_name(name1)
        norm2 = normalize_name(name2)

        if norm1 and norm1 == norm2:
            # Perfect agreement
            comparison_info["consensus"] = True
            await self._notify_progress('name_detection_done', {
                'name': name1,
                'consensus': True
            })
            return {
                "name": name1,  # Use original case from LLM1
                "confidence": max(conf1, conf2),
                "comparison": comparison_info,
                "consensus": True
            }

        # Step 3: Try cross-verification
        verified_results = await self._cross_verify_name(
            results, image_path, language
        )

        vname1 = verified_results[0].get("name")
        vname2 = verified_results[1].get("name")
        vnorm1 = normalize_name(vname1)
        vnorm2 = normalize_name(vname2)

        comparison_info["after_cross_verification"] = {
            "llm1": vname1,
            "llm2": vname2
        }

        if vnorm1 and vnorm1 == vnorm2:
            # Agreement after verification
            comparison_info["consensus"] = True
            await self._notify_progress('name_detection_done', {
                'name': vname1,
                'consensus': True
            })
            return {
                "name": vname1,
                "confidence": max(verified_results[0].get("confidence", 0), verified_results[1].get("confidence", 0)),
                "comparison": comparison_info,
                "consensus": True
            }

        # Step 4: Still in disagreement - use callback or choose higher confidence
        comparison_info["consensus"] = False

        if name_disagreement_callback:
            final_name = await name_disagreement_callback(
                comparison_info["llm1"],
                comparison_info["llm2"]
            )
        else:
            # Default: use the name with higher confidence, or first if equal
            if conf1 >= conf2:
                final_name = name1
            else:
                final_name = name2

        await self._notify_progress('name_detection_done', {
            'name': final_name,
            'consensus': False
        })

        return {
            "name": final_name,
            "confidence": 0.5,  # Lower confidence for non-consensus
            "comparison": comparison_info,
            "consensus": False
        }

    async def _cross_verify_name(
        self,
        results: List[Dict],
        image_path,
        language: str
    ) -> List[Dict]:
        """
        Cross-verify names by showing each LLM the other's detection.

        Args:
            results: Initial detection results
            image_path: Image path
            language: Language

        Returns:
            List of verified results
        """
        verified = []

        for i, (name, provider) in enumerate(self.providers):
            other_result = results[1 - i]
            other_name = other_result.get("name") or "Inconnu"
            other_reasoning = other_result.get("reasoning", "")

            # Build verification prompt
            if language == "fr":
                verify_context = f"""
AUTRE DÉTECTION:
Nom détecté par l'autre IA: {other_name}
Raisonnement: {other_reasoning}

Prends en compte cette détection. Si tu es d'accord, confirme. Sinon, explique pourquoi.
"""
            else:
                verify_context = f"""
OTHER DETECTION:
Name detected by other AI: {other_name}
Reasoning: {other_reasoning}

Consider this detection. If you agree, confirm. Otherwise, explain why.
"""

            # Re-detect with context
            try:
                base_result = provider.detect_student_name(image_path, language)
                # The verify_context would need to be added to the prompt
                # For simplicity, we just use the base result
                verified.append(base_result)
            except Exception:
                verified.append(results[i])

        return verified

    def detect_student_name(self, image_path, language: str = "fr") -> Dict[str, Any]:
        """Delegate to primary provider (non-async version for compatibility)."""
        return self.primary_provider.detect_student_name(image_path, language)

    # ==================== READING CONSENSUS ====================

    async def read_student_answer_with_consensus(
        self,
        image_path,
        question_text: str,
        language: str = "fr",
        reading_disagreement_callback: callable = None
    ) -> Dict[str, Any]:
        """
        Read what the student wrote using both LLMs with consensus.

        This is Phase 1 of grading - establish WHAT the student wrote
        before grading (Phase 2).

        Args:
            image_path: Path(s) to image(s)
            question_text: The question to look for
            language: Language for prompts
            reading_disagreement_callback: Optional async callback for reading disagreements
                                           Signature: async def callback(llm1_reading, llm2_reading, question_text, image_path) -> str
                                           Returns the chosen reading

        Returns:
            Dict with:
            - reading: final agreed reading of student's answer
            - consensus: True if agreement, False if resolved by callback
            - llm1_reading: what LLM1 read
            - llm2_reading: what LLM2 read
            - user_validated: True if user had to choose
        """
        # Notify start
        await self._notify_progress('reading_start', {
            'providers': [name for name, _ in self.providers],
            'question': question_text[:50] + '...' if len(question_text) > 50 else question_text
        })

        # Step 1: Read with both providers in parallel
        async def read_with_provider(index: int, name: str, provider):
            try:
                result = await self._describe_answer(
                    provider, image_path, question_text, language
                )
                await self._notify_progress('reading_complete', {
                    'provider': name,
                    'provider_index': index,
                    'reading': result.get('reading', '')[:100] + '...' if len(result.get('reading', '')) > 100 else result.get('reading', ''),
                    'confidence': result.get('confidence')
                })
                return (index, result)
            except Exception as e:
                error_result = {
                    "reading": f"Erreur: {str(e)}",
                    "confidence": 0.0
                }
                return (index, error_result)

        tasks = [read_with_provider(i, name, provider) for i, (name, provider) in enumerate(self.providers)]
        indexed_results = await asyncio.gather(*tasks)
        indexed_results.sort(key=lambda x: x[0])
        results = [r for _, r in indexed_results]

        reading1 = results[0].get("reading", "")
        reading2 = results[1].get("reading", "")
        conf1 = results[0].get("confidence", 0)
        conf2 = results[1].get("confidence", 0)

        comparison_info = {
            "llm1": {
                "provider": self.providers[0][0],
                "reading": reading1,
                "confidence": conf1
            },
            "llm2": {
                "provider": self.providers[1][0],
                "reading": reading2,
                "confidence": conf2
            }
        }

        # Step 2: Check for compatibility
        if self._readings_compatible(reading1, reading2):
            # Compatible readings - merge them
            merged_reading = self._merge_readings(reading1, reading2, conf1, conf2)
            comparison_info["consensus"] = True

            await self._notify_progress('reading_done', {
                'reading': merged_reading[:100] + '...' if len(merged_reading) > 100 else merged_reading,
                'consensus': True
            })

            return {
                "reading": merged_reading,
                "consensus": True,
                "user_validated": False,
                "comparison": comparison_info
            }

        # Step 3: Try cross-verification for readings (reading-only, no grade re-evaluation)
        verified_readings, _ = await self._cross_verify_reading(
            results, image_path, question_text, language, include_grade=False
        )

        vreading1 = verified_readings[0].get("reading", "")
        vreading2 = verified_readings[1].get("reading", "")

        comparison_info["after_cross_verification"] = {
            "llm1": vreading1,
            "llm2": vreading2
        }

        if self._readings_compatible(vreading1, vreading2):
            merged_reading = self._merge_readings(vreading1, vreading2,
                                                   verified_readings[0].get("confidence", 0),
                                                   verified_readings[1].get("confidence", 0))
            comparison_info["consensus"] = True

            await self._notify_progress('reading_done', {
                'reading': merged_reading[:100] + '...' if len(merged_reading) > 100 else merged_reading,
                'consensus': True
            })

            return {
                "reading": merged_reading,
                "consensus": True,
                "user_validated": False,
                "comparison": comparison_info
            }

        # Step 4: Still in disagreement - need user intervention
        comparison_info["consensus"] = False

        if reading_disagreement_callback:
            # Let user choose
            final_reading = await reading_disagreement_callback(
                comparison_info["llm1"],
                comparison_info["llm2"],
                question_text,
                image_path
            )
            user_validated = True
        else:
            # Default: use longer reading (usually more detailed) or higher confidence
            if conf1 > conf2:
                final_reading = reading1
            elif conf2 > conf1:
                final_reading = reading2
            elif len(reading1) >= len(reading2):
                final_reading = reading1
            else:
                final_reading = reading2
            user_validated = False

        await self._notify_progress('reading_done', {
            'reading': final_reading[:100] + '...' if len(final_reading) > 100 else final_reading,
            'consensus': False,
            'user_validated': user_validated
        })

        return {
            "reading": final_reading,
            "consensus": False,
            "user_validated": user_validated,
            "comparison": comparison_info
        }

    async def _describe_answer(
        self,
        provider,
        image_path,
        question_text: str,
        language: str
    ) -> Dict[str, Any]:
        """
        Ask a provider to describe what the student wrote.

        This is a pure reading task - no grading involved.

        Args:
            provider: The LLM provider
            image_path: Image path(s)
            question_text: The question to look for
            language: Language for prompt

        Returns:
            Dict with reading and confidence
        """
        num_pages = len(image_path) if isinstance(image_path, list) else 1

        if language == "fr":
            prompt = f"""Tu es un lecteur neutre. Ton UNIQUE tâche est de lire ce que l'élève a répondu.

QUESTION RECHERCHÉE: {question_text}

INSTRUCTIONS:
1. Localise cette question sur la copie (peut être sur n'importe quelle page)
2. Lis EXACTEMENT ce que l'élève a écrit/dessiné
3. Transcris le texte brut, sans phrase introductive

FORMAT DE RÉPONSE:
TROUVÉ: [oui/non/partiellement]
TEXTE_LU: [texte exact écrit par l'élève, sans phrase introductive]
CONFIDENCE: [0.0 à 1.0 - ta certitude sur ta lecture]
"""
        else:
            prompt = f"""You are a neutral reader. Your ONLY task is to read what the student answered.

QUESTION TO FIND: {question_text}

INSTRUCTIONS:
1. Locate this question on the copy (may be on any page)
2. Read EXACTLY what the student wrote/drew
3. Transcribe the raw text, without introductory phrase

RESPONSE FORMAT:
FOUND: [yes/no/partially]
TEXT_READ: [exact text written by the student, no introductory phrase]
CONFIDENCE: [0.0 to 1.0 - your certainty about your reading]
"""

        # Add multi-page context if needed
        if num_pages > 1:
            if language == "fr":
                prompt += f"\n\nNOTE: Tu as accès à {num_pages} pages. Cherche la question sur TOUTES les pages."
            else:
                prompt += f"\n\nNOTE: You have access to {num_pages} pages. Search for the question on ALL pages."

        try:
            response = provider.call_vision(prompt, image_path=image_path)
            return self._parse_reading_response(response)
        except Exception as e:
            return {
                "reading": f"Erreur de lecture: {str(e)}",
                "confidence": 0.0
            }

    def _parse_reading_response(self, response: str) -> Dict[str, Any]:
        """Parse reading response from LLM."""
        result = {
            "reading": "",
            "confidence": 0.5,
            "found": False
        }

        if not response:
            return result

        lines = response.strip().split('\n')
        found = False
        content_parts = []

        for line in lines:
            line = line.strip()

            if line.upper().startswith("TROUVÉ:") or line.upper().startswith("FOUND:"):
                value = line.split(':', 1)[1].strip().lower()
                found = value in ['oui', 'yes', 'partially', 'partiellement']
                result["found"] = found

            elif line.upper().startswith("CONTENU:") or line.upper().startswith("CONTENT:") or line.upper().startswith("TEXTE_LU:") or line.upper().startswith("TEXT_READ:"):
                content = line.split(':', 1)[1].strip()
                content_parts.append(content)

            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line.split(':', 1)[1].strip())
                except ValueError:
                    pass

            elif content_parts and line:
                # Continuation of content
                content_parts.append(line)

        result["reading"] = " ".join(content_parts) if content_parts else response
        return result

    def _readings_compatible(self, reading1: str, reading2: str) -> bool:
        """
        Check if two readings are compatible (describe the same thing).

        Args:
            reading1: First LLM's reading
            reading2: Second LLM's reading

        Returns:
            True if readings are compatible
        """
        if not reading1 or not reading2:
            return False

        # Normalize for comparison
        r1 = reading1.lower().strip()
        r2 = reading2.lower().strip()

        # If both say not found, compatible
        not_found_indicators = ['non', 'no', 'pas visible', 'not visible', 'absent']
        r1_not_found = any(ind in r1 for ind in not_found_indicators)
        r2_not_found = any(ind in r2 for ind in not_found_indicators)

        if r1_not_found and r2_not_found:
            return True

        # If one found and one not, incompatible
        if r1_not_found != r2_not_found:
            return False

        # Check for key contradictions (common problematic cases)
        contradictions = [
            (['erlenmeyer'], ['fiole jaugée', 'volumetric flask', 'fiole']),
            (['becher', 'beaker'], ['erlenmeyer', 'fiole']),
            (['bleu'], ['rouge', 'vert', 'jaune']),
            (['carré'], ['rond', 'cercle', 'triangle']),
            (['oui', 'yes', 'vrai', 'true'], ['non', 'no', 'faux', 'false'])
        ]

        for group1, group2 in contradictions:
            r1_has_g1 = any(g in r1 for g in group1)
            r1_has_g2 = any(g in r1 for g in group2)
            r2_has_g1 = any(g in r2 for g in group1)
            r2_has_g2 = any(g in r2 for g in group2)

            # If one has group1 and other has group2, they contradict
            if (r1_has_g1 and r2_has_g2) or (r1_has_g2 and r2_has_g1):
                return False

        # Check for significant word overlap
        words1 = set(r1.split())
        words2 = set(r2.split())

        # Remove common words
        stop_words = {'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'a', 'est', 'sont',
                      'the', 'a', 'an', 'is', 'are', 'was', 'were', 'et', 'and', 'ou', 'or',
                      'avec', 'with', 'sur', 'on', 'dans', 'in', 'pour', 'for', 'que', 'that',
                      "l'élève", "l'eleve", 'student', 'a écrit', 'wrote', 'dessiné', 'drew'}
        words1 -= stop_words
        words2 -= stop_words

        if not words1 or not words2:
            return False

        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        similarity = intersection / union if union > 0 else 0

        # Consider compatible if > 30% word overlap
        return similarity > 0.3

    def _merge_readings(self, reading1: str, reading2: str, conf1: float, conf2: float) -> str:
        """
        Merge two compatible readings into one.

        Args:
            reading1: First reading
            reading2: Second reading
            conf1: Confidence of first reading
            conf2: Confidence of second reading

        Returns:
            Merged reading
        """
        if not reading1:
            return reading2
        if not reading2:
            return reading1

        # If one is much longer and more detailed, use it
        if len(reading1) > len(reading2) * 1.5:
            return reading1
        if len(reading2) > len(reading1) * 1.5:
            return reading2

        # If similar length, prefer higher confidence
        if conf1 > conf2:
            return reading1
        elif conf2 > conf1:
            return reading2

        # Default: use first (usually more consistent)
        return reading1

    async def _cross_verify_reading(
        self,
        results: List[Dict],
        image_path,
        question_text: str,
        language: str,
        max_points: float = 5.0,
        grading_context: str = "",
        include_grade: bool = True
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Cross-verify readings by showing each LLM the other's reading.
        Optionally re-evaluates the grade if the reading changes.

        Args:
            results: Initial reading results (from grading phase)
            image_path: Image path(s)
            question_text: The question
            language: Language
            max_points: Maximum points for re-grading (only used if include_grade=True)
            grading_context: Original grading criteria (only used if include_grade=True)
            include_grade: If True, also ask for grade re-evaluation

        Returns:
            Tuple of (verified_results, prompts_sent)
        """
        verified = []
        prompts_sent = {"llm1": None, "llm2": None}

        for i, (name, provider) in enumerate(self.providers):
            other_result = results[1 - i]
            other_reading = other_result.get("reading", "")
            my_reading = results[i].get("reading", "")
            my_grade = results[i].get("grade", 0)
            other_grade = other_result.get("grade", 0)

            if include_grade:
                # Full verification with grade re-evaluation
                if language == "fr":
                    verify_prompt = f"""─── VÉRIFICATION DE LECTURE ───
Tu as lu: "{my_reading}"
Tu as noté: {my_grade}/{max_points}

Un autre correcteur a lu: "{other_reading}"
Il a noté: {other_grade}/{max_points}

Question: {question_text}

─── TA TÂCHE ───
1. Compare les deux lectures
2. Si l'autre lecture est plus précise, adopte-la
3. RÉÉVALUE ta note si ta lecture change significativement
4. Si tu maintiens ta lecture, confirme ta note

─── RÈGLES ───
- Une lecture plus précise peut justifier une note différente
- Ne change ta note QUE si tu identifies une différence factuelle
- Explique ta décision

FORMAT DE RÉPONSE:
LECTURE FINALE: [ta lecture finale]
CONFIDENCE: [0.0 à 1.0]
NOTE RÉÉVALUÉE: [ta note sur {max_points}]
JUSTIFICATION: [pourquoi tu maintiens ou changes]"""
                else:
                    verify_prompt = f"""─── READING VERIFICATION ───
You read: "{my_reading}"
You graded: {my_grade}/{max_points}

Another grader read: "{other_reading}"
They graded: {other_grade}/{max_points}

Question: {question_text}

─── YOUR TASK ───
1. Compare the two readings
2. If the other reading is more accurate, adopt it
3. RE-EVALUATE your grade if your reading changes significantly
4. If you maintain your reading, confirm your grade

─── RULES ───
- A more accurate reading may justify a different grade
- Only change your grade if you identify a factual difference
- Explain your decision

RESPONSE FORMAT:
FINAL READING: [your final reading]
CONFIDENCE: [0.0 to 1.0]
RE-EVALUATED GRADE: [your grade out of {max_points}]
JUSTIFICATION: [why you maintain or change]"""
            else:
                # Reading-only verification (no grade re-evaluation)
                if language == "fr":
                    verify_prompt = f"""─── VÉRIFICATION DE LECTURE ───
Tu as lu: "{my_reading}"

Un autre correcteur a lu: "{other_reading}"

Question: {question_text}

─── TA TÂCHE ───
1. Compare les deux lectures
2. Si l'autre lecture est plus précise, adopte-la
3. Sinon, maintiens ta lecture

FORMAT DE RÉPONSE:
LECTURE FINALE: [ta lecture finale]
CONFIDENCE: [0.0 à 1.0]"""
                else:
                    verify_prompt = f"""─── READING VERIFICATION ───
You read: "{my_reading}"

Another grader read: "{other_reading}"

Question: {question_text}

─── YOUR TASK ───
1. Compare the two readings
2. If the other reading is more accurate, adopt it
3. Otherwise, maintain your reading

RESPONSE FORMAT:
FINAL READING: [your final reading]
CONFIDENCE: [0.0 to 1.0]"""

            # Store the prompt
            prompts_sent[f"llm{i+1}"] = verify_prompt.strip()

            try:
                response = provider.call_vision(verify_prompt, image_path=image_path)
                parsed = self._parse_reading_with_grade(response, max_points, include_grade=include_grade)

                # Store the raw response
                parsed["raw_response"] = response

                # If parsing didn't extract reading, use the whole response
                if not parsed.get("reading"):
                    parsed["reading"] = response
                    parsed["confidence"] = 0.5

                # Preserve original grade if not re-evaluating
                if not include_grade or parsed.get("grade") is None:
                    parsed["grade"] = my_grade

                verified.append(parsed)
            except Exception as e:
                verified.append({
                    **results[i],
                    "error": str(e)
                })

        return verified, prompts_sent

    def _parse_reading_with_grade(self, response: str, max_points: float = 5.0, include_grade: bool = True) -> Dict[str, Any]:
        """Parse reading response, optionally including a re-evaluated grade."""
        result = {
            "reading": "",
            "confidence": 0.5,
            "grade": None,
            "justification": ""
        }

        if not response:
            return result

        lines = response.strip().split('\n')
        content_parts = []

        for line in lines:
            line = line.strip()

            if line.upper().startswith("LECTURE FINALE:") or line.upper().startswith("FINAL READING:"):
                content = line.split(':', 1)[1].strip()
                content_parts.append(content)

            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line.split(':', 1)[1].strip())
                except ValueError:
                    pass

            elif line.upper().startswith("NOTE RÉÉVALUÉE:") or line.upper().startswith("RE-EVALUATED GRADE:"):
                try:
                    grade_str = line.split(':', 1)[1].strip()
                    # Extract number from string like "2.0/5" or just "2.0"
                    grade_str = grade_str.split('/')[0].strip()
                    result["grade"] = float(grade_str)
                except ValueError:
                    pass

            elif line.upper().startswith("JUSTIFICATION:"):
                result["justification"] = line.split(':', 1)[1].strip()

            elif content_parts and line:
                # Continuation of reading
                content_parts.append(line)

        result["reading"] = " ".join(content_parts) if content_parts else response
        return result

    # ==================== DUAL LLM GRADING ====================

    async def grade_copy(
        self,
        questions: List[Dict[str, Any]],
        image_paths: List[str],
        language: str = "fr",
        disagreement_callback: callable = None,
        reading_disagreement_callback: callable = None,
        second_reading: bool = False
    ) -> Dict[str, Any]:
        """
        Grade a copy using dual-LLM verification.

        Flow:
        1. Single-pass: Both LLMs grade all questions in parallel
        2. Disagreement analysis: Identify questions needing verification
        3. Phase 2a: Isolated verification (each LLM re-reads independently)
        4. Phase 2b: Exchange verification (LLMs share readings and discuss)
        5. Phase 3: Ultimatum (final decision if still disagreeing)

        Args:
            questions: List of question dicts with:
                - id: Question identifier
                - text: Question text
                - criteria: Grading criteria
                - max_points: Maximum points
            image_paths: List of image paths (all pages)
            language: Language for prompts
            disagreement_callback: Optional callback for grade disagreements
            reading_disagreement_callback: Optional callback for reading disagreements
            second_reading: If True, include second reading instruction in prompts

        Returns:
            Dict with:
            - questions: Dict of question_id -> final result
            - audit: Complete audit trail
            - summary: Statistics
        """
        from ai.single_pass_grader import SinglePassGrader
        from ai.disagreement_analyzer import DisagreementAnalyzer

        total_start = time.time()

        # Notify start
        await self._notify_progress('grading_phase_start', {
            'phase': 'single_pass',
            'num_questions': len(questions),
            'num_pages': len(image_paths)
        })

        # ===== PHASE 1: Single-Pass Grading (FRESH call with images) =====
        single_pass_results = {}
        single_pass_errors = {}

        async def run_single_pass(index: int, name: str, provider):
            """Run single-pass grading for one provider (FRESH call with images)."""
            grader = SinglePassGrader(provider)
            try:
                result = await grader.grade_all_questions(
                    questions, image_paths, language, second_reading=second_reading
                )
                return (index, name, result)
            except Exception as e:
                return (index, name, None)

        # Run both providers in parallel
        tasks = [run_single_pass(i, name, provider) for i, (name, provider) in enumerate(self.providers)]
        results = await asyncio.gather(*tasks)

        for idx, name, result in results:
            if result and result.parse_success:
                single_pass_results[name] = result
            else:
                single_pass_errors[name] = {
                    "provider": name,
                    "error": result.parse_errors if result else "Unknown error"
                }

        # Check if we have results from both providers
        if len(single_pass_results) < 2:
            # Fallback to per-question grading
            await self._notify_progress('grading_fallback', {
                'reason': 'single_pass_failed',
                'errors': single_pass_errors
            })
            return await self._fallback_per_question(
                questions, image_paths, language,
                disagreement_callback, reading_disagreement_callback
            )

        # ===== PHASE 2: Analyze Disagreements =====
        await self._notify_progress('grading_phase_start', {
            'phase': 'disagreement_analysis'
        })

        analyzer = DisagreementAnalyzer()
        provider_names = [name for name, _ in self.providers]

        report = analyzer.analyze(
            single_pass_results[provider_names[0]].to_dict(),
            single_pass_results[provider_names[1]].to_dict()
        )

        # Note: analysis_complete is emitted by session.py with full data

        # ===== Determine initial student name (may be updated during verification) =====
        # Start with single-pass consensus or first available
        final_student_name = report.llm1_name or report.llm2_name
        name_needs_verification = report.name_disagreement is not None

        # ===== PHASE 3: Unified Verification (NEW) =====
        final_results = {}
        verification_audit = {}

        # Questions with agreement: use single-pass results directly
        for qid in report.agreed_questions:
            llm1_q = single_pass_results[provider_names[0]].questions.get(qid)
            llm2_q = single_pass_results[provider_names[1]].questions.get(qid)

            # Use detected max_points (prefer LLM1's detection, or average if different)
            detected_max_points = llm1_q.max_points if llm1_q else 1.0

            # Use LLM1's result as base (they agreed anyway)
            final_results[qid] = {
                "grade": llm1_q.grade,
                "max_points": detected_max_points,  # Include detected scale
                "confidence": (llm1_q.confidence + llm2_q.confidence) / 2,
                "student_answer_read": llm1_q.student_answer_read,
                "student_feedback": llm1_q.feedback,
                "internal_reasoning": llm1_q.reasoning,
                "method": "single_pass_consensus"
            }
            # Simplified audit - no need to repeat llm1/llm2 data (already in single_pass)
            verification_audit[qid] = {
                "method": "single_pass",
                "agreement": True
            }

        # ===== UNIFIED VERIFICATION for ALL disagreements =====
        if report.has_any_disagreement:
            await self._notify_progress('grading_phase_start', {
                'phase': 'unified_verification',
                'num_disagreements': len(report.flagged_questions),
                'name_disagreement': report.name_disagreement is not None
            })

            # Run unified verification for ALL disagreements in ONE call per LLM
            verified_results, consensus_name, verification_info = await self._run_unified_verification(
                questions=questions,
                flagged_questions=report.flagged_questions,
                name_disagreement=report.name_disagreement,
                single_pass_results=single_pass_results,
                image_paths=image_paths,
                language=language
            )

            # Update student name if consensus reached
            if consensus_name:
                final_student_name = consensus_name

            # Merge verified results into final_results
            final_results.update(verified_results)

            # Build audit for verified questions
            for d in report.flagged_questions:
                qid = d.question_id
                sp_llm1_q = single_pass_results[provider_names[0]].questions.get(qid)
                sp_llm2_q = single_pass_results[provider_names[1]].questions.get(qid)

                verification_audit[qid] = {
                    "method": verified_results[qid].get("method", "unified"),
                    "agreement": not verified_results[qid].get("_needs_ultimatum", False),
                    "reason": d.reason,
                    "evolution": {
                        "initial": [
                            sp_llm1_q.grade if sp_llm1_q else 0,
                            sp_llm2_q.grade if sp_llm2_q else 0
                        ],
                        "final": verified_results[qid].get("grade")
                    }
                }

            # Check if there are remaining disagreements needing ultimatum
            remaining = getattr(self, '_unified_remaining', [])
            if remaining:
                await self._notify_progress('grading_phase_start', {
                    'phase': 'unified_ultimatum',
                    'num_remaining': len(remaining)
                })

                # Run unified ultimatum
                ultimatum_results, final_name, ultimatum_info = await self._run_unified_ultimatum(
                    questions=questions,
                    remaining_disagreements=remaining,
                    evolution=getattr(self, '_unified_evolution', {}),
                    name_disagreement=report.name_disagreement if name_needs_verification else None,
                    verification_results=verified_results,
                    image_paths=image_paths,
                    language=language
                )

                # Update with ultimatum results
                for qid, result in ultimatum_results.items():
                    final_results[qid] = result
                    verification_audit[qid] = {
                        "method": result.get("method", "ultimatum"),
                        "agreement": "consensus" in result.get("method", ""),
                        "evolution": {
                            **verification_audit[qid].get("evolution", {}),
                            "after_ultimatum": result.get("grade")
                        }
                    }

                if final_name:
                    final_student_name = final_name

            # Store verification info in audit
            verification_audit["_unified"] = verification_info

        # ===== PHASE 4: Assemble Final Results =====
        total_duration = (time.time() - total_start) * 1000

        await self._notify_progress('grading_complete', {
            'total_questions': len(questions),
            'single_pass_agreed': len(report.agreed_questions),
            'verified': len(report.flagged_questions),
            'total_duration_ms': round(total_duration, 1)
        })

        # Helper for natural sorting (Q1, Q2, Q10 instead of Q1, Q10, Q2)
        def natural_sort_key(s):
            import re
            match = re.match(r'Q(\d+)', str(s))
            if match:
                return (0, int(match.group(1)))
            return (1, str(s))

        # Build audit organized BY QUESTION (not by phase), in natural order
        questions_audit = {}

        for qid in sorted(final_results.keys(), key=natural_sort_key):
            sp_llm1_q = single_pass_results[provider_names[0]].questions.get(qid)
            sp_llm2_q = single_pass_results[provider_names[1]].questions.get(qid)

            questions_audit[qid] = {
                # Single pass results for each LLM
                provider_names[0]: {
                    "grade": sp_llm1_q.grade if sp_llm1_q else None,
                    "max_points": sp_llm1_q.max_points if sp_llm1_q else 1.0,
                    "reading": sp_llm1_q.student_answer_read if sp_llm1_q else "",
                    "reasoning": sp_llm1_q.reasoning if sp_llm1_q else "",
                    "feedback": sp_llm1_q.feedback if sp_llm1_q else "",
                    "confidence": sp_llm1_q.confidence if sp_llm1_q else 0.5
                },
                provider_names[1]: {
                    "grade": sp_llm2_q.grade if sp_llm2_q else None,
                    "max_points": sp_llm2_q.max_points if sp_llm2_q else 1.0,
                    "reading": sp_llm2_q.student_answer_read if sp_llm2_q else "",
                    "reasoning": sp_llm2_q.reasoning if sp_llm2_q else "",
                    "feedback": sp_llm2_q.feedback if sp_llm2_q else "",
                    "confidence": sp_llm2_q.confidence if sp_llm2_q else 0.5
                },
                # Verification (null if agreed in single pass)
                "verification": verification_audit.get(qid),
                # Final result
                "final": {
                    "grade": final_results[qid].get("grade"),
                    "max_points": final_results[qid].get("max_points"),
                    "confidence": final_results[qid].get("confidence"),
                    "feedback": final_results[qid].get("student_feedback")
                }
            }

        # Build final_results in natural order too
        sorted_final_results = {qid: final_results[qid] for qid in sorted(final_results.keys(), key=natural_sort_key)}

        # Build results section (simplified final results per question)
        results = {}
        for qid in sorted(final_results.keys(), key=natural_sort_key):
            results[qid] = {
                "grade": final_results[qid].get("grade"),
                "max_points": final_results[qid].get("max_points"),
                "reading": final_results[qid].get("student_answer_read", ""),
                "feedback": final_results[qid].get("student_feedback", "")
            }

        # Build verification summary
        verification_summary = {
            "type": "unified",
            "questions_verified": list(verification_audit.keys()),
            "name_verified": report.name_disagreement is not None
        }
        if "_unified" in verification_audit:
            verification_summary["phases"] = verification_audit["_unified"].get("phases", 1)
            del verification_audit["_unified"]

        return {
            "results": results,
            "student_name": final_student_name,
            "student_name_disagreement": report.name_disagreement.to_dict() if report.name_disagreement else None,
            "options": {
                "mode": "dual_llm",
                "providers": provider_names,
                "verification_type": "unified",
                "second_reading": second_reading,
                "num_pages": len(image_paths),
                "num_questions": len(questions)
            },
            "verification": verification_summary,
            "llm_comparison": questions_audit,
            "summary": {
                "total_questions": len(questions),
                "agreed_in_single_pass": len(report.agreed_questions),
                "required_verification": len(report.flagged_questions),
                "agreement_rate": report.agreement_rate,
                "total_score": sum(r["grade"] or 0 for r in final_results.values()),
                "max_score": sum(r.get("max_points", 1.0) for r in final_results.values())
            },
            "timing": {
                "total_ms": round(total_duration, 1)
            }
        }

    async def _fallback_per_question(
        self,
        questions: List[Dict[str, Any]],
        image_paths: List[str],
        language: str,
        disagreement_callback: callable,
        reading_disagreement_callback: callable
    ) -> Dict[str, Any]:
        """
        Fallback to per-question grading when single-pass fails.
        """
        total_start = time.time()
        final_results = {}
        verification_audit = {}

        for q in questions:
            qid = q["id"]

            result = await self.grade_with_vision(
                question_text=q["text"],
                criteria=q["criteria"],
                image_path=image_paths,
                max_points=q["max_points"],
                language=language,
                question_id=qid,
                reading_disagreement_callback=reading_disagreement_callback
            )

            final_results[qid] = {
                "grade": result.get("grade"),
                "confidence": result.get("confidence"),
                "student_answer_read": result.get("student_answer_read", ""),
                "student_feedback": result.get("student_feedback", ""),
                "method": "per_question_fallback"
            }

            verification_audit[qid] = {
                "method": "per_question_fallback",
                "comparison": result.get("comparison")
            }

        total_duration = (time.time() - total_start) * 1000

        return {
            "questions": final_results,
            "audit": {
                "method": "per_question_fallback",
                "options": {
                    "second_reading": False  # Fallback doesn't support second reading
                },
                "verification": verification_audit,
                "timing": {
                    "total_ms": round(total_duration, 1)
                }
            },
            "summary": {
                "total_questions": len(questions),
                "agreed_in_single_pass": 0,
                "required_verification": len(questions),
                "agreement_rate": 0.0,
                "total_score": sum(r["grade"] for r in final_results.values()),
                "max_score": sum(q["max_points"] for q in questions)
            }
        }
