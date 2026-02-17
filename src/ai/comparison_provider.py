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

    async def _run_verification_with_fresh_call(
        self,
        results: List[Dict],
        image_paths: List[str],
        **kwargs
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Verification with FRESH call + explicit context + images re-sent.

        Each LLM gets a completely fresh call with images re-sent and
        explicit context about the other LLM's reasoning. This eliminates
        anchoring bias and ensures a fresh look at the images.

        Args:
            results: Initial grading results from both LLMs
            image_paths: List of image paths to RE-SEND
            **kwargs: Original grading arguments

        Returns:
            Tuple of (verified_results, prompts_sent)
        """
        verified = []
        prompts_sent = {"llm1": None, "llm2": None}

        for i, (name, provider) in enumerate(self.providers):
            my_initial = results[i]
            other_initial = results[1 - i]
            other_grade = other_initial.get("grade", 0)
            my_grade = my_initial.get("grade", 0)

            language = kwargs.get("language", "fr")
            max_points = kwargs.get("max_points", 5)

            # Build explicit context prompt
            # NOTE: We do NOT include previous readings OR reasoning to avoid anchoring on OCR errors
            # The LLM must re-read the student's answer entirely fresh from the images
            if language == "fr":
                verify_prompt = f"""─── QUESTION À VÉRIFIER ───
{kwargs.get('question_text', '')}

Critères: {kwargs.get('criteria', 'Non spécifiés')}
Note maximale: {max_points} points

─── SITUATION ───
Tu as initialement noté: {my_grade}/{max_points}
Un autre correcteur a noté: {other_grade}/{max_points}

Il y a un désaccord. Tu dois ré-examiner cette question.

─── INSTRUCTION ───
1. RELIS TOI-MÊME la réponse de l'élève sur les images (ne présume de rien)
2. Analyse OBJECTIVEMENT ce qui est correct et ce qui ne l'est pas
3. Décide de TA note finale

RÈGLES:
- Lis attentivement l'écriture manuscrite lettre par lettre
- Ne te fie à aucune lecture précédente - fais ta propre lecture
- Note selon les critères, pas selon ton jugement initial
- Si incertain: abaisse ta confiance (< 0.5)

─── FORMAT DE RÉPONSE ───
STUDENT_ANSWER_READ: [ce que tu lis toi-même sur la copie]
GRADE: [note]/{max_points}
CONFIDENCE: [0.0-1.0]
INTERNAL_REASONING: [analyse]
STUDENT_FEEDBACK: [feedback]"""
            else:
                verify_prompt = f"""─── QUESTION TO VERIFY ───
{kwargs.get('question_text', '')}

Criteria: {kwargs.get('criteria', 'Not specified')}
Max points: {max_points}

─── SITUATION ───
You initially graded: {my_grade}/{max_points}
Another grader graded: {other_grade}/{max_points}

There is a disagreement. You must re-examine this question.

─── INSTRUCTION ───
1. RE-READ the student's answer yourself from the images (presume nothing)
2. Analyze OBJECTIVELY what is correct and what is not
3. Decide on YOUR final grade

RULES:
- Read the handwriting carefully letter by letter
- Do not rely on any previous reading - make your own reading
- Grade according to criteria, not according to your initial judgment
- If uncertain: lower your confidence (< 0.5)

─── RESPONSE FORMAT ───
STUDENT_ANSWER_READ: [what you read yourself from the copy]
GRADE: [grade]/{max_points}
CONFIDENCE: [0.0-1.0]
INTERNAL_REASONING: [analysis]
STUDENT_FEEDBACK: [feedback]"""

            prompts_sent[f"llm{i+1}"] = verify_prompt.strip()

            try:
                # FRESH call with images RE-SENT
                new_result = provider.grade_with_vision(
                    question_text=kwargs.get("question_text", ""),
                    criteria=verify_prompt,
                    image_path=image_paths,
                    max_points=max_points,
                    class_context=kwargs.get("class_context", ""),
                    language=language
                )
                if asyncio.iscoroutine(new_result):
                    new_result = await new_result

                # Warn if grade couldn't be parsed
                if new_result.get("grade") is None:
                    import logging
                    logging.warning(f"Failed to parse grade from {name} verification response. Using original grade.")
                    new_result["grade"] = my_grade
                    new_result["_parse_failed"] = True

                verified.append(new_result)
            except Exception as e:
                import logging
                logging.error(f"Verification failed for {name}: {e}")
                verified.append(results[i])

        return verified, prompts_sent

    async def _run_exchange_verification(
        self,
        isolated_results: List[Dict],
        image_paths: List[str],
        **kwargs
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Exchange verification: each LLM sees the other's isolated reading and can adjust.

        This phase comes AFTER isolated verification to allow cross-checking
        without anchoring bias. Each LLM made its own fresh reading, now they
        can compare and discuss.

        Args:
            isolated_results: Results from isolated verification phase
            image_paths: List of image paths to RE-SEND
            **kwargs: Original grading arguments

        Returns:
            Tuple of (exchange_results, prompts_sent)
        """
        verified = []
        prompts_sent = {"llm1": None, "llm2": None}

        for i, (name, provider) in enumerate(self.providers):
            my_isolated = isolated_results[i]
            other_isolated = isolated_results[1 - i]

            my_grade = my_isolated.get("grade", 0)
            my_reading = my_isolated.get("student_answer_read", "")
            my_reasoning = my_isolated.get("internal_reasoning", "")

            other_grade = other_isolated.get("grade", 0)
            other_reading = other_isolated.get("student_answer_read", "")
            other_reasoning = other_isolated.get("internal_reasoning", "")

            language = kwargs.get("language", "fr")
            max_points = kwargs.get("max_points", 5)

            # Build exchange prompt - NOW we share the readings!
            if language == "fr":
                exchange_prompt = f"""─── QUESTION EN DÉSACCORD ───
{kwargs.get('question_text', '')}

Critères: {kwargs.get('criteria', 'Non spécifiés')}
Note maximale: {max_points} points

─── ÉCHANGE DES LECTURES ───
Tu as lu: "{my_reading}"
Tu as noté: {my_grade}/{max_points}

L'autre correcteur a lu: "{other_reading}"
Il a noté: {other_grade}/{max_points}

─── ANALYSE COMPARATIVE ───
1. Compare les deux lectures - y a-t-il une erreur de lecture?
2. Si l'autre lecture semble plus précise, adopte-la
3. Si ta lecture semble plus précise, maintiens-la avec justification
4. Décide de ta note finale

RÈGLES:
- Les erreurs de lecture manuscrite sont courantes - reste ouvert
- Ne change que si tu es convaincu que l'autre lecture est meilleure
- Tu peux aussi proposer une troisième lecture si les deux sont incorrectes

─── FORMAT DE RÉPONSE ───
STUDENT_ANSWER_READ: [ta lecture finale]
GRADE: [note]/{max_points}
CONFIDENCE: [0.0-1.0]
INTERNAL_REASONING: [analyse comparative]
STUDENT_FEEDBACK: [feedback]"""
            else:
                exchange_prompt = f"""─── DISPUTED QUESTION ───
{kwargs.get('question_text', '')}

Criteria: {kwargs.get('criteria', 'Not specified')}
Max points: {max_points}

─── READING EXCHANGE ───
You read: "{my_reading}"
You graded: {my_grade}/{max_points}

The other grader read: "{other_reading}"
They graded: {other_grade}/{max_points}

─── COMPARATIVE ANALYSIS ───
1. Compare the two readings - is there a reading error?
2. If the other reading seems more accurate, adopt it
3. If your reading seems more accurate, maintain it with justification
4. Decide on your final grade

RULES:
- Handwriting reading errors are common - stay open-minded
- Only change if you are convinced the other reading is better
- You can also propose a third reading if both are incorrect

─── RESPONSE FORMAT ───
STUDENT_ANSWER_READ: [your final reading]
GRADE: [grade]/{max_points}
CONFIDENCE: [0.0-1.0]
INTERNAL_REASONING: [comparative analysis]
STUDENT_FEEDBACK: [feedback]"""

            prompts_sent[f"llm{i+1}"] = exchange_prompt.strip()

            try:
                # FRESH call with images RE-SENT
                new_result = provider.grade_with_vision(
                    question_text=kwargs.get("question_text", ""),
                    criteria=exchange_prompt,
                    image_path=image_paths,
                    max_points=max_points,
                    class_context=kwargs.get("class_context", ""),
                    language=language
                )
                if asyncio.iscoroutine(new_result):
                    new_result = await new_result

                # Warn if grade couldn't be parsed
                if new_result.get("grade") is None:
                    import logging
                    logging.warning(f"Failed to parse grade from {name} exchange response. Using isolated grade.")
                    new_result["grade"] = my_grade
                    new_result["_parse_failed"] = True

                verified.append(new_result)
            except Exception as e:
                import logging
                logging.error(f"Exchange verification failed for {name}: {e}")
                verified.append(isolated_results[i])

        return verified, prompts_sent

    async def _run_ultimatum_round(
        self,
        round1_results: List[Dict],
        original_results: List[Dict],
        **kwargs
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Ultimatum round: final attempt when disagreement persists after cross-verification.

        Args:
            round1_results: Results from first verification round
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
            other_original_grade = original_results[1 - i].get("grade", 0)
            my_original_grade = original_results[i].get("grade", 0)
            my_round1_grade = round1_results[i].get("grade", 0)

            language = kwargs.get("language", "fr")

            # Detect if this LLM changed their grade
            i_changed = abs(my_round1_grade - my_original_grade) > 0.01
            other_changed = abs(other_grade - other_original_grade) > 0.01

            # Build evolution summary
            if language == "fr":
                my_evolution = f"{my_original_grade} → {my_round1_grade}" if i_changed else f"{my_original_grade} (maintenue)"
                other_evolution = f"{other_original_grade} → {other_grade}" if other_changed else f"{other_original_grade} (maintenue)"
                change_warning = "\n⚠ ATTENTION: Tu as MODIFIÉ ta note après avoir vu l'avis de l'autre. Confirme que ce changement est justifié objectivement." if i_changed else ""
            else:
                my_evolution = f"{my_original_grade} → {my_round1_grade}" if i_changed else f"{my_original_grade} (maintained)"
                other_evolution = f"{other_original_grade} → {other_grade}" if other_changed else f"{other_original_grade} (maintained)"
                change_warning = "\n⚠ WARNING: You CHANGED your grade after seeing the other's opinion. Confirm that this change is objectively justified." if i_changed else ""

            if language == "fr":
                ultimatum_prompt = f"""
─── ULTIMATUM - DÉCISION FINALE ───
DÉSACCORD PERSISTANT après vérification croisée:
- Ta note: {my_evolution}/{kwargs.get('max_points', 5)}
- Autre note: {other_evolution}/{kwargs.get('max_points', 5)}
{change_warning}
Son raisonnement: {other_reasoning[:400]}

─── RÉEXAMEN INDÉPENDANT ───
1. ANALYSE objectivement la réponse de l'élève
2. Identifie ce qui est correct et ce qui ne l'est pas
3. Prends TA décision finale

Tu dois choisir:
- Option A: Accepter l'autre note → explique pourquoi cette analyse est meilleure
- Option B: Maintenir ta note → arguments précis qui justifient ta position
- SI INCERTAIN: abaisse ta CONFIANCE (< 0.5)

INTERDICTION: Ne choisis pas au hasard. Chaque décision doit être justifiée.
"""
            else:
                ultimatum_prompt = f"""
─── ULTIMATUM - FINAL DECISION ───
PERSISTENT DISAGREEMENT after cross-verification:
- Your grade: {my_evolution}/{kwargs.get('max_points', 5)}
- Other grade: {other_evolution}/{kwargs.get('max_points', 5)}
{change_warning}
Their reasoning: {other_reasoning[:400]}

─── INDEPENDENT RE-EXAMINATION ───
1. ANALYZE the student's answer objectively
2. Identify what is correct and what is not
3. Make YOUR final decision

You must choose:
- Option A: Accept the other grade → explain why this analysis is better
- Option B: Maintain your grade → precise arguments supporting your position
- IF UNCERTAIN: lower your CONFIDENCE (< 0.5)

FORBIDDEN: Don't choose randomly. Every decision must be justified.
"""

            # Store the prompt
            prompts_sent[f"llm{i+1}"] = ultimatum_prompt.strip()

            try:
                new_result = provider.grade_with_vision(
                    question_text=kwargs.get("question_text", ""),
                    criteria=ultimatum_prompt,
                    image_path=kwargs.get("image_path"),
                    max_points=kwargs.get("max_points", 5),
                    class_context=kwargs.get("class_context", ""),
                    language=language
                )
                if asyncio.iscoroutine(new_result):
                    new_result = await new_result
                verified.append(new_result)
            except Exception as e:
                verified.append(round1_results[i])

        return verified, prompts_sent

    async def _run_ultimatum_with_fresh_call(
        self,
        round1_results: List[Dict],
        original_results: List[Dict],
        image_paths: List[str],
        **kwargs
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Ultimatum with FRESH call + explicit evolution context + images re-sent.

        Final attempt when disagreement persists after cross-verification.
        Each LLM gets a completely fresh call with images re-sent.

        Args:
            round1_results: Results from first verification round
            original_results: Original grading results (before any verification)
            image_paths: List of image paths to RE-SEND
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
            other_original_grade = original_results[1 - i].get("grade", 0)
            my_original_grade = original_results[i].get("grade", 0)
            my_round1_grade = round1_results[i].get("grade", 0)

            language = kwargs.get("language", "fr")
            max_points = kwargs.get("max_points", 5)

            # Detect if this LLM changed their grade
            i_changed = abs(my_round1_grade - my_original_grade) > 0.01
            other_changed = abs(other_grade - other_original_grade) > 0.01

            # Build evolution summary
            if language == "fr":
                my_evolution = f"{my_original_grade} → {my_round1_grade}" if i_changed else f"{my_original_grade} (maintenue)"
                other_evolution = f"{other_original_grade} → {other_grade}" if other_changed else f"{other_original_grade} (maintenue)"
                change_warning = "\n⚠ ATTENTION: Tu as MODIFIÉ ta note après avoir vu l'avis de l'autre. Confirme que ce changement est justifié objectivement." if i_changed else ""
            else:
                my_evolution = f"{my_original_grade} → {my_round1_grade}" if i_changed else f"{my_original_grade} (maintained)"
                other_evolution = f"{other_original_grade} → {other_grade}" if other_changed else f"{other_original_grade} (maintained)"
                change_warning = "\n⚠ WARNING: You CHANGED your grade after seeing the other's opinion. Confirm that this change is objectively justified." if i_changed else ""

            if language == "fr":
                ultimatum_prompt = f"""
─── ULTIMATUM - DÉCISION FINALE ───
Question: {kwargs.get('question_text', '')}
Note maximale: {max_points} points

DÉSACCORD PERSISTANT après vérification croisée:
- Ta note: {my_evolution}/{max_points}
- Autre note: {other_evolution}/{max_points}
{change_warning}
Son raisonnement: {other_reasoning[:400]}

─── RÉEXAMEN INDÉPENDANT ───
1. RELIS TOI-MÊME la réponse de l'élève sur les images
2. ANALYSE objectivement ce qui est correct et ce qui ne l'est pas
3. Prends TA décision finale

Tu dois choisir:
- Option A: Accepter l'autre note → explique pourquoi cette analyse est meilleure
- Option B: Maintenir ta note → arguments précis qui justifient ta position
- SI INCERTAIN: abaisse ta CONFIANCE (< 0.5)

INTERDICTION: Ne choisis pas au hasard. Chaque décision doit être justifiée.

─── FORMAT DE RÉPONSE REQUIS ───
STUDENT_ANSWER_READ: [ce que tu lis toi-même sur la copie]
GRADE: [ta note finale]/{max_points}
CONFIDENCE: [0.0-1.0]
INTERNAL_REASONING: [ton analyse finale]
STUDENT_FEEDBACK: [feedback pour l'élève]
"""
            else:
                ultimatum_prompt = f"""
─── ULTIMATUM - FINAL DECISION ───
Question: {kwargs.get('question_text', '')}
Max points: {max_points}

PERSISTENT DISAGREEMENT after cross-verification:
- Your grade: {my_evolution}/{max_points}
- Other grade: {other_evolution}/{max_points}
{change_warning}
Their reasoning: {other_reasoning[:400]}

─── INDEPENDENT RE-EXAMINATION ───
1. RE-READ the student's answer yourself from the images
2. ANALYZE objectively what is correct and what is not
3. Make YOUR final decision

You must choose:
- Option A: Accept the other grade → explain why this analysis is better
- Option B: Maintain your grade → precise arguments supporting your position
- IF UNCERTAIN: lower your CONFIDENCE (< 0.5)

FORBIDDEN: Don't choose randomly. Every decision must be justified.

─── REQUIRED RESPONSE FORMAT ───
STUDENT_ANSWER_READ: [what you read yourself from the copy]
GRADE: [your final grade]/{max_points}
CONFIDENCE: [0.0-1.0]
INTERNAL_REASONING: [your final analysis]
STUDENT_FEEDBACK: [feedback for the student]
"""

            prompts_sent[f"llm{i+1}"] = ultimatum_prompt.strip()

            try:
                # FRESH call with images RE-SENT
                new_result = provider.grade_with_vision(
                    question_text=kwargs.get("question_text", ""),
                    criteria=ultimatum_prompt,
                    image_path=image_paths,
                    max_points=max_points,
                    class_context=kwargs.get("class_context", ""),
                    language=language
                )
                if asyncio.iscoroutine(new_result):
                    new_result = await new_result

                # Warn if grade couldn't be parsed
                if new_result.get("grade") is None:
                    import logging
                    logging.warning(f"Failed to parse grade from {name} ultimatum response. Using round1 grade.")
                    new_result["grade"] = my_round1_grade
                    new_result["_parse_failed"] = True

                verified.append(new_result)
            except Exception as e:
                import logging
                logging.error(f"Ultimatum failed for {name}: {e}")
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
        reading_disagreement_callback: callable = None
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
                    questions, image_paths, language
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

        # ===== PHASE 3: Targeted Verification with FRESH calls (images RE-SENT) =====
        final_results = {}
        verification_audit = {}

        # Questions with agreement: use single-pass results directly
        for qid in report.agreed_questions:
            llm1_q = single_pass_results[provider_names[0]].questions.get(qid)
            llm2_q = single_pass_results[provider_names[1]].questions.get(qid)

            # Use LLM1's result as base (they agreed anyway)
            final_results[qid] = {
                "grade": llm1_q.grade,
                "confidence": (llm1_q.confidence + llm2_q.confidence) / 2,
                "student_answer_read": llm1_q.student_answer_read,
                "student_feedback": llm1_q.feedback,
                "internal_reasoning": llm1_q.reasoning,
                "method": "single_pass_consensus"
            }
            verification_audit[qid] = {
                "method": "single_pass",
                "agreement": True,
                "llm1": llm1_q.to_dict(),
                "llm2": llm2_q.to_dict(),
                "final": final_results[qid]
            }

        # Questions with disagreement: run verification with FRESH calls (images RE-SENT)
        for disagreement in report.flagged_questions:
            qid = disagreement.question_id

            # Note: verification_start is emitted by session.py

            # Find the question definition
            q_def = next((q for q in questions if q["id"] == qid), None)
            if not q_def:
                continue

            # Build initial results from single-pass for this question
            sp_llm1_q = single_pass_results[provider_names[0]].questions.get(qid)
            sp_llm2_q = single_pass_results[provider_names[1]].questions.get(qid)

            initial_results = [
                {
                    "grade": sp_llm1_q.grade if sp_llm1_q else 0,
                    "confidence": sp_llm1_q.confidence if sp_llm1_q else 0.5,
                    "student_answer_read": sp_llm1_q.student_answer_read if sp_llm1_q else "",
                    "internal_reasoning": sp_llm1_q.reasoning if sp_llm1_q else "",
                    "student_feedback": sp_llm1_q.feedback if sp_llm1_q else ""
                },
                {
                    "grade": sp_llm2_q.grade if sp_llm2_q else 0,
                    "confidence": sp_llm2_q.confidence if sp_llm2_q else 0.5,
                    "student_answer_read": sp_llm2_q.student_answer_read if sp_llm2_q else "",
                    "internal_reasoning": sp_llm2_q.reasoning if sp_llm2_q else "",
                    "student_feedback": sp_llm2_q.feedback if sp_llm2_q else ""
                }
            ]

            # Run verification with FRESH call (images RE-SENT)
            verified_results, verification_prompts = await self._run_verification_with_fresh_call(
                initial_results,
                image_paths,  # RE-SEND images
                question_text=q_def["text"],
                criteria=q_def["criteria"],
                max_points=q_def["max_points"],
                language=language
            )

            verified_grades = [r.get("grade", 0) for r in verified_results]
            verified_readings = [r.get("student_answer_read", "") for r in verified_results]
            verified_result = None
            method = "verification"
            agreement = True

            # Check if still disagreeing after ISOLATED verification
            if len(verified_grades) == 2 and verified_grades[0] != verified_grades[1]:
                # Phase 2b: Exchange verification - share readings between LLMs
                exchange_results, exchange_prompts = await self._run_exchange_verification(
                    verified_results,
                    image_paths,  # RE-SEND images
                    question_text=q_def["text"],
                    criteria=q_def["criteria"],
                    max_points=q_def["max_points"],
                    language=language
                )
                exchange_grades = [r.get("grade", 0) for r in exchange_results]

                # Check if still disagreeing after EXCHANGE
                if len(exchange_grades) == 2 and exchange_grades[0] != exchange_grades[1]:
                    # Phase 3: Ultimatum - final decision
                    final_results_list, ultimatum_prompts = await self._run_ultimatum_with_fresh_call(
                        exchange_results,
                        verified_results,  # Pass verified results as "original"
                        image_paths,  # RE-SEND images again
                        question_text=q_def["text"],
                        criteria=q_def["criteria"],
                        max_points=q_def["max_points"],
                        language=language
                    )
                    final_grades = [r.get("grade", 0) for r in final_results_list]

                    if len(final_grades) == 2 and final_grades[0] != final_grades[1]:
                        # Persistent disagreement - average
                        final_grade = sum(final_grades) / 2
                        agreement = False
                        method = "ultimatum_averaged"
                    else:
                        final_grade = final_grades[0] if final_grades else 0
                        agreement = True
                        method = "ultimatum_consensus"

                    verified_result = {
                        "grade": final_grade,
                        "confidence": min(r.get("confidence", 0.5) for r in final_results_list),
                        "student_answer_read": final_results_list[0].get("student_answer_read", ""),
                        "student_feedback": final_results_list[0].get("student_feedback", ""),
                        "internal_reasoning": final_results_list[0].get("internal_reasoning", ""),
                        "comparison": {
                            "initial": {"llm1": initial_results[0], "llm2": initial_results[1]},
                            "after_isolated_verification": {"llm1": verified_results[0], "llm2": verified_results[1]},
                            "after_exchange": {"llm1": exchange_results[0], "llm2": exchange_results[1]},
                            "after_ultimatum": {"llm1": final_results_list[0], "llm2": final_results_list[1]},
                            "final": {"grade": final_grade, "agreement": agreement}
                        }
                    }
                else:
                    # Agreement reached after EXCHANGE
                    verified_result = {
                        "grade": exchange_grades[0] if exchange_grades else 0,
                        "confidence": (exchange_results[0].get("confidence", 0.5) + exchange_results[1].get("confidence", 0.5)) / 2,
                        "student_answer_read": exchange_results[0].get("student_answer_read", ""),
                        "student_feedback": exchange_results[0].get("student_feedback", ""),
                        "internal_reasoning": exchange_results[0].get("internal_reasoning", ""),
                        "comparison": {
                            "initial": {"llm1": initial_results[0], "llm2": initial_results[1]},
                            "after_isolated_verification": {"llm1": verified_results[0], "llm2": verified_results[1]},
                            "after_exchange": {"llm1": exchange_results[0], "llm2": exchange_results[1]},
                            "final": {"grade": exchange_grades[0] if exchange_grades else 0, "agreement": True}
                        }
                    }
                    method = "exchange_consensus"
                    agreement = True
            else:
                # Agreement reached after ISOLATED verification
                verified_result = {
                    "grade": verified_grades[0] if verified_grades else 0,
                    "confidence": (verified_results[0].get("confidence", 0.5) + verified_results[1].get("confidence", 0.5)) / 2,
                    "student_answer_read": verified_results[0].get("student_answer_read", ""),
                    "student_feedback": verified_results[0].get("student_feedback", ""),
                    "internal_reasoning": verified_results[0].get("internal_reasoning", ""),
                    "comparison": {
                        "initial": {"llm1": initial_results[0], "llm2": initial_results[1]},
                        "after_isolated_verification": {"llm1": verified_results[0], "llm2": verified_results[1]},
                        "final": {"grade": verified_grades[0] if verified_grades else 0, "agreement": True}
                    }
                }
                method = "consensus"
                agreement = True

            final_results[qid] = {
                "grade": verified_result.get("grade"),
                "confidence": verified_result.get("confidence"),
                "student_answer_read": verified_result.get("student_answer_read", ""),
                "student_feedback": verified_result.get("student_feedback", ""),
                "internal_reasoning": verified_result.get("internal_reasoning", ""),
                "method": method
            }

            verification_audit[qid] = {
                "method": method,
                "agreement": agreement,
                "single_pass": {
                    "llm1": sp_llm1_q.to_dict() if sp_llm1_q else None,
                    "llm2": sp_llm2_q.to_dict() if sp_llm2_q else None,
                    "flagged_reason": disagreement.reason
                },
                "verification": verified_result.get("comparison") if verified_result else None,
                "final": final_results[qid]
            }

        # ===== PHASE 4: Assemble Final Results =====
        total_duration = (time.time() - total_start) * 1000

        await self._notify_progress('grading_complete', {
            'total_questions': len(questions),
            'single_pass_agreed': len(report.agreed_questions),
            'verified': len(report.flagged_questions),
            'total_duration_ms': round(total_duration, 1)
        })

        return {
            "questions": final_results,
            "audit": {
                "method": "dual_llm",
                "single_pass": {
                    provider_names[0]: single_pass_results[provider_names[0]].to_dict(),
                    provider_names[1]: single_pass_results[provider_names[1]].to_dict()
                },
                "disagreement_report": report.to_dict(),
                "verification": verification_audit,
                "timing": {
                    "total_ms": round(total_duration, 1)
                }
            },
            "summary": {
                "total_questions": len(questions),
                "agreed_in_single_pass": len(report.agreed_questions),
                "required_verification": len(report.flagged_questions),
                "agreement_rate": report.agreement_rate,
                "total_score": sum(r["grade"] or 0 for r in final_results.values()),
                "max_score": sum(q["max_points"] for q in questions)
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
                disagreement_callback=disagreement_callback,
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
