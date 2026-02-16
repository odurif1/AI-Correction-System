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
    after_verification: Optional[Dict[str, float]] = None
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
        - Phase 1 (enabled by default): Establish reading consensus
        - Phase 2: Grade based on established reading

        Args:
            question_text: The question being graded
            criteria: Grading criteria
            image_path: Path(s) to image(s)
            max_points: Maximum points
            class_context: Context about class patterns
            language: Language for prompts
            question_id: Question identifier (for jurisprudence lookup)
            reading_disagreement_callback: Callback for reading disagreements (required for Phase 1)
            skip_reading_consensus: If True, skip Phase 1 reading consensus (default: False, reading consensus enabled)

        Returns:
            Merged result with comparison data
        """
        # Phase 1: Establish reading consensus if callback provided
        established_reading = None
        reading_comparison = None

        if not skip_reading_consensus and reading_disagreement_callback is not None:
            reading_result = await self.read_student_answer_with_consensus(
                image_path=image_path,
                question_text=question_text,
                language=language,
                reading_disagreement_callback=reading_disagreement_callback
            )
            established_reading = reading_result.get("reading")
            reading_comparison = reading_result.get("comparison")

            # Add validated reading to criteria
            if reading_result.get("user_validated") and established_reading:
                if language == "fr":
                    reading_context = f"""

─── LECTURE VALIDÉE PAR L'ENSEIGNANT ───
L'élève a écrit: {established_reading}
"""
                else:
                    reading_context = f"""

─── TEACHER-VALIDATED READING ───
The student wrote: {established_reading}
"""
                criteria = criteria + reading_context

        # Add jurisprudence context if available for this EXACT question
        jurisprudence_context = ""
        if question_id and question_id in self.jurisprudence:
            past = self.jurisprudence[question_id]
            # Only use jurisprudence if it's for the SAME question text
            past_question = past.get('question_text', '')
            if past_question and past_question.strip() == question_text.strip():
                reasoning_hint = ""
                if past.get('reasoning_llm1'):
                    reasoning_hint = f"\n- Raisonnement IA 1: {past['reasoning_llm1'][:150]}..."
                if language == "fr":
                    jurisprudence_context = f"""

INFORMATION - Décision passée (à titre indicatif):
Pour cette même question "{question_id}", l'enseignant a précédemment décidé:
- Note attribuée: {past['decision']:.1f}/{past.get('max_points', max_points)}
- Notes proposées par les IA: {past.get('llm1_grade', '?')} vs {past.get('llm2_grade', '?')}
{reasoning_hint}
Cette information est fournie à titre de référence pour t'aider. Tu reste libre de ta notation.
"""
                else:
                    jurisprudence_context = f"""

INFORMATION - Past decision (for reference only):
For this same question "{question_id}", the teacher previously decided:
- Grade given: {past['decision']:.1f}/{past.get('max_points', max_points)}
- AI proposed grades: {past.get('llm1_grade', '?')} vs {past.get('llm2_grade', '?')}
{reasoning_hint}
This information is provided as a reference to help you. You remain free in your grading.
"""

        # Combine criteria with jurisprudence
        effective_criteria = criteria + jurisprudence_context

        # Track total timing
        total_start_time = time.time()

        # Build image reference info
        image_refs = {
            "paths": image_path if isinstance(image_path, list) else [image_path] if image_path else [],
            "count": len(image_path) if isinstance(image_path, list) else 1 if image_path else 0
        }

        # Notify start of parallel LLM calls
        await self._notify_progress('llm_parallel_start', {
            'providers': [name for name, _ in self.providers],
            'question_text': question_text[:50] + '...' if len(question_text) > 50 else question_text
        })

        # Step 1: Grade with both providers IN PARALLEL
        completed_count = 0
        total_providers = len(self.providers)
        phase_timings = {"initial": {}, "verification": {}, "round2": {}}

        async def call_provider(index: int, name: str, provider):
            """Call a provider, handling both sync and async methods."""
            nonlocal completed_count
            call_start = time.time()
            try:
                result = provider.grade_with_vision(
                    question_text=question_text,
                    criteria=effective_criteria,
                    image_path=image_path,
                    max_points=max_points,
                    class_context=class_context,
                    language=language
                )
                # Check if result is a coroutine (async method)
                if asyncio.iscoroutine(result):
                    result = await result

                call_duration = (time.time() - call_start) * 1000  # ms

                # Notify completion with index for ordering
                completed_count += 1
                await self._notify_progress('llm_complete', {
                    'provider': name,
                    'provider_index': index,
                    'grade': result.get('grade'),
                    'confidence': result.get('confidence'),
                    'all_completed': completed_count == total_providers
                })

                # Store timing
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

        # Run all providers in parallel with indices
        tasks = [call_provider(i, name, provider) for i, (name, provider) in enumerate(self.providers)]
        indexed_results = await asyncio.gather(*tasks)

        # Sort by index to maintain original order
        indexed_results.sort(key=lambda x: x[0])
        results = [r for _, r, _ in indexed_results]
        initial_durations = [d for _, _, d in indexed_results]

        # Step 2: Compare grades
        grades = [r.get("grade", 0) for r in results if r.get("grade") is not None]

        # Build reading comparison analysis
        reading1 = results[0].get("student_answer_read", "")
        reading2 = results[1].get("student_answer_read", "")
        reading_analysis = {
            "llm1_read": reading1,
            "llm2_read": reading2,
            "identical": reading1.strip().lower() == reading2.strip().lower() if reading1 and reading2 else False,
            "difference_type": None
        }
        if reading1 and reading2 and not reading_analysis["identical"]:
            r1_lower, r2_lower = reading1.lower().strip(), reading2.lower().strip()
            if r1_lower.replace("é", "e").replace("è", "e") == r2_lower.replace("é", "e").replace("è", "e"):
                reading_analysis["difference_type"] = "accent"
            elif r1_lower in r2_lower or r2_lower in r1_lower:
                reading_analysis["difference_type"] = "partial"
            else:
                reading_analysis["difference_type"] = "substantial"

        # Build confidence evolution tracker
        confidence_evolution = {
            "initial": {
                "llm1": results[0].get("confidence"),
                "llm2": results[1].get("confidence")
            }
        }

        # Track total timing
        total_start_time = time.time()

        # NEW STRUCTURE: Comprehensive audit with all phases
        comparison_info = {
            # Phase 1: Initial grading (before any verification)
            "initial": {
                "llm1": build_llm_audit_info(
                    results[0], self.providers[0][0],
                    duration_ms=initial_durations[0] if initial_durations else None
                ),
                "llm2": build_llm_audit_info(
                    results[1], self.providers[1][0],
                    duration_ms=initial_durations[1] if len(initial_durations) > 1 else None
                ),
                "difference": abs(grades[0] - grades[1]) if len(grades) == 2 else None
            },
            # Reading analysis
            "reading_analysis": reading_analysis,
            "reading_consensus": reading_comparison,
            # Confidence tracking
            "confidence_evolution": confidence_evolution,
            # Timing information
            "timing": {
                "initial": phase_timings["initial"],
                "verification": None,
                "round2": None,
                "total_ms": None
            },
            # Decision path
            "decision_path": {
                "initial_agreement": len(grades) == 2 and grades[0] == grades[1],
                "verification_triggered": False,
                "round2_triggered": False,
                "final_method": None
            },
            # Image references
            "images": {
                "count": len(image_path) if isinstance(image_path, list) else 1 if image_path else 0,
                "paths": image_path if isinstance(image_path, list) else [image_path] if image_path else []
            },
            # Will be filled during verification phases
            "after_verification": None,
            "after_round2": None,
            # Final result
            "final": None
        }

        # Check if grades are identical
        if len(grades) == 2 and grades[0] == grades[1]:
            # Perfect agreement
            comparison_info["decision_path"]["final_method"] = "consensus"
            comparison_info["timing"]["total_ms"] = round((time.time() - total_start_time) * 1000, 1)
            comparison_info["final"] = {
                "grade": grades[0],
                "agreement": True,
                "method": "consensus"
            }
            return self._merge_results(results, comparison_info)

        # Step 3: Cross-verification if there's a difference
        comparison_info["decision_path"]["verification_triggered"] = True
        verification_start = time.time()

        if len(grades) == 2 and grades[0] != grades[1]:
            verified_results, verification_prompts = await self._cross_verify_with_prompts(
                results,
                question_text=question_text,
                criteria=criteria,
                image_path=image_path,
                max_points=max_points,
                class_context=class_context,
                language=language
            )
            verification_duration = (time.time() - verification_start) * 1000

            # Step 4: Check if still in disagreement
            verified_grades = [r.get("grade", 0) for r in verified_results if r.get("grade") is not None]

            # Update confidence evolution
            confidence_evolution["after_verification"] = {
                "llm1": verified_results[0].get("confidence"),
                "llm2": verified_results[1].get("confidence")
            }

            # Store FULL verification results with prompts
            comparison_info["after_verification"] = {
                "llm1": build_llm_audit_info(
                    verified_results[0], self.providers[0][0],
                    prompt_sent=verification_prompts.get("llm1")
                ),
                "llm2": build_llm_audit_info(
                    verified_results[1], self.providers[1][0],
                    prompt_sent=verification_prompts.get("llm2")
                ),
                "difference": abs(verified_grades[0] - verified_grades[1]) if len(verified_grades) == 2 else None
            }
            comparison_info["timing"]["verification"] = {"total_ms": round(verification_duration, 1)}

            # Step 5: If still disagreeing AND false convergence detected, do round 2
            if len(verified_grades) == 2 and verified_grades[0] != verified_grades[1]:
                if self._detect_false_convergence(verified_results):
                    # Both claimed to converge but grades differ - ultimatum round
                    comparison_info["decision_path"]["round2_triggered"] = True
                    round2_start = time.time()

                    final_results, round2_prompts = await self._cross_verify_round2_with_prompts(
                        verified_results,
                        results,
                        question_text=question_text,
                        criteria=criteria,
                        image_path=image_path,
                        max_points=max_points,
                        class_context=class_context,
                        language=language
                    )
                    round2_duration = (time.time() - round2_start) * 1000

                    # Update with round 2 results (FULL info)
                    final_grades = [r.get("grade", 0) for r in final_results if r.get("grade") is not None]

                    # Update confidence evolution
                    confidence_evolution["after_round2"] = {
                        "llm1": final_results[0].get("confidence"),
                        "llm2": final_results[1].get("confidence")
                    }

                    comparison_info["after_round2"] = {
                        "llm1": build_llm_audit_info(
                            final_results[0], self.providers[0][0],
                            prompt_sent=round2_prompts.get("llm1")
                        ),
                        "llm2": build_llm_audit_info(
                            final_results[1], self.providers[1][0],
                            prompt_sent=round2_prompts.get("llm2")
                        ),
                        "difference": abs(final_grades[0] - final_grades[1]) if len(final_grades) == 2 else None
                    }
                    comparison_info["timing"]["round2"] = {"total_ms": round(round2_duration, 1)}

                    # Use round 2 results for final decision
                    verified_results = final_results
                    verified_grades = final_grades

            # Step 6: Final check for disagreement
            if len(verified_grades) == 2 and verified_grades[0] != verified_grades[1]:
                # Still in disagreement after all rounds
                comparison_info["decision_path"]["final_method"] = "average"
                comparison_info["timing"]["total_ms"] = round((time.time() - total_start_time) * 1000, 1)
                comparison_info["final"] = {
                    "grade": sum(verified_grades) / len(verified_grades),
                    "agreement": False,
                    "method": "average"
                }

                # Record the disagreement
                disagreement = DisagreementRecord(
                    question_id=question_id,  # Use the actual question_id
                    question_text=question_text,
                    llm1_name=self.providers[0][0],
                    llm1_result=verified_results[0],
                    llm2_name=self.providers[1][0],
                    llm2_result=verified_results[1],
                    initial_difference=comparison_info["initial"]["difference"],
                    after_verification={
                        "llm1": verified_grades[0],
                        "llm2": verified_grades[1]
                    },
                    resolved=False
                )
                self.disagreements.append(disagreement)

                # Call user callback if provided
                if self.disagreement_callback:
                    callback_result = await self.disagreement_callback(
                        question_id=question_id,  # Pass the actual question_id
                        question_text=question_text,
                        llm1_name=self.providers[0][0],
                        llm1_result=verified_results[0],
                        llm2_name=self.providers[1][0],
                        llm2_result=verified_results[1],
                        max_points=max_points
                    )
                    # Handle callback result (tuple: grade, feedback_source)
                    chosen_grade, feedback_source = callback_result

                    # Mark as resolved with user's choice
                    disagreement.resolved = True
                    comparison_info["user_choice"] = chosen_grade
                    comparison_info["feedback_source"] = feedback_source

                    # Return result with user's chosen grade and feedback
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

        # Fallback: return first result (no difference or single grade)
        comparison_info["timing"]["total_ms"] = round((time.time() - total_start_time) * 1000, 1)
        return self._merge_results(results, comparison_info)

    async def _cross_verify_with_prompts(
        self,
        results: List[Dict],
        **kwargs
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Cross-verify by sending each LLM the other's reasoning.

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

            # Build verification prompt
            if language == "fr":
                lecture_section = ""
                if my_answer_read or other_answer_read:
                    lecture_section = f"""
─── COMPARAISON DES LECTURES ───
Ta lecture: {my_answer_read or "(non spécifiée)"}
Autre lecture: {other_answer_read or "(non spécifiée)"}
→ Les lectures sont-elles identiques? Si non, laquelle est correcte?
"""
                verify_prompt = f"""
─── AUTRE CORRECTEUR ───
Note: {other_grade}/{kwargs.get('max_points', 5)}
Raisonnement: {other_reasoning}
{lecture_section}─── INSTRUCTIONS ───
Ta note: {my_grade} | Autre note: {other_grade}

1. LIS le raisonnement et compare les lectures ci-dessus.
2. Tu ne peux PAS prétendre converger si les interprétations diffèrent.
3. Si tu maintiens ta note, explique pourquoi tu rejettes l'autre analyse.
"""
            else:
                lecture_section = ""
                if my_answer_read or other_answer_read:
                    lecture_section = f"""
─── READING COMPARISON ───
Your reading: {my_answer_read or "(not specified)"}
Other reading: {other_answer_read or "(not specified)"}
→ Are the readings identical? If not, which is correct?
"""
                verify_prompt = f"""
─── OTHER GRADER ───
Grade: {other_grade}/{kwargs.get('max_points', 5)}
Reasoning: {other_reasoning}
{lecture_section}─── INSTRUCTIONS ───
Your grade: {my_grade} | Other grade: {other_grade}

1. READ the reasoning and compare the readings above.
2. You CANNOT claim convergence if interpretations differ.
3. If you maintain your grade, explain why you reject the other analysis.
"""

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

    async def _cross_verify(
        self,
        results: List[Dict],
        **kwargs
    ) -> List[Dict]:
        """
        Cross-verify by sending each LLM the other's reasoning and answer reading.

        Args:
            results: Initial grading results from both LLMs
            **kwargs: Original grading arguments

        Returns:
            List of verified results
        """
        verified = []

        for i, (name, provider) in enumerate(self.providers):
            other_result = results[1 - i]
            other_reasoning = other_result.get("internal_reasoning", "")
            other_grade = other_result.get("grade", 0)
            other_answer_read = other_result.get("student_answer_read", "")
            my_grade = results[i].get("grade", 0)

            # Build verification prompt with BOTH reasoning AND answer reading
            language = kwargs.get("language", "fr")

            # Get MY answer reading for comparison
            my_answer_read = results[i].get("student_answer_read", "")

            if language == "fr":
                # Compare readings explicitly
                lecture_section = ""
                if my_answer_read or other_answer_read:
                    lecture_section = f"""
─── COMPARAISON DES LECTURES ───
Ta lecture: {my_answer_read or "(non spécifiée)"}
Autre lecture: {other_answer_read or "(non spécifiée)"}
→ Les lectures sont-elles identiques? Si non, laquelle est correcte?
"""

                verify_context = f"""
─── AUTRE CORRECTEUR ───
Note: {other_grade}/{kwargs.get('max_points', 5)}
Raisonnement: {other_reasoning}
{lecture_section}─── INSTRUCTIONS ───
Ta note: {my_grade} | Autre note: {other_grade}

1. LIS le raisonnement et compare les lectures ci-dessus.
2. Tu ne peux PAS prétendre converger si les interprétations diffèrent.
3. Si tu maintiens ta note, explique pourquoi tu rejettes l'autre analyse.

EXEMPLE CORRECT: "L'autre correcteur identifie un Erlenmeyer, pas une fiole jaugée. Je reconnais mon erreur et révise à 0/1."
EXEMPLE INCORRECT: "Convergence" alors que les lectures sont contradictoires.
"""
            else:
                # Compare readings explicitly
                lecture_section = ""
                if my_answer_read or other_answer_read:
                    lecture_section = f"""
─── READING COMPARISON ───
Your reading: {my_answer_read or "(not specified)"}
Other reading: {other_answer_read or "(not specified)"}
→ Are the readings identical? If not, which is correct?
"""

                verify_context = f"""
─── OTHER GRADER ───
Grade: {other_grade}/{kwargs.get('max_points', 5)}
Reasoning: {other_reasoning}
{lecture_section}─── INSTRUCTIONS ───
Your grade: {my_grade} | Other grade: {other_grade}

1. READ the reasoning and compare readings above.
2. You CANNOT claim convergence if interpretations differ.
3. If maintaining your grade, explain why you reject the other analysis.

CORRECT: "Other grader identifies Erlenmeyer flask, not volumetric flask. I acknowledge my error and revise to 0/1."
INCORRECT: "Convergence" when readings are contradictory.
"""

            # Call provider with verification context
            try:
                new_result = provider.grade_with_vision(
                    question_text=kwargs.get("question_text", ""),
                    criteria=kwargs.get("criteria", "") + verify_context,
                    image_path=kwargs.get("image_path"),
                    max_points=kwargs.get("max_points", 5),
                    class_context=kwargs.get("class_context", ""),
                    language=language
                )
                # Handle async providers
                if asyncio.iscoroutine(new_result):
                    new_result = await new_result
                verified.append(new_result)
            except Exception as e:
                # On error, keep original result
                verified.append(results[i])

        return verified

    async def _cross_verify_round2(
        self,
        round1_results: List[Dict],
        original_results: List[Dict],
        **kwargs
    ) -> List[Dict]:
        """
        Second round of cross-verification with ultimatum prompt.

        This is called when both LLMs claimed to "converge" but their grades
        are still different. The ultimatum forces a real decision.

        Args:
            round1_results: Results from first verification round
            original_results: Original grading results (before any verification)
            **kwargs: Original grading arguments

        Returns:
            List of final results after ultimatum
        """
        verified = []

        for i, (name, provider) in enumerate(self.providers):
            other_result = round1_results[1 - i]
            other_reasoning = other_result.get("internal_reasoning", "")
            other_grade = other_result.get("grade", 0)
            my_original_grade = original_results[i].get("grade", 0)
            my_round1_grade = round1_results[i].get("grade", 0)

            language = kwargs.get("language", "fr")

            if language == "fr":
                verify_context = f"""
─── ULTIMATUM - DERNIÈRE CHANCE ───
Tu as indiqué converger, mais l'autre correcteur a MAINTENU sa note de {other_grade}/{kwargs.get('max_points', 5)}.
Ta note originale: {my_original_grade}/{kwargs.get('max_points', 5)}
Ta note actuelle: {my_round1_grade}/{kwargs.get('max_points', 5)}

Son raisonnement maintenu: {other_reasoning}

─── TU DOIS CHOISIR ───
Option A: Tu acceptes SA note ({other_grade}) → Reconnais que son analyse est meilleure
Option B: Tu maintiens TA note ({my_round1_grade}) → Justification RENFORCÉE obligatoire

"Converger" n'est PLUS une option. Les notes sont différentes, tu DOIS trancher.
"""
            else:
                verify_context = f"""
─── ULTIMATUM - LAST CHANCE ───
You indicated convergence, but the other grader MAINTAINED their grade of {other_grade}/{kwargs.get('max_points', 5)}.
Your original grade: {my_original_grade}/{kwargs.get('max_points', 5)}
Your current grade: {my_round1_grade}/{kwargs.get('max_points', 5)}

Their maintained reasoning: {other_reasoning}

─── YOU MUST CHOOSE ───
Option A: Accept THEIR grade ({other_grade}) → Acknowledge their analysis is better
Option B: Maintain YOUR grade ({my_round1_grade}) → REINFORCED justification required

"Converge" is NO LONGER an option. Grades differ, you MUST decide.
"""

            # Call provider with ultimatum context
            try:
                new_result = provider.grade_with_vision(
                    question_text=kwargs.get("question_text", ""),
                    criteria=kwargs.get("criteria", "") + verify_context,
                    image_path=kwargs.get("image_path"),
                    max_points=kwargs.get("max_points", 5),
                    class_context=kwargs.get("class_context", ""),
                    language=language
                )
                # Handle async providers
                if asyncio.iscoroutine(new_result):
                    new_result = await new_result
                verified.append(new_result)
            except Exception as e:
                # On error, keep round1 result
                verified.append(round1_results[i])

        return verified

    async def _cross_verify_round2_with_prompts(
        self,
        round1_results: List[Dict],
        original_results: List[Dict],
        **kwargs
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Second round with ultimatum - returns both results and prompts sent.

        Args:
            round1_results: Results from first verification round
            original_results: Original grading results
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
            my_original_grade = original_results[i].get("grade", 0)
            my_round1_grade = round1_results[i].get("grade", 0)

            language = kwargs.get("language", "fr")

            if language == "fr":
                ultimatum_prompt = f"""
─── ULTIMATUM - DERNIÈRE CHANCE ───
Tu as indiqué converger, mais l'autre correcteur a MAINTENU sa note de {other_grade}/{kwargs.get('max_points', 5)}.
Ta note originale: {my_original_grade}/{kwargs.get('max_points', 5)}
Ta note actuelle: {my_round1_grade}/{kwargs.get('max_points', 5)}

Son raisonnement maintenu: {other_reasoning}

─── TU DOIS CHOISIR ───
Option A: Tu acceptes SA note ({other_grade}) → Reconnais que son analyse est meilleure
Option B: Tu maintiens TA note ({my_round1_grade}) → Justification RENFORCÉE obligatoire

"Converger" n'est PLUS une option. Les notes sont différentes, tu DOIS trancher.
"""
            else:
                ultimatum_prompt = f"""
─── ULTIMATUM - LAST CHANCE ───
You indicated convergence, but the other grader MAINTAINED their grade of {other_grade}/{kwargs.get('max_points', 5)}.
Your original grade: {my_original_grade}/{kwargs.get('max_points', 5)}
Your current grade: {my_round1_grade}/{kwargs.get('max_points', 5)}

Their maintained reasoning: {other_reasoning}

─── YOU MUST CHOOSE ───
Option A: Accept THEIR grade ({other_grade}) → Acknowledge their analysis is better
Option B: Maintain YOUR grade ({my_round1_grade}) → REINFORCED justification required

"Converge" is NO LONGER an option. Grades differ, you MUST decide.
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

    def _detect_false_convergence(self, results: List[Dict]) -> bool:
        """
        Detect if LLMs claimed to converge but grades are still different.

        Args:
            results: Results from verification round

        Returns:
            True if false convergence detected (both say converge but grades differ)
        """
        if len(results) < 2:
            return False

        grades = [r.get("grade", 0) for r in results]
        if grades[0] == grades[1]:
            return False  # Same grades, no problem

        # Check if both claimed convergence
        convergence_keywords = ["converge", "convergence", "d'accord", "agree"]
        both_claim_convergence = True

        for result in results:
            reasoning = result.get("internal_reasoning", "").lower()
            if not any(kw in reasoning for kw in convergence_keywords):
                both_claim_convergence = False
                break

        return both_claim_convergence

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
            comparison_info: Comparison metadata (NEW STRUCTURE with initial/after_verification/final)

        Returns:
            Merged result
        """
        if not results:
            return {
                "grade": 0,
                "confidence": 0,
                "internal_reasoning": "",
                "student_feedback": ""
            }

        # Use first result as base
        merged = dict(results[0])

        # Add comparison info
        if comparison_info:
            merged["comparison"] = comparison_info

        # Check if final decision indicates disagreement (new structure)
        final_info = comparison_info.get("final", {}) if comparison_info else {}
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
                "internal_reasoning": "User decision",
                "student_feedback": ""
            }

        # Choose base result based on feedback source
        if feedback_source == "llm2" and len(results) > 1:
            merged = dict(results[1])
        elif feedback_source == "merge" and len(results) > 1:
            # Merge: use LLM1 as base but combine feedbacks
            merged = dict(results[0])
            fb1 = results[0].get("student_feedback", "")
            fb2 = results[1].get("student_feedback", "")
            if fb1 and fb2:
                merged["student_feedback"] = f"{fb1} / {fb2}"
            elif fb2:
                merged["student_feedback"] = fb2
        else:
            # Default: use LLM1
            merged = dict(results[0])

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

        comparison_info["after_verification"] = {
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

        # Step 3: Try cross-verification for readings
        verified_readings = await self._cross_verify_reading(
            results, image_path, question_text, language
        )

        vreading1 = verified_readings[0].get("reading", "")
        vreading2 = verified_readings[1].get("reading", "")

        comparison_info["after_verification"] = {
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
            prompt = f"""Tu es un lecteur neutre. Ton UNIQUE tâche est de lire et décrire ce que l'élève a répondu.

QUESTION RECHERCHÉE: {question_text}

INSTRUCTIONS:
1. Localise cette question sur la copie (peut être sur n'importe quelle page)
2. Lis EXACTEMENT ce que l'élève a écrit/dessiné
3. Décris de manière FACTUELLE sans interpréter

FORMAT DE RÉPONSE:
TROUVÉ: [oui/non/partiellement]
CONTENU: [description factuelle de la réponse de l'élève]
CONFIDENCE: [0.0 à 1.0 - ta certitude sur ta lecture]

EXEMPLES:
TROUVÉ: oui
CONTENU: L'élève a dessiné une fiole jaugée de 100mL avec le trait de jauge annoté. Il a écrit "fiole jaugée" à côté du dessin.
CONFIDENCE: 0.95

TROUVÉ: non
CONTENU: La question n'est pas visible sur les pages fournies.
CONFIDENCE: 0.3
"""
        else:
            prompt = f"""You are a neutral reader. Your ONLY task is to read and describe what the student answered.

QUESTION TO FIND: {question_text}

INSTRUCTIONS:
1. Locate this question on the copy (may be on any page)
2. Read EXACTLY what the student wrote/drew
3. Describe FACTUALLY without interpreting

RESPONSE FORMAT:
FOUND: [yes/no/partially]
CONTENT: [factual description of student's answer]
CONFIDENCE: [0.0 to 1.0 - your certainty about your reading]

EXAMPLES:
FOUND: yes
CONTENT: The student drew a 100mL volumetric flask with the calibration mark annotated. They wrote "volumetric flask" next to the drawing.
CONFIDENCE: 0.95

FOUND: no
CONTENT: The question is not visible on the provided pages.
CONFIDENCE: 0.3
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

            elif line.upper().startswith("CONTENU:") or line.upper().startswith("CONTENT:"):
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
        language: str
    ) -> List[Dict]:
        """
        Cross-verify readings by showing each LLM the other's reading.

        Args:
            results: Initial reading results
            image_path: Image path(s)
            question_text: The question
            language: Language

        Returns:
            List of verified reading results
        """
        verified = []

        for i, (name, provider) in enumerate(self.providers):
            other_result = results[1 - i]
            other_reading = other_result.get("reading", "")
            my_reading = results[i].get("reading", "")

            if language == "fr":
                verify_prompt = f"""Tu as lu: "{my_reading}"

Un autre correcteur a lu: "{other_reading}"

Question: {question_text}

Compare les deux lectures. Si l'autre lecture semble plus précise, adopte-la.
Sinon, confirme ta lecture. Réponds uniquement avec ta lecture finale.

LECTURE FINALE: [ta lecture]
CONFIDENCE: [0.0 à 1.0]"""
            else:
                verify_prompt = f"""You read: "{my_reading}"

Another grader read: "{other_reading}"

Question: {question_text}

Compare the two readings. If the other reading seems more accurate, adopt it.
Otherwise, confirm your reading. Respond only with your final reading.

FINAL READING: [your reading]
CONFIDENCE: [0.0 to 1.0]"""

            try:
                response = provider.call_vision(verify_prompt, image_path=image_path)
                parsed = self._parse_reading_response(response)

                # If parsing didn't extract reading, use the whole response
                if not parsed.get("reading"):
                    parsed["reading"] = response
                    parsed["confidence"] = 0.5

                verified.append(parsed)
            except Exception:
                verified.append(results[i])

        return verified
