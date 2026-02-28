"""
AuditBuilder - Builds unified grading audit from various sources.

This module provides a centralized way to construct the unified
GradingAudit structure, ensuring consistency across all grading modes.
"""

from typing import Dict, List, Optional, Any
from core.models import (
    GradingAudit,
    ProviderInfo,
    LLMResult,
    ResolutionInfo,
    QuestionAudit,
    StudentDetectionAudit,
    AuditSummary,
)


class AuditBuilder:
    """
    Builds unified grading audit from various sources.

    Usage:
        builder = AuditBuilder(
            mode="dual",
            grading_method="batch",
            verification_mode="grouped"
        )

        # Add providers
        builder.add_provider("LLM1", "gemini-2.5-flash", tokens={"prompt": 15000, "completion": 3000})
        builder.add_provider("LLM2", "gpt-4o", tokens={"prompt": 15000, "completion": 3000})

        # Add question results
        builder.add_question_result(
            question_id="Q1",
            llm_results={
                "LLM1": {"grade": 1.0, "max_points": 1.0, "reading": "...", "reasoning": "...", "feedback": "...", "confidence": 0.9},
                "LLM2": {"grade": 1.0, "max_points": 1.0, "reading": "...", "reasoning": "...", "feedback": "...", "confidence": 0.95}
            },
            resolution={"final_grade": 1.0, "final_max_points": 1.0, "method": "consensus", "phases": ["initial"], "agreement": True}
        )

        # Build the audit
        audit = builder.build()
    """

    def __init__(
        self,
        mode: str,
        grading_method: str,
        verification_mode: str
    ):
        """
        Initialize the AuditBuilder.

        Args:
            mode: "single" or "dual"
            grading_method: "batch", "individual", or "hybrid"
            verification_mode: "grouped", "per-copy", "per-question", or "none"
        """
        self.mode = mode
        self.grading_method = grading_method
        self.verification_mode = verification_mode
        self.providers: List[ProviderInfo] = []
        self.questions: Dict[str, QuestionAudit] = {}
        self.student_detection: Optional[StudentDetectionAudit] = None

    def add_provider(
        self,
        id: str,
        model: str,
        tokens: Dict[str, int] = None
    ) -> None:
        """
        Add a provider (LLM1, LLM2).

        Args:
            id: Provider ID ("LLM1" or "LLM2")
            model: Model name (e.g., "gemini-2.5-flash")
            tokens: Optional token usage {"prompt": int, "completion": int}
        """
        self.providers.append(ProviderInfo(
            id=id,
            model=model,
            tokens=tokens
        ))

    def add_question_result(
        self,
        question_id: str,
        llm_results: Dict[str, Dict[str, Any]],
        resolution: Dict[str, Any]
    ) -> None:
        """
        Add a question's LLM results and resolution.

        Args:
            question_id: Question identifier (e.g., "Q1")
            llm_results: Dict mapping provider ID to result
                {"LLM1": {"grade": 1.0, "max_points": 1.0, "reading": "...", ...}}
            resolution: Resolution info
                {"final_grade": 1.0, "final_max_points": 1.0, "method": "consensus",
                 "phases": ["initial"], "agreement": True, "initial_reading_similarity": 0.95}
        """
        # Convert llm_results dicts to LLMResult models
        llm_result_models = {}
        for provider_id, result in llm_results.items():
            max_pts = result.get("max_points")
            llm_result_models[provider_id] = LLMResult(
                grade=float(result.get("grade", 0)),
                max_points=float(max_pts) if max_pts is not None else None,
                reading=result.get("reading", "") or result.get("student_answer_read", ""),
                reasoning=result.get("reasoning", ""),
                feedback=result.get("feedback", "") or result.get("student_feedback", ""),
                confidence=float(result.get("confidence", 0.8))
            )

        # Convert resolution dict to ResolutionInfo model
        final_max_pts = resolution.get("final_max_points")
        resolution_model = ResolutionInfo(
            final_grade=float(resolution.get("final_grade", 0)),
            final_max_points=float(final_max_pts) if final_max_pts is not None else 0.0,
            method=resolution.get("method", "unknown"),
            phases=resolution.get("phases", ["initial"]),
            agreement=resolution.get("agreement"),
            initial_reading_similarity=resolution.get("initial_reading_similarity")
        )

        self.questions[question_id] = QuestionAudit(
            llm_results=llm_result_models,
            resolution=resolution_model
        )

    def set_student_detection(
        self,
        final_name: str,
        llm_results: Dict[str, str],
        resolution: Dict[str, Any]
    ) -> None:
        """
        Set student name detection audit.

        Args:
            final_name: Final resolved student name
            llm_results: Dict mapping provider ID to detected name
                {"LLM1": "Jean Dupont", "LLM2": "J. Dupont"}
            resolution: Resolution info
                {"method": "consensus", "phases": ["initial"], "agreement": True}
        """
        resolution_model = ResolutionInfo(
            final_grade=0,  # Not applicable for name detection
            final_max_points=0,  # Not applicable
            method=resolution.get("method", "unknown"),
            phases=resolution.get("phases", ["initial"]),
            agreement=resolution.get("agreement")
        )

        self.student_detection = StudentDetectionAudit(
            final_name=final_name,
            llm_results=llm_results,
            resolution=resolution_model
        )

    def _compute_summary(self) -> AuditSummary:
        """Compute summary statistics from questions."""
        total = len(self.questions)
        agreed_initial = 0
        required_verification = 0
        required_ultimatum = 0

        for qid, qaudit in self.questions.items():
            # Check if agreed initially
            if qaudit.resolution.agreement and qaudit.resolution.phases == ["initial"]:
                agreed_initial += 1

            # Check if verification was needed
            if "verification" in qaudit.resolution.phases:
                required_verification += 1

            # Check if ultimatum was needed
            if "ultimatum" in qaudit.resolution.phases:
                required_ultimatum += 1

        # Calculate final agreement rate
        final_agreed = sum(
            1 for q in self.questions.values()
            if q.resolution.agreement is True
        )
        agreement_rate = final_agreed / total if total > 0 else 1.0

        return AuditSummary(
            total_questions=total,
            agreed_initial=agreed_initial,
            required_verification=required_verification,
            required_ultimatum=required_ultimatum,
            final_agreement_rate=round(agreement_rate, 3)
        )

    def build(self) -> GradingAudit:
        """
        Build the final audit object.

        Returns:
            GradingAudit with all collected data
        """
        return GradingAudit(
            mode=self.mode,
            grading_method=self.grading_method,
            verification_mode=self.verification_mode,
            providers=self.providers,
            questions=self.questions,
            student_detection=self.student_detection,
            summary=self._compute_summary()
        )


def build_audit_from_llm_comparison(
    llm_comparison_data: Dict[str, Any],
    mode: str,
    grading_method: str,
    verification_mode: str,
    provider_names: List[str] = None,
    grading_scale: Dict[str, float] = None
) -> GradingAudit:
    """
    Build a GradingAudit from legacy llm_comparison data structure.

    This is a helper function to migrate from the old llm_comparison
    structure to the new unified GradingAudit structure.

    Args:
        llm_comparison_data: The old llm_comparison dict
        mode: "single" or "dual"
        grading_method: "batch", "individual", or "hybrid"
        verification_mode: "grouped", "per-copy", "per-question", or "none"
        provider_names: List of provider names ["gemini-2.5-flash", "gpt-4o"]
        grading_scale: Dict of {question_id: max_points} - source of truth for max_points

    Returns:
        GradingAudit with converted data
    """
    builder = AuditBuilder(
        mode=mode,
        grading_method=grading_method,
        verification_mode=verification_mode
    )

    # Add providers
    options = llm_comparison_data.get("options", {})
    providers = options.get("providers", provider_names or [])

    for i, provider_name in enumerate(providers):
        builder.add_provider(f"LLM{i+1}", provider_name)

    # Process each copy's questions
    llm_comparison = llm_comparison_data.get("llm_comparison", {})

    for copy_key, copy_data in llm_comparison.items():
        if not isinstance(copy_data, dict):
            continue

        # Get provider names from options
        llm1_name = providers[0] if len(providers) > 0 else "LLM1"
        llm2_name = providers[1] if len(providers) > 1 else "LLM2"

        # Process questions
        questions = copy_data.get("questions", {})
        for qid, qdata in questions.items():
            if not isinstance(qdata, dict):
                continue

            # Get max_points from grading_scale (source of truth)
            question_max_points = grading_scale.get(qid) if grading_scale else None

            # Fallback: try to get max_points from LLM data if not in grading_scale
            if question_max_points is None:
                # Check LLM1 data
                llm1_mp = qdata.get(llm1_name, {}).get("max_points")
                llm2_mp = qdata.get(llm2_name, {}).get("max_points")
                # Use LLM1's max_points, or LLM2's if LLM1 doesn't have it
                question_max_points = llm1_mp if llm1_mp is not None else llm2_mp

            # Build llm_results
            llm_results = {}

            # LLM1 data
            llm1_data = qdata.get(llm1_name, {})
            if llm1_data:
                # Use grading_scale for max_points, not LLM-reported value
                llm1_data_copy = dict(llm1_data)
                if question_max_points is not None:
                    llm1_data_copy["max_points"] = question_max_points
                llm_results["LLM1"] = llm1_data_copy

            # LLM2 data
            llm2_data = qdata.get(llm2_name, {})
            if llm2_data:
                # Use grading_scale for max_points, not LLM-reported value
                llm2_data_copy = dict(llm2_data)
                if question_max_points is not None:
                    llm2_data_copy["max_points"] = question_max_points
                llm_results["LLM2"] = llm2_data_copy

            # Build resolution (phases block removed - redundant with llm_results)
            agreement = True

            # Calculate initial agreement (for resolution)
            if llm_results:
                initial_grade1 = llm1_data.get("grade", 0) if llm1_data else 0
                initial_grade2 = llm2_data.get("grade", 0) if llm2_data else 0
                # Use grading_scale for max_points in agreement calculation
                mp_for_threshold = question_max_points if question_max_points is not None else 1.0

                # Check agreement (within threshold - default 10%)
                try:
                    from config.settings import get_settings
                    threshold = mp_for_threshold * get_settings().grade_agreement_threshold
                except Exception:
                    threshold = mp_for_threshold * 0.1  # Default 10%
                agreement = abs(initial_grade1 - initial_grade2) < threshold

            # Build phases list (for tracking which phases occurred)
            phases_list = []
            if qdata.get("verification"):
                phases_list.append("verification")
            if qdata.get("ultimatum"):
                phases_list.append("ultimatum")
            if not phases_list:
                phases_list = ["initial"]  # No verification/ultimatum = stayed at initial

            final_data = qdata.get("final", qdata.get("_initial_final", qdata.get("_pending_final", {})))

            # Use grading_scale for final_max_points
            final_max_pts = question_max_points if question_max_points is not None else final_data.get("max_points")

            resolution = {
                "final_grade": final_data.get("grade", 0),
                "final_max_points": final_max_pts,
                "method": final_data.get("method", "unknown"),
                "phases": phases_list,
                "agreement": final_data.get("agreement", agreement),
                "initial_reading_similarity": qdata.get("_initial_final", {}).get("reading_similarity")
            }

            builder.add_question_result(
                question_id=qid,
                llm_results=llm_results,
                resolution=resolution
            )

        # Process student detection
        student_detection = copy_data.get("student_detection", {})
        student_name_section = copy_data.get("student_name", {})

        if student_detection or student_name_section:
            # Get LLM results
            llm1_name_detected = None
            llm2_name_detected = None

            if student_detection:
                llm1_name_detected = student_detection.get("llm1_student_name")
                llm2_name_detected = student_detection.get("llm2_student_name")

            # Check for name resolution section
            final_name_section = student_name_section.get("final", {})
            initial_section = student_name_section.get("initial", {})

            if initial_section:
                llm1_name_detected = initial_section.get("llm1_name", llm1_name_detected)
                llm2_name_detected = initial_section.get("llm2_name", llm2_name_detected)

            final_name = final_name_section.get("resolved_name") or student_detection.get("final_resolved_name", "")

            if final_name:
                llm_results_names = {}
                if llm1_name_detected:
                    llm_results_names["LLM1"] = llm1_name_detected
                if llm2_name_detected:
                    llm_results_names["LLM2"] = llm2_name_detected

                # Build resolution for name
                name_resolution = {
                    "method": final_name_section.get("method", "unknown"),
                    "phases": ["initial"],
                    "agreement": initial_section.get("agreement", False) if initial_section else True
                }

                if student_name_section.get("verification"):
                    name_resolution["phases"].append("verification")
                if student_name_section.get("ultimatum"):
                    name_resolution["phases"].append("ultimatum")

                builder.set_student_detection(
                    final_name=final_name,
                    llm_results=llm_results_names,
                    resolution=name_resolution
                )

    return builder.build()
