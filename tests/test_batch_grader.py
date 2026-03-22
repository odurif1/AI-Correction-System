from core.services.graders.batch_grader import _merge_copy_question_comparison
from audit.builder import build_audit_from_llm_comparison, extract_final_question_outputs


def test_merge_copy_question_comparison_keeps_all_questions():
    merged = _merge_copy_question_comparison(
        existing_questions={
            "Q6": {
                "LLM1": {"grade": 2.0, "max_points": 2.0, "confidence": 0.9},
                "LLM2": {"grade": 1.5, "max_points": 2.0, "confidence": 0.8},
                "_initial_final": {"agreement": False},
            }
        },
        final_questions={
            "Q1": {"grade": 1.0, "max_points": 1.0, "confidence": 0.95, "feedback": "OK"},
            "Q6": {"grade": 1.75, "max_points": 2.0, "confidence": 0.8},
        },
        llm1_questions={
            "Q1": {"grade": 1.0, "max_points": 1.0, "confidence": 0.95},
            "Q6": {"grade": 2.0, "max_points": 2.0, "confidence": 0.9},
        },
        llm2_questions={
            "Q1": {"grade": 1.0, "max_points": 1.0, "confidence": 0.9},
            "Q6": {"grade": 1.5, "max_points": 2.0, "confidence": 0.8},
        },
        llm1_name="LLM1",
        llm2_name="LLM2",
        grading_scale={"Q1": 1.0, "Q6": 2.0},
    )

    assert list(merged.keys()) == ["Q1", "Q6"]
    assert merged["Q1"]["LLM1"]["grade"] == 1.0
    assert merged["Q1"]["LLM2"]["grade"] == 1.0
    assert merged["Q1"]["final"]["grade"] == 1.0
    assert merged["Q6"]["final"]["agreement"] is False


def test_build_audit_keeps_single_mode_feedback_from_direct_question_payload():
    audit = build_audit_from_llm_comparison(
        {
            "options": {"providers": ["LLM1: gemini-2.5-flash-lite"]},
            "llm_comparison": {
                "copy_1": {
                    "questions": {
                        "Q1": {
                            "LLM1: gemini-2.5-flash-lite": {
                                "grade": 1.0,
                                "max_points": 1.0,
                                "feedback": "Bien identifié.",
                                "confidence": 1.0,
                            },
                            "grade": 1.0,
                            "max_points": 1.0,
                            "feedback": "Bien identifié.",
                            "confidence": 1.0,
                            "method": "single_llm",
                            "agreement": True,
                        }
                    }
                }
            },
        },
        mode="single",
        grading_method="batch",
        verification_mode="per-copy",
        provider_names=["LLM1: gemini-2.5-flash-lite"],
        grading_scale={"Q1": 1.0},
    )

    final_outputs = extract_final_question_outputs(audit)
    assert final_outputs["Q1"]["feedback"] == "Bien identifié."
