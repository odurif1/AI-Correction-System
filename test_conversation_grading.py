#!/usr/bin/env python3
"""
Test script for conversation-based grading architecture.

Usage:
    python test_conversation_grading.py              # Test with mock data
    python test_conversation_grading.py --real       # Test with real LLM (requires API keys)
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_conversation_session():
    """Test the ConversationSession class."""
    from ai.conversation import ConversationSession

    print("\n━━━ TEST: ConversationSession ━━━")

    session = ConversationSession("test_copy", "TestProvider")

    # Start with images
    turn = session.start("Grade this", ["/img1.png", "/img2.png"])
    assert session.images_sent
    assert len(session.get_initial_images()) == 2

    # Add response
    session.add_response("Here are the grades...")
    assert session.get_last_response() == "Here are the grades..."

    # Continue
    session.continue_("Verify Q2")
    assert session.get_turn_count() == 3

    # Check formats
    openai_fmt = session.to_openai_format()
    assert len(openai_fmt) == 3

    # Summary
    summary = session.get_summary()
    assert summary["estimated_tokens"] > 0

    print(f"  Session: {session}")
    print(f"  Turns: {session.get_turn_count()}")
    print(f"  Token estimate: {summary['estimated_tokens']}")
    print("\n  ✓ All assertions passed")
    return True


def test_disagreement_analyzer():
    """Test the disagreement analyzer with mock data."""
    from ai.disagreement_analyzer import DisagreementAnalyzer

    print("\n━━━ TEST: DisagreementAnalyzer ━━━")

    analyzer = DisagreementAnalyzer()

    # Mock results from two LLMs
    llm1_results = {
        "questions": {
            "Q1": {"grade": 2.0, "confidence": 0.9, "student_answer_read": "m = V × Cm"},
            "Q2": {"grade": 1.5, "confidence": 0.8, "student_answer_read": "C = n/V"},
            "Q3": {"grade": 0.0, "confidence": 0.7, "student_answer_read": "non visible"},
        }
    }

    llm2_results = {
        "questions": {
            "Q1": {"grade": 2.0, "confidence": 0.85, "student_answer_read": "m = V x Cm"},  # Accord (différence mineure)
            "Q2": {"grade": 0.5, "confidence": 0.75, "student_answer_read": "C = n/V"},     # Désaccord note
            "Q3": {"grade": 2.0, "confidence": 0.8, "student_answer_read": "V = 100 mL"},   # Conflit trouvé/non trouvé
        }
    }

    report = analyzer.analyze(llm1_results, llm2_results)

    print(f"  Total questions: {report.total_questions}")
    print(f"  Agreed: {len(report.agreed_questions)} → {report.agreed_questions}")
    print(f"  Flagged: {len(report.flagged_questions)}")
    print(f"  Agreement rate: {report.agreement_rate:.0%}")

    for d in report.flagged_questions:
        print(f"\n  ⚠ {d.question_id}: {d.reason}")
        print(f"     LLM1: {d.llm1_grade} | LLM2: {d.llm2_grade}")

    # Assertions
    assert report.total_questions == 3, "Should have 3 questions"
    assert len(report.agreed_questions) == 1, "Q1 should be agreed"
    assert len(report.flagged_questions) == 2, "Q2 and Q3 should be flagged"
    assert report.agreement_rate == 1/3, "Agreement rate should be 33%"

    print("\n  ✓ All assertions passed")
    return True


def test_single_pass_grader_mock():
    """Test single-pass grader with mock provider."""
    from ai.single_pass_grader import SinglePassGrader

    print("\n━━━ TEST: SinglePassGrader (mock) ━━━")

    # Mock provider with conversation support
    class MockProvider:
        def __init__(self):
            self._conversation_manager = type('obj', (object,), {
                'create_session': lambda s, c, p: type('obj', (object,), {
                    'session_id': c,
                    'provider_name': p,
                    'turns': [],
                    'images_sent': False,
                    '_initial_images': [],
                    'start': lambda slf, prompt, images: slf.turns.append(type('obj', (object,), {'role': 'user', 'content': prompt, 'images': images})()),
                    'continue_': lambda slf, prompt: slf.turns.append(type('obj', (object,), {'role': 'user', 'content': prompt, 'images': None})()),
                    'add_response': lambda slf, resp: slf.turns.append(type('obj', (object,), {'role': 'assistant', 'content': resp})()),
                    'get_turn_count': lambda slf: len(slf.turns),
                })(),
                'get_session': lambda s, c, p: None,
                'get_or_create': lambda s, c, p: s.create_session(c, p),
                'end_session': lambda s, c, p: None,
                'clear_all': lambda s: None,
            })()

        def create_conversation(self, copy_id):
            from ai.conversation import ConversationSession
            return ConversationSession(copy_id, "MockProvider")

        def get_or_create_conversation(self, copy_id):
            return self.create_conversation(copy_id)

        def end_conversation(self, copy_id):
            pass

        def call_vision(self, prompt, image_path):
            return '''```json
{
  "student_name": "Test Student",
  "questions": {
    "Q1": {
      "location": "page 1, haut",
      "student_answer_read": "m = V × Cm",
      "grade": 2.0,
      "confidence": 0.9,
      "reasoning": "Formule correcte",
      "feedback": "Exact"
    },
    "Q2": {
      "location": "page 1, bas",
      "student_answer_read": "C = n/V",
      "grade": 1.0,
      "confidence": 0.7,
      "reasoning": "Partiellement correct",
      "feedback": "Incomplet"
    }
  }
}
```'''

        def call_vision_with_conversation(self, conversation, prompt, images=None):
            # Simulate conversation-based call
            if images:
                conversation.start(prompt, images)
            else:
                conversation.continue_(prompt)
            response = self.call_vision(prompt, images)
            conversation.add_response(response)
            return response

        def _parse_grading_response(self, response):
            from ai.response_parser import parse_grading_response
            return parse_grading_response(response)

    grader = SinglePassGrader(MockProvider())

    questions = [
        {"id": "Q1", "text": "Question 1", "criteria": "Critères", "max_points": 2},
        {"id": "Q2", "text": "Question 2", "criteria": "Critères", "max_points": 2},
    ]

    # Run async
    result = asyncio.run(grader.grade_all_questions(questions, ["test.png"], "fr"))

    print(f"  Parse success: {result.parse_success}")
    print(f"  Student name: {result.student_name}")
    print(f"  Questions graded: {len(result.questions)}")

    for qid, q in result.questions.items():
        print(f"    {qid}: {q.grade}/{questions[0]['max_points']} (conf: {q.confidence})")

    # Assertions
    assert result.parse_success, "Parsing should succeed"
    assert result.student_name == "Test Student"
    assert len(result.questions) == 2
    assert result.questions["Q1"].grade == 2.0
    assert result.questions["Q2"].grade == 1.0

    print("\n  ✓ All assertions passed")
    return True


async def test_with_real_llm():
    """Test conversation-based grading with real LLM calls."""
    from ai.comparison_provider import ComparisonProvider
    from ai.gemini_provider import GeminiProvider
    from config.settings import get_settings

    print("\n━━━ TEST: Conversation-based grading with real LLM ━━━")

    settings = get_settings()

    # Check if we have API keys
    if not settings.gemini_api_key:
        print("  ⚠ No GEMINI_API_KEY set, skipping real LLM test")
        return None

    # Create providers
    providers = []

    # LLM1: Gemini
    providers.append(("gemini", GeminiProvider(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model
    )))

    # LLM2: Gemini with different model (or OpenAI if available)
    if settings.openai_api_key:
        from ai.openai_provider import OpenAIProvider
        providers.append(("openai", OpenAIProvider(
            api_key=settings.openai_api_key,
            model="gpt-4o"
        )))
    else:
        # Use same Gemini with flash model
        providers.append(("gemini-flash", GeminiProvider(
            api_key=settings.gemini_api_key,
            model="models/gemini-2.5-flash"
        )))

    comparison = ComparisonProvider(providers)

    # Test questions
    questions = [
        {
            "id": "Q1",
            "text": "Donner la formule de la concentration molaire",
            "criteria": "C = n/V, unité mol/L",
            "max_points": 2
        },
        {
            "id": "Q2",
            "text": "Quel est le rôle du témoin dans une expérience?",
            "criteria": "Permet de comparer, valider les résultats",
            "max_points": 2
        }
    ]

    # Check if we have test images
    test_images = list(Path("copies").glob("*.png")) + list(Path("copies").glob("*.jpg"))
    if not test_images:
        print("  ⚠ No test images found in 'copies/' directory")
        print("  Creating a simple test image...")
        # Create a simple test image
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Q1: C = n/V", fill='black')
        draw.text((10, 50), "Q2: Témoin = comparaison", fill='black')
        img.save("test_copy.png")
        test_images = [Path("test_copy.png")]

    print(f"  Using images: {[str(p) for p in test_images]}")

    # Run conversation-based grading
    result = await comparison.grade_copy_conversation(
        questions=questions,
        image_paths=[str(p) for p in test_images],
        language="fr"
    )

    print(f"\n  Method: {result['audit']['method']}")
    print(f"  Total questions: {result['summary']['total_questions']}")
    print(f"  Agreed in single-pass: {result['summary']['agreed_in_single_pass']}")
    print(f"  Required verification: {result['summary']['required_verification']}")
    print(f"  Agreement rate: {result['summary']['agreement_rate']:.0%}")
    print(f"  Total score: {result['summary']['total_score']}/{result['summary']['max_score']}")

    for qid, q in result['questions'].items():
        print(f"\n  {qid}: {q['grade']:.1f} (method: {q['method']})")
        print(f"    Reading: {q['student_answer_read'][:50]}...")
        print(f"    Feedback: {q['student_feedback']}")

    # Cleanup
    if Path("test_copy.png").exists():
        Path("test_copy.png").unlink()

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("CONVERSATION-BASED GRADING TESTS")
    print("=" * 60)

    use_real_llm = "--real" in sys.argv

    # Test 1: Conversation session
    try:
        test_conversation_session()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return 1

    # Test 2: Disagreement analyzer (no API calls)
    try:
        test_disagreement_analyzer()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return 1

    # Test 3: Single-pass grader (mock)
    try:
        test_single_pass_grader_mock()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return 1

    # Test 4: Real LLM (optional)
    if use_real_llm:
        try:
            result = asyncio.run(test_with_real_llm())
            if result is None:
                print("\n  (Real LLM test skipped)")
            elif not result:
                return 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("\n━━━ TEST: Real LLM (skipped) ━━━")
        print("  Run with --real to test with actual LLM calls")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
