"""
Single-Pass Grader for stateless architecture.

Grades all questions in a single API call, returning structured JSON.
Used as Phase 1 of the stateless grading approach.

Each call is independent with images sent fresh.
"""

import asyncio
import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class QuestionResult:
    """Result for a single question from single-pass grading."""
    question_id: str
    question_text: str  # Detected question text (for auto-detect mode)
    location: str
    student_answer_read: str
    grade: float
    max_points: float  # Detected from the copy
    confidence: float
    reasoning: str
    feedback: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_text": self.question_text,
            "location": self.location,
            "student_answer_read": self.student_answer_read,
            "grade": self.grade,
            "max_points": self.max_points,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "feedback": self.feedback
        }


@dataclass
class SinglePassResult:
    """Complete result from single-pass grading."""
    student_name: Optional[str]
    questions: Dict[str, QuestionResult]
    overall_comments: str
    raw_response: str
    parse_success: bool
    parse_errors: List[str]
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "student_name": self.student_name,
            "questions": {qid: q.to_dict() for qid, q in self.questions.items()},
            "overall_comments": self.overall_comments,
            "raw_response": self.raw_response[:5000] if self.raw_response else "",  # Truncate for storage
            "parse_success": self.parse_success,
            "parse_errors": self.parse_errors,
            "duration_ms": self.duration_ms
        }


class SinglePassGrader:
    """
    Grade all questions in a single API call.

    This is Phase 1 of the stateless grading:
    - Send all questions + all images in one call
    - Receive structured JSON with all grades
    - Fast and efficient for the common case (agreement)

    Each call is independent with fresh images.
    """

    def __init__(self, provider):
        """
        Initialize single-pass grader.

        Args:
            provider: LLM provider (GeminiProvider or OpenAIProvider)
        """
        self.provider = provider

    async def grade_all_questions(
        self,
        questions: List[Dict[str, Any]],
        image_paths: List[str],
        language: str = "fr",
        second_reading: bool = False
    ) -> SinglePassResult:
        """
        Grade all questions in a single API call.

        Args:
            questions: List of question dicts with:
                - id: Question identifier
                - text: Question text
                - criteria: Grading criteria
                - max_points: Maximum points
            image_paths: List of image paths (all pages)
            language: Language for prompts
            second_reading: If True, include second reading instruction in prompt

        Returns:
            SinglePassResult with all question grades
        """
        from config.prompts import build_multi_question_grading_prompt, build_auto_detect_grading_prompt

        start_time = time.time()

        # Build the prompt - use auto-detect if no questions provided
        if not questions:
            prompt = build_auto_detect_grading_prompt(language, second_reading=second_reading)
        else:
            prompt = build_multi_question_grading_prompt(questions, language, second_reading=second_reading)

        # Add multi-page context
        num_pages = len(image_paths)
        if num_pages > 1:
            if language == "fr":
                page_context = f"\n\nIMPORTANT: Tu as accès à {num_pages} pages de cette copie. Cherche les réponses sur TOUTES les pages."
            else:
                page_context = f"\n\nIMPORTANT: You have access to {num_pages} pages of this copy. Search for answers on ALL pages."
            prompt = prompt + page_context

        try:
            # Fresh call with images
            response = self.provider.call_vision(
                prompt=prompt,
                image_path=image_paths if len(image_paths) > 1 else image_paths[0]
            )

            duration_ms = (time.time() - start_time) * 1000

            # Parse the response
            result = self._parse_response(response, questions, duration_ms)

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            # Return empty result on error
            return SinglePassResult(
                student_name=None,
                questions={},
                overall_comments="",
                raw_response=str(e),
                parse_success=False,
                parse_errors=[f"API call failed: {str(e)}"],
                duration_ms=duration_ms
            )

    def _parse_response(
        self,
        response: str,
        questions: List[Dict[str, Any]],
        duration_ms: float
    ) -> SinglePassResult:
        """
        Parse JSON response from LLM.

        Uses multiple strategies:
        1. Direct JSON parse
        2. Extract JSON from markdown code blocks
        3. Regex extraction of key fields

        Args:
            response: Raw response string
            questions: Original questions (for validation)
            duration_ms: Time taken for the call

        Returns:
            SinglePassResult
        """
        parse_errors = []

        # Strategy 1: Direct JSON parse
        try:
            data = json.loads(response.strip())
            return self._build_result(data, questions, response, duration_ms, [])
        except json.JSONDecodeError as e:
            parse_errors.append(f"Direct parse failed: {str(e)}")

        # Strategy 2: Extract from markdown code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return self._build_result(data, questions, response, duration_ms, parse_errors)
            except json.JSONDecodeError as e:
                parse_errors.append(f"Code block parse failed: {str(e)}")

        # Strategy 3: Regex extraction
        try:
            data = self._extract_with_regex(response, questions)
            if data:
                return self._build_result(data, questions, response, duration_ms, parse_errors)
        except Exception as e:
            parse_errors.append(f"Regex extraction failed: {str(e)}")

        # All strategies failed
        parse_errors.append("Could not parse response as JSON")
        return SinglePassResult(
            student_name=None,
            questions={},
            overall_comments="",
            raw_response=response,
            parse_success=False,
            parse_errors=parse_errors,
            duration_ms=duration_ms
        )

    def _extract_with_regex(
        self,
        response: str,
        questions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract question data using regex patterns.

        Fallback when JSON parsing fails.
        """
        data = {
            "student_name": None,
            "questions": {},
            "overall_comments": ""
        }

        # Try to extract student name
        name_match = re.search(r'"student_name"\s*:\s*"([^"]*)"', response)
        if name_match:
            data["student_name"] = name_match.group(1)

        # Extract each question's data
        for q in questions:
            qid = q["id"]
            pattern = rf'"{qid}"\s*:\s*\{{([^}}]*)\}}'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                q_block = match.group(1)

                # Extract fields
                grade_match = re.search(r'"grade"\s*:\s*([\d.]+)', q_block)
                max_pts_match = re.search(r'"max_points"\s*:\s*([\d.]+)', q_block)
                conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', q_block)
                read_match = re.search(r'"student_answer_read"\s*:\s*"([^"]*)"', q_block)
                feedback_match = re.search(r'"feedback"\s*:\s*"([^"]*)"', q_block)
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', q_block)
                loc_match = re.search(r'"location"\s*:\s*"([^"]*)"', q_block)

                data["questions"][qid] = {
                    "grade": float(grade_match.group(1)) if grade_match else 0.0,
                    "max_points": float(max_pts_match.group(1)) if max_pts_match else q.get("max_points", 1.0),
                    "confidence": float(conf_match.group(1)) if conf_match else 0.5,
                    "student_answer_read": read_match.group(1) if read_match else "",
                    "feedback": feedback_match.group(1) if feedback_match else "",
                    "reasoning": reasoning_match.group(1) if reasoning_match else "",
                    "location": loc_match.group(1) if loc_match else ""
                }

        # Check if we got at least some questions
        if data["questions"]:
            return data
        return None

    def _build_result(
        self,
        data: Dict[str, Any],
        questions: List[Dict[str, Any]],
        raw_response: str,
        duration_ms: float,
        parse_errors: List[str]
    ) -> SinglePassResult:
        """Build SinglePassResult from parsed data.

        If questions list is empty (auto-detect mode), uses all questions
        returned by the LLM. Otherwise, only processes questions from the list.
        """
        questions_result = {}
        llm_questions = data.get("questions", {})

        # Auto-detect mode: use all questions returned by LLM
        if not questions:
            for qid, q_data in llm_questions.items():
                # Skip non-question keys
                if not (qid.startswith('Q') and qid[1:].replace('_', '').isdigit()):
                    continue

                max_points = float(q_data.get("max_points", 1.0))
                grade = float(q_data.get("grade", 0))
                grade = max(0, min(grade, max_points))

                questions_result[qid] = QuestionResult(
                    question_id=qid,
                    question_text=q_data.get("question_text", ""),
                    location=q_data.get("location", ""),
                    student_answer_read=q_data.get("student_answer_read", ""),
                    grade=grade,
                    max_points=max_points,
                    confidence=float(q_data.get("confidence", 0.5)),
                    reasoning=q_data.get("reasoning", ""),
                    feedback=q_data.get("feedback", "")
                )
        else:
            # Normal mode: process only questions from the list
            for q in questions:
                qid = q["id"]
                q_data = llm_questions.get(qid, {})

                # Get max_points from LLM response (detected) or fallback to question default
                max_points = float(q_data.get("max_points", q.get("max_points", 1.0)))

                # Validate grade is within bounds
                grade = float(q_data.get("grade", 0))
                grade = max(0, min(grade, max_points))

                questions_result[qid] = QuestionResult(
                    question_id=qid,
                    question_text=q_data.get("question_text", q.get("text", "")),
                    location=q_data.get("location", ""),
                    student_answer_read=q_data.get("student_answer_read", ""),
                    grade=grade,
                    max_points=max_points,
                    confidence=float(q_data.get("confidence", 0.5)),
                    reasoning=q_data.get("reasoning", ""),
                    feedback=q_data.get("feedback", "")
                )

        # In auto-detect mode, success means we detected at least one question
        # In normal mode, success means we got all expected questions
        if not questions:
            parse_success = len(questions_result) > 0
        else:
            parse_success = len(questions_result) == len(questions)

        return SinglePassResult(
            student_name=data.get("student_name"),
            questions=questions_result,
            overall_comments=data.get("overall_comments", ""),
            raw_response=raw_response,
            parse_success=parse_success,
            parse_errors=parse_errors,
            duration_ms=duration_ms
        )

    async def grade_with_self_verification(
        self,
        questions: List[Dict[str, Any]],
        image_paths: List[str],
        language: str = "fr"
    ) -> Tuple[SinglePassResult, Dict[str, Any]]:
        """
        Grade with two passes for self-verification.

        Pass 1: Initial grading
        Pass 2: Self-verification with own results

        Args:
            questions: List of question dicts
            image_paths: List of image paths
            language: Language for prompts

        Returns:
            Tuple of (final_result, verification_audit)
        """
        # Phase 1: Initial grading
        first_result = await self.grade_all_questions(questions, image_paths, language)

        if not first_result.parse_success:
            # If first pass failed, return as-is
            return first_result, {"phase1_success": False, "error": "First pass parsing failed"}

        # Phase 2: Self-verification
        verification_prompt = self._build_self_verification_prompt(
            questions, first_result, language
        )

        # Add multi-page context
        num_pages = len(image_paths)
        if num_pages > 1:
            if language == "fr":
                page_context = f"\n\nIMPORTANT: Tu as accès à {num_pages} pages de cette copie."
            else:
                page_context = f"\n\nIMPORTANT: You have access to {num_pages} pages of this copy."
            verification_prompt = verification_prompt + page_context

        start_time = time.time()
        try:
            response = self.provider.call_vision(
                prompt=verification_prompt,
                image_path=image_paths if len(image_paths) > 1 else image_paths[0]
            )
            duration_ms = (time.time() - start_time) * 1000

            # Parse the verification response
            second_result = self._parse_response(response, questions, duration_ms)

            # Build audit trail
            verification_audit = {
                "phase1_success": True,
                "phase2_success": second_result.parse_success,
                "phase1_duration_ms": first_result.duration_ms,
                "phase2_duration_ms": second_result.duration_ms,
                "phase1_questions": {qid: q.to_dict() for qid, q in first_result.questions.items()},
                "phase2_questions": {qid: q.to_dict() for qid, q in second_result.questions.items()},
                "changes": self._detect_changes(first_result, second_result)
            }

            # Return the second result (verified)
            return second_result, verification_audit

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            # Return first result if verification fails
            return first_result, {
                "phase1_success": True,
                "phase2_success": False,
                "phase2_error": str(e),
                "phase1_duration_ms": first_result.duration_ms,
                "phase2_duration_ms": duration_ms
            }

    def _build_self_verification_prompt(
        self,
        questions: List[Dict[str, Any]],
        first_result: SinglePassResult,
        language: str = "fr"
    ) -> str:
        """Build prompt for self-verification phase."""
        # Format first results as JSON for the prompt
        first_results_json = {
            "student_name": first_result.student_name,
            "questions": {}
        }
        for qid, q_result in first_result.questions.items():
            first_results_json["questions"][qid] = {
                "student_answer_read": q_result.student_answer_read,
                "grade": q_result.grade,
                "max_points": q_result.max_points,
                "confidence": q_result.confidence,
                "reasoning": q_result.reasoning
            }

        results_str = json.dumps(first_results_json, indent=2, ensure_ascii=False)

        if language == "fr":
            return f"""Tu as déjà corrigé cette copie. Voici tes premiers résultats:

```json
{results_str}
```

TÂCHE: Relis cette correction et vérifie:
1. Les notes sont-elles cohérentes avec le barème détecté?
2. Les lectures des réponses sont-elles exactes?
3. Y a-t-il des erreurs évidentes à corriger?

Si tu dois ajuster une note, explique pourquoi dans le champ "reasoning".
Réponds avec le MÊME format JSON que précédemment.

IMPORTANT: Ne change pas tes notes "juste au cas où". Change SEULEMENT si tu identifies une erreur réelle."""
        else:
            return f"""You have already graded this copy. Here are your first results:

```json
{results_str}
```

TASK: Re-read this grading and verify:
1. Are grades consistent with the detected scale?
2. Are the answer readings accurate?
3. Are there obvious errors to correct?

If you need to adjust a grade, explain why in the "reasoning" field.
Respond with the SAME JSON format as before.

IMPORTANT: Do not change your grades "just in case". Change ONLY if you identify a real error."""

    def _detect_changes(
        self,
        first_result: SinglePassResult,
        second_result: SinglePassResult
    ) -> Dict[str, Any]:
        """Detect what changed between first and second pass."""
        changes = {}
        for qid in first_result.questions.keys():
            if qid in second_result.questions:
                q1 = first_result.questions[qid]
                q2 = second_result.questions[qid]
                if abs(q1.grade - q2.grade) > 0.01:
                    changes[qid] = {
                        "grade_changed": True,
                        "grade_before": q1.grade,
                        "grade_after": q2.grade,
                        "reading_changed": q1.student_answer_read != q2.student_answer_read,
                        "reasoning_after": q2.reasoning[:200] if q2.reasoning else ""
                    }
        return changes
