"""
Single-Pass Grader for stateless architecture.

Grades all questions in a single API call, returning structured JSON.
Used as Phase 1 of the stateless grading approach.

Each call is independent with images sent fresh.
"""

import json
import re
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class QuestionResult:
    """Result for a single question from single-pass grading."""
    question_id: str
    location: str
    student_answer_read: str
    grade: float
    confidence: float
    reasoning: str
    feedback: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location": self.location,
            "student_answer_read": self.student_answer_read,
            "grade": self.grade,
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
        language: str = "fr"
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

        Returns:
            SinglePassResult with all question grades
        """
        from config.prompts import build_multi_question_grading_prompt

        start_time = time.time()

        # Build the prompt
        prompt = build_multi_question_grading_prompt(questions, language)

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
                conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', q_block)
                read_match = re.search(r'"student_answer_read"\s*:\s*"([^"]*)"', q_block)
                feedback_match = re.search(r'"feedback"\s*:\s*"([^"]*)"', q_block)
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', q_block)
                loc_match = re.search(r'"location"\s*:\s*"([^"]*)"', q_block)

                data["questions"][qid] = {
                    "grade": float(grade_match.group(1)) if grade_match else 0.0,
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
        """Build SinglePassResult from parsed data."""
        questions_result = {}

        for q in questions:
            qid = q["id"]
            q_data = data.get("questions", {}).get(qid, {})

            # Validate grade is within bounds
            grade = float(q_data.get("grade", 0))
            max_points = q.get("max_points", 5)
            grade = max(0, min(grade, max_points))

            questions_result[qid] = QuestionResult(
                question_id=qid,
                location=q_data.get("location", ""),
                student_answer_read=q_data.get("student_answer_read", ""),
                grade=grade,
                confidence=float(q_data.get("confidence", 0.5)),
                reasoning=q_data.get("reasoning", ""),
                feedback=q_data.get("feedback", "")
            )

        return SinglePassResult(
            student_name=data.get("student_name"),
            questions=questions_result,
            overall_comments=data.get("overall_comments", ""),
            raw_response=raw_response,
            parse_success=len(questions_result) == len(questions),
            parse_errors=parse_errors,
            duration_ms=duration_ms
        )
