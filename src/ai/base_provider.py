"""
Base provider class for AI API interactions.

Provides shared functionality for vision and text interactions,
including image processing, token tracking, and response parsing.
"""

import base64
import time
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Dict, Any, List, Optional

from PIL import Image

from core.models import AICallResult


class BaseProvider(ABC):
    """
    Abstract base class for AI providers.

    Provides common functionality:
    - Image to base64 conversion
    - Token usage tracking
    - Response parsing
    - Common grading/analysis methods

    Subclasses must implement:
    - call_vision()
    - call_text()
    - get_embedding()
    - get_embeddings()
    """

    def __init__(self, mock_mode: bool = False):
        """Initialize base provider."""
        self.mock_mode = mock_mode
        self.call_history: List[AICallResult] = []

    # ==================== IMAGE UTILITIES ====================

    def _image_to_base64(self, image_path: str) -> str:
        """Convert an image file to base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _image_to_base64_from_pil(self, image: Image.Image) -> str:
        """Convert a PIL Image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # ==================== TOKEN TRACKING ====================

    def _log_call(
        self,
        prompt_type: str,
        input_summary: str,
        response_summary: str,
        duration_ms: float,
        prompt_tokens: int = None,
        completion_tokens: int = None
    ):
        """Log an AI call to audit trail."""
        self.call_history.append(AICallResult(
            prompt_type=prompt_type,
            input_summary=input_summary,
            response_summary=response_summary,
            duration_ms=duration_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        ))

    def get_token_usage(self) -> Dict[str, int]:
        """Get total token usage from all calls."""
        prompt_tokens = sum(c.prompt_tokens or 0 for c in self.call_history)
        completion_tokens = sum(c.completion_tokens or 0 for c in self.call_history)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "calls": len(self.call_history)
        }

    # ==================== ABSTRACT METHODS ====================

    @abstractmethod
    def call_vision(
        self,
        prompt: str,
        image_path=None,
        image_bytes: bytes = None,
        pil_image: Image.Image = None,
        response_format: str = "text"
    ) -> str:
        """Call the vision API with an image."""
        pass

    @abstractmethod
    def call_text(
        self,
        prompt: str,
        system_prompt: str = None,
        response_format: str = "text"
    ) -> str:
        """Call the text API."""
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        pass

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        pass

    # ==================== GRADING METHODS ====================

    def grade_with_vision(
        self,
        question_text: str,
        criteria: str,
        image_path,
        max_points: float,
        class_context: str = "",
        language: str = "en",
        question_id: str = "",
        reading_disagreement_callback: callable = None,
        skip_reading_consensus: bool = True
    ) -> Dict[str, Any]:
        """
        Grade a student's answer using vision.

        This is a template method that builds the prompt and parses the response.
        Subclasses can override if needed.
        """
        from config.prompts import build_vision_grading_prompt

        prompt = build_vision_grading_prompt(
            question_text=question_text,
            criteria=criteria,
            max_points=max_points,
            class_context=class_context,
            language=language
        )

        response = self.call_vision(prompt, image_path=image_path)
        return self._parse_grading_response(response)

    def _parse_grading_response(self, response: str) -> Dict[str, Any]:
        """Parse grading response using shared parser."""
        from ai.response_parser import parse_grading_response
        return parse_grading_response(response)

    # ==================== NAME DETECTION ====================

    def detect_student_name(
        self,
        image_path,
        language: str = "fr"
    ) -> Dict[str, Any]:
        """
        Detect student name from the first page of a copy.

        Args:
            image_path: Path to image file (first page)
            language: Language for prompt ("fr" or "en")

        Returns:
            Dict with name, confidence, reasoning
        """
        prompt = self._build_name_detection_prompt(language)
        response = self.call_vision(prompt, image_path=image_path)
        return self._parse_name_response(response, language)

    def _build_name_detection_prompt(self, language: str) -> str:
        """Build prompt for name detection."""
        if language == "fr":
            return """Analyse cette première page de copie et identifie le NOM et PRÉNOM de l'élève.

Réponds UNIQUEMENT dans ce format:
NOM: [Nom Prénom détecté ou "Inconnu"]
CONFIANCE: [0.0 à 1.0]
RAISONNEMENT: [Brève explication de où tu as trouvé le nom]

Exemples:
NOM: Dupont Marie
CONFIANCE: 0.95
RAISONNEMENT: Nom clairement visible en haut à droite de la copie

NOM: Inconnu
CONFIANCE: 0.2
RAISONNEMENT: Aucun nom visible sur cette page"""
        else:
            return """Analyze this first page and identify the student's FIRST and LAST NAME.

Respond ONLY in this format:
NAME: [Detected First Last or "Unknown"]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Brief explanation of where you found the name]

Examples:
NAME: Smith John
CONFIDENCE: 0.95
REASONING: Name clearly visible at top right of the page

NAME: Unknown
CONFIDENCE: 0.2
REASONING: No name visible on this page"""

    def _parse_name_response(self, response: str, language: str = "fr") -> Dict[str, Any]:
        """Parse name detection response."""
        result = {
            "name": None,
            "confidence": 0.5,
            "reasoning": ""
        }

        name_key = "NOM:" if language == "fr" else "NAME:"
        conf_key = "CONFIANCE:" if language == "fr" else "CONFIDENCE:"
        reason_key = "RAISONNEMENT:" if language == "fr" else "REASONING:"

        for line in response.strip().split('\n'):
            line = line.strip()

            if line.upper().startswith(name_key):
                name = line.split(':', 1)[1].strip()
                if name.lower() not in ['inconnu', 'unknown', 'none', 'n/a', '-']:
                    result["name"] = name

            elif line.upper().startswith(conf_key):
                try:
                    result["confidence"] = float(line.split(':', 1)[1].strip())
                except ValueError:
                    pass

            elif line.upper().startswith(reason_key):
                result["reasoning"] = line.split(':', 1)[1].strip()

        return result

    # ==================== RULE EXTRACTION ====================

    EXTRACTION_SYSTEM_PROMPT = """You are a rule extraction specialist.

Your role is to extract generalizable grading principles from specific teacher decisions.

Guidelines:
- Extract the underlying principle, not just the specific case
- The rule should be actionable and clear
- Avoid overfitting to specific examples
- Consider edge cases

Output format should be:
IF [condition on student answer]
THEN [grading action]
BECAUSE [reasoning]
"""

    def extract_rule(
        self,
        teacher_decision: str,
        question_context: str,
        original_grade: float,
        new_grade: float,
        student_answer: str = ""
    ) -> str:
        """
        Extract a generalizable rule from a teacher's decision.

        Args:
            teacher_decision: What the teacher said
            question_context: Context about the question
            original_grade: Grade before adjustment
            new_grade: Grade after adjustment
            student_answer: The student's answer

        Returns:
            Extracted rule text
        """
        from config.prompts import build_rule_extraction_prompt

        prompt = build_rule_extraction_prompt(
            teacher_decision=teacher_decision,
            question_context=question_context,
            original_grade=original_grade,
            new_grade=new_grade,
            student_answer=student_answer
        )

        response = self.call_text(prompt, system_prompt=self.EXTRACTION_SYSTEM_PROMPT)
        return response.strip()

    # ==================== CROSS-COPY ANALYSIS ====================

    def analyze_cross_copy(
        self,
        question_id: str,
        question_text: str,
        answer_summaries: List[tuple],
        max_points: float
    ) -> Dict[str, Any]:
        """
        Analyze answers across all copies.

        Args:
            question_id: Question identifier
            question_text: The question text
            answer_summaries: List of (copy_id, answer_summary) tuples
            max_points: Maximum points

        Returns:
            Analysis with patterns, errors, unique approaches
        """
        from config.prompts import build_cross_copy_analysis_prompt

        prompt = build_cross_copy_analysis_prompt(
            question_id=question_id,
            question_text=question_text,
            answer_summaries=answer_summaries,
            max_points=max_points
        )

        response = self.call_text(prompt)
        return self._parse_analysis_response(response)

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse cross-copy analysis response."""
        result = {
            "common_correct": [],
            "common_errors": [],
            "unique_approaches": [],
            "difficulty_estimate": 0.5
        }

        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()

            if line.startswith("COMMON_CORRECT:"):
                items = line.split("COMMON_CORRECT:")[1].strip()
                if items and items.lower() != "none":
                    result["common_correct"] = [
                        i.strip() for i in items.split(",") if i.strip()
                    ]

            elif line.startswith("COMMON_ERRORS:"):
                items = line.split("COMMON_ERRORS:")[1].strip()
                if items and items.lower() != "none":
                    result["common_errors"] = [
                        i.strip() for i in items.split(",") if i.strip()
                    ]

            elif line.startswith("UNIQUE_APPROACHES:"):
                items = line.split("UNIQUE_APPROACHES:")[1].strip()
                if items and items.lower() != "none":
                    result["unique_approaches"] = [
                        i.strip() for i in items.split(",") if i.strip()
                    ]

            elif line.startswith("DIFFICULTY_ESTIMATE:"):
                try:
                    result["difficulty_estimate"] = float(
                        line.split("DIFFICULTY_ESTIMATE:")[1].strip()
                    )
                except (ValueError, IndexError):
                    pass

        return result

    # ==================== FEEDBACK GENERATION ====================

    def generate_feedback(
        self,
        student_name: str,
        graded_copy: Dict[str, Any],
        class_performance: Dict[str, float]
    ) -> str:
        """
        Generate personalized feedback for a student.

        Args:
            student_name: Student's name
            graded_copy: Grading results
            class_performance: Class performance stats

        Returns:
            Feedback text
        """
        from config.prompts import build_feedback_prompt

        prompt = build_feedback_prompt(
            student_name=student_name,
            graded_copy=graded_copy,
            class_performance=class_performance
        )

        return self.call_text(prompt).strip()
