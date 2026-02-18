"""
Base provider class for AI API interactions.

Provides shared functionality for vision and text interactions,
including image processing, token tracking, and response parsing.
"""

import base64
import functools
import logging
import re
import time
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Dict, Any, List, Optional

from PIL import Image

from core.models import AICallResult
from core.exceptions import (
    ProviderError,
    APIConnectionError,
    APITimeoutError,
    APIResponseError,
    ParsingError,
)


def _sanitize_for_logging(text: str) -> str:
    """
    Sanitize text for logging by removing potential API keys and secrets.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text with sensitive values masked
    """
    if not text:
        return text

    # Patterns for common API key formats
    patterns = [
        # Generic API keys (sk-*, api_key=, etc.)
        (r'(sk-[a-zA-Z0-9]{20,})', r'sk-[REDACTED]'),
        (r'(api[_-]?key\s*[=:]\s*["\']?)([a-zA-Z0-9_-]{20,})', r'\1[REDACTED]'),
        # Bearer tokens
        (r'(Bearer\s+)([a-zA-Z0-9_-]{20,})', r'\1[REDACTED]'),
        # JWT-like tokens
        (r'(eyJ[a-zA-Z0-9_-]*\.)([a-zA-Z0-9_-]+)(\.[a-zA-Z0-9_-]+)', r'\1[REDACTED]\3'),
        # Google API keys (AIza...)
        (r'(AIza[a-zA-Z0-9_-]{35})', r'AIza[REDACTED]'),
        # OpenAI API keys
        (r'(sk-[a-zA-Z0-9]{48})', r'sk-[REDACTED]'),
        # Generic long alphanumeric strings that look like keys
        (r'(["\']?[a-zA-Z0-9_-]{32,}["\']?)', lambda m: m.group(0) if len(m.group(0)) < 40 else '[REDACTED]'),
    ]

    sanitized = text
    for pattern, replacement in patterns:
        if callable(replacement):
            sanitized = re.sub(pattern, replacement, sanitized)
        else:
            sanitized = re.sub(pattern, replacement, sanitized)

    return sanitized


@functools.lru_cache(maxsize=128)
def _cached_image_to_base64(image_path: str) -> str:
    """
    Convert an image file to base64 string with caching.

    Uses LRU cache to avoid re-reading the same file multiple times.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class APIErrorContext:
    """
    Context manager for consistent API error handling.

    Wraps API calls with proper error translation and logging.

    Usage:
        with APIErrorContext("vision call"):
            response = client.call(...)
    """

    def __init__(self, operation: str, provider_name: str = "unknown"):
        """
        Initialize error context.

        Args:
            operation: Description of the operation being performed
            provider_name: Name of the provider (for error messages)
        """
        self.operation = operation
        self.provider_name = provider_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False

        # Log the error
        logging.error(f"{self.provider_name} API error during {self.operation}: {exc_val}")

        # Translate common exceptions to custom exceptions
        exc_name = exc_type.__name__ if exc_type else "Unknown"

        # Connection errors
        if any(name in exc_name for name in ['Connection', 'Connect', 'Network']):
            raise APIConnectionError(
                f"Failed to connect during {self.operation}: {exc_val}"
            ) from exc_val

        # Timeout errors
        if any(name in exc_name for name in ['Timeout', 'TimedOut']):
            raise APITimeoutError(
                f"Timeout during {self.operation}: {exc_val}"
            ) from exc_val

        # Rate limiting
        if 'Rate' in exc_name or '429' in str(exc_val):
            raise APIResponseError(
                f"Rate limited during {self.operation}: {exc_val}"
            ) from exc_val

        # JSON/parsing errors
        if any(name in exc_name for name in ['JSON', 'Parse', 'Decode']):
            raise ParsingError(
                f"Failed to parse response during {self.operation}: {exc_val}"
            ) from exc_val

        # Generic API error
        raise ProviderError(
            f"API error during {self.operation}: {exc_val}"
        ) from exc_val


def handle_api_errors(operation: str):
    """
    Decorator for consistent API error handling.

    Usage:
        @handle_api_errors("vision call")
        def call_vision(self, ...):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            provider_name = getattr(self, '__class__.__name__', 'unknown')
            with APIErrorContext(operation, provider_name):
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


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
        """Convert an image file to base64 string (cached)."""
        return _cached_image_to_base64(image_path)

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
        """Log an AI call to audit trail with sanitized output."""
        # Sanitize summaries to remove potential API keys
        safe_input = _sanitize_for_logging(input_summary)
        safe_response = _sanitize_for_logging(response_summary)

        self.call_history.append(AICallResult(
            prompt_type=prompt_type,
            input_summary=safe_input,
            response_summary=safe_response,
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
