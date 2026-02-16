"""
OpenAI API provider for vision and text interactions.

Handles communication with OpenAI's GPT-4o and embedding models.
"""

import time
import base64
from typing import Optional, List, Dict, Any, BinaryIO
from io import BytesIO
from pathlib import Path

from openai import OpenAI, OpenAIError
from tenacity import retry, stop_after_attempt, retry_if_exception_type
from PIL import Image

from core.models import AICallResult
from config.settings import get_settings
from config.constants import (
    MAX_RETRIES, MAX_TOKENS, TEMPERATURE, EMBEDDING_MODEL
)


class OpenAIProvider:
    """
    Provider for OpenAI API interactions.

    Supports:
    - Vision (GPT-4o) for grading handwritten work
    - Text for rule extraction and analysis
    - Embeddings for clustering
    """

    def __init__(self, api_key: str = None, base_url: str = None, mock_mode: bool = False):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key (default: from settings)
            base_url: Custom base URL (default: from settings)
            mock_mode: If True, skip API key check for testing
        """
        settings = get_settings()

        self.api_key = api_key or settings.openai_api_key
        self.mock_mode = mock_mode

        if not self.api_key and not mock_mode:
            raise ValueError("OpenAI API key is required. Set AI_CORRECTION_OPENAI_API_KEY.")

        if not mock_mode:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=base_url or settings.openai_organization
            )

        self.model = settings.model
        self.vision_model = settings.vision_model
        self.embedding_model = settings.embedding_model

        # Audit trail
        self.call_history: List[AICallResult] = []

    def _image_to_base64(self, image_path: str) -> str:
        """Convert an image file to base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _image_to_base64_from_pil(self, image: Image.Image) -> str:
        """Convert a PIL Image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        retry=retry_if_exception_type(OpenAIError)
    )
    def call_vision(
        self,
        prompt: str,
        image_path: str = None,
        image_bytes: bytes = None,
        pil_image: Image.Image = None,
        response_format: str = "text"
    ) -> str:
        """
        Call the vision API with an image.

        Args:
            prompt: Text prompt to send with the image
            image_path: Path to image file
            image_bytes: Raw image bytes
            pil_image: PIL Image object
            response_format: "text" or "json"

        Returns:
            Model response text
        """
        start_time = time.time()

        # Prepare image content
        image_content = None

        if image_path:
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self._image_to_base64(image_path)}",
                    "detail": "high"
                }
            }
        elif image_bytes:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high"
                }
            }
        elif pil_image:
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self._image_to_base64_from_pil(pil_image)}",
                    "detail": "high"
                }
            }

        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]

        if image_content:
            messages[0]["content"].append(image_content)

        response = self.client.chat.completions.create(
            model=self.vision_model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )

        result = response.choices[0].message.content or ""

        # Log call
        duration = (time.time() - start_time) * 1000
        self.call_history.append(AICallResult(
            prompt_type="vision",
            input_summary=f"Vision prompt: {prompt[:100]}...",
            response_summary=result[:200],
            duration_ms=duration,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None
        ))

        return result

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        retry=retry_if_exception_type(OpenAIError)
    )
    def call_text(
        self,
        prompt: str,
        system_prompt: str = None,
        response_format: str = "text"
    ) -> str:
        """
        Call the text API.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            response_format: "text" or "json"

        Returns:
            Model response text
        """
        start_time = time.time()

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE
        }

        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)

        result = response.choices[0].message.content or ""

        # Log call
        duration = (time.time() - start_time) * 1000
        self.call_history.append(AICallResult(
            prompt_type="text",
            input_summary=f"Text prompt: {prompt[:100]}...",
            response_summary=result[:200],
            duration_ms=duration,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None
        ))

        return result

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        retry=retry_if_exception_type(OpenAIError)
    )
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )

        return response.data[0].embedding

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        retry=retry_if_exception_type(OpenAIError)
    )
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )

        return [item.embedding for item in response.data]

    def grade_with_vision(
        self,
        question_text: str,
        criteria: str,
        image_path: str,
        max_points: float,
        class_context: str = "",
        language: str = "en",
        question_id: str = "",
        reading_disagreement_callback: callable = None,
        skip_reading_consensus: bool = True
    ) -> Dict[str, Any]:
        """
        Grade a student's answer using vision.

        Args:
            question_text: The question
            criteria: Grading criteria
            image_path: Path to image of student's answer
            max_points: Maximum points
            class_context: Context from other students' answers
            language: Language for response (fr/en)
            question_id: Question identifier (ignored in single LLM mode)
            reading_disagreement_callback: Callback for reading disagreements (ignored in single LLM mode)
            skip_reading_consensus: Whether to skip reading consensus (ignored in single LLM mode)

        Returns:
            Dictionary with grade, confidence, internal_reasoning, student_feedback
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
            Dict with:
            - name: detected name or None
            - confidence: 0.0 to 1.0
            - reasoning: explanation of detection
        """
        if language == "fr":
            prompt = """Analyse cette première page de copie et identifie le NOM et PRÉNOM de l'élève.

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
            prompt = """Analyze this first page and identify the student's FIRST and LAST NAME.

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

        response = self.call_vision(prompt, image_path=image_path)
        return self._parse_name_response(response, language)

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

        response = self.call_text(prompt, system_prompt=EXTRACTION_SYSTEM_PROMPT)

        return response.strip()

    def analyze_cross_copy(
        self,
        question_id: str,
        question_text: str,
        answer_summaries: List[tuple[str, str]],
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


# System prompt for rule extraction
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
