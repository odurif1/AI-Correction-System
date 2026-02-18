"""
Google Gemini API provider for vision and text interactions.

Handles communication with Google's Gemini models.
"""

import time
import base64
from typing import Optional, List, Dict, Any

from PIL import Image
import google.genai as genai
from google.api_core import exceptions as google_exceptions
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_exponential

from ai.base_provider import BaseProvider
from config.settings import get_settings
from config.constants import MAX_RETRIES


# Define retryable exceptions for Google API
RETRYABLE_EXCEPTIONS = (
    google_exceptions.TooManyRequests,  # 429 rate limit
    google_exceptions.ServiceUnavailable,  # 503
    google_exceptions.DeadlineExceeded,  # timeout
    google_exceptions.InternalServerError,  # 500
    google_exceptions.GatewayTimeout,  # 504
    ConnectionError,
    TimeoutError,
)


class GeminiError(Exception):
    """Base exception for Gemini API errors."""
    pass


class GeminiProvider(BaseProvider):
    """
    Provider for Google Gemini API interactions.

    Inherits from BaseProvider for shared functionality.
    Implements Gemini-specific API calls.
    """

    # Available Gemini models
    MODEL_PRO = "models/gemini-3-pro-preview"
    MODEL_FLASH = "models/gemini-2.5-flash"
    MODEL_EMBEDDING = "models/gemini-embedding-001"
    DEFAULT_VISION_MODEL = MODEL_FLASH
    DEFAULT_TEXT_MODEL = MODEL_FLASH

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        vision_model: str = None,
        embedding_model: str = None,
        mock_mode: bool = False
    ):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google API key (default: from settings)
            model: Model for text operations
            vision_model: Model for vision operations
            embedding_model: Model for embeddings
            mock_mode: If True, skip API key check for testing
        """
        super().__init__(mock_mode=mock_mode)

        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key

        if not self.api_key and not mock_mode:
            raise ValueError(
                "Gemini API key is required. "
                "Set AI_CORRECTION_GEMINI_API_KEY."
            )

        if not mock_mode:
            self.client = genai.Client(api_key=self.api_key)

        self.model = model or settings.gemini_model or self.DEFAULT_TEXT_MODEL
        self.vision_model = vision_model or settings.gemini_vision_model or self.DEFAULT_VISION_MODEL
        self.embedding_model = embedding_model or settings.gemini_embedding_model or self.MODEL_EMBEDDING

    def _prepare_image(
        self,
        image_path=None,
        image_bytes: bytes = None,
        pil_image: Image.Image = None
    ) -> Optional[Dict[str, Any]]:
        """Prepare image for Gemini API."""
        if image_path:
            image = Image.open(image_path)
            return {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": self._image_to_base64_from_pil(image)
                }
            }
        elif image_bytes:
            return {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(image_bytes).decode("utf-8")
                }
            }
        elif pil_image:
            return {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": self._image_to_base64_from_pil(pil_image)
                }
            }
        return None

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS)
    )
    def call_vision(
        self,
        prompt: str,
        image_path=None,
        image_bytes: bytes = None,
        pil_image: Image.Image = None,
        response_format: str = "text"
    ) -> str:
        """
        Call the vision API with one or multiple images.

        Args:
            prompt: Text prompt to send with the image
            image_path: Path to image file OR list of paths for multiple images
            image_bytes: Raw image bytes
            pil_image: PIL Image object
            response_format: "text" or "json"

        Returns:
            Model response text
        """
        start_time = time.time()

        if self.mock_mode:
            return "Mock vision response"

        content = [prompt]

        # Handle multiple images (list of paths)
        if isinstance(image_path, list):
            for i, path in enumerate(image_path):
                image_data = self._prepare_image(image_path=path)
                if image_data:
                    content.append(f"\n--- PAGE {i + 1} ---")
                    content.append(image_data)
        else:
            image_data = self._prepare_image(
                image_path=image_path,
                image_bytes=image_bytes,
                pil_image=pil_image
            )
            if image_data:
                content.append(image_data)

        response = self.client.models.generate_content(
            model=self.vision_model,
            contents=content
        )

        result = response.text or ""

        # Extract token usage
        prompt_tokens = None
        completion_tokens = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)
            completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)

        # Log call
        duration = (time.time() - start_time) * 1000
        num_images = len(image_path) if isinstance(image_path, list) else (1 if image_path else 0)
        self._log_call(
            prompt_type="vision",
            input_summary=f"Vision prompt ({num_images} images): {prompt[:100]}...",
            response_summary=result[:200],
            duration_ms=duration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )

        return result

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS)
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

        if self.mock_mode:
            return "Mock text response"

        if system_prompt:
            content = [{"role": "user", "parts": [{"text": system_prompt + "\n\n" + prompt}]}]
        else:
            content = [prompt]

        response = self.client.models.generate_content(
            model=self.model,
            contents=content
        )

        result = response.text or ""

        # Extract token usage
        prompt_tokens = None
        completion_tokens = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)
            completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)

        # Log call
        duration = (time.time() - start_time) * 1000
        self._log_call(
            prompt_type="text",
            input_summary=f"Text prompt: {prompt[:100]}...",
            response_summary=result[:200],
            duration_ms=duration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )

        return result

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        if self.mock_mode:
            import hashlib
            hash_obj = hashlib.sha256(text.encode())
            hash_bytes = hash_obj.digest()
            return [(hash_bytes[i % len(hash_bytes)] - 128) / 128.0 for i in range(768)]

        result = self.client.models.embed_content(
            model=self.embedding_model,
            contents=text
        )
        return result.embeddings[0].values

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts using batch API.

        Uses the Gemini batch embedding endpoint for efficiency.
        Falls back to individual calls if batch fails.
        """
        if self.mock_mode:
            return [self.get_embedding(text) for text in texts]

        try:
            # Use batch API for multiple texts (up to 100 per request)
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=texts
            )
            return [embedding.values for embedding in result.embeddings]
        except Exception as e:
            # Fall back to individual calls if batch fails
            import logging
            logging.warning(f"Batch embedding failed, falling back to individual calls: {e}")
            return [self.get_embedding(text) for text in texts]

    # Override grade_with_vision to add multi-page support
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

        Extended to support multi-page PDFs.
        """
        from config.prompts import build_vision_grading_prompt

        # Build context about pages if multiple
        num_pages = len(image_path) if isinstance(image_path, list) else 1
        page_context = ""
        if num_pages > 1:
            page_context = f"\n\nIMPORTANT: Tu as accès à {num_pages} pages de la copie de cet élève. Cherche la question {question_text} sur TOUTES les pages, pas seulement la première."

        prompt = build_vision_grading_prompt(
            question_text=question_text,
            criteria=criteria,
            max_points=max_points,
            class_context=class_context,
            language=language
        ) + page_context

        response = self.call_vision(prompt, image_path=image_path)
        return self._parse_grading_response(response)
