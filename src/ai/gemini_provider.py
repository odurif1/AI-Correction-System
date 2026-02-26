"""
Google Gemini API provider for vision and text interactions.

Handles communication with Google's Gemini models.
"""

import time
import base64
import logging
from typing import Optional, List, Dict, Any

from PIL import Image
import google.genai as genai
from google.api_core import exceptions as google_exceptions
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_exponential

from ai.base_provider import BaseProvider
from config.settings import get_settings
from config.constants import MAX_RETRIES

logger = logging.getLogger(__name__)

# Import types for context caching (optional)
try:
    from google.genai import types as genai_types
    HAS_GENAI_TYPES = True
except ImportError:
    HAS_GENAI_TYPES = False


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
            model: Model for text operations (required in .env)
            vision_model: Model for vision operations (required in .env)
            embedding_model: Model for embeddings (required in .env)
            mock_mode: If True, skip API key check for testing
        """
        super().__init__(mock_mode=mock_mode)

        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key

        if not self.api_key and not mock_mode:
            raise ValueError(
                "Gemini API key is required. "
                "Set AI_CORRECTION_GEMINI_API_KEY in .env"
            )

        if not mock_mode:
            self.client = genai.Client(api_key=self.api_key)

        self.model = model or settings.gemini_model
        # Vision model falls back to text model if not specified
        self.vision_model = vision_model or settings.gemini_vision_model or self.model
        self.embedding_model = embedding_model or settings.gemini_embedding_model

        if not self.model:
            raise ValueError(
                "Gemini model is required. "
                "Set AI_CORRECTION_GEMINI_MODEL in .env"
            )

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

        # Prepare images list for debug capture
        images_list = None
        if image_path:
            images_list = image_path if isinstance(image_path, list) else [image_path]

        self._log_call(
            prompt_type="vision",
            input_summary=f"Vision prompt ({num_images} images): {prompt[:100]}...",
            response_summary=result[:200],
            duration_ms=duration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            full_prompt=prompt,
            full_response=result,
            images=images_list,
            model_name=self.vision_model
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
            completion_tokens=completion_tokens,
            full_prompt=prompt,
            full_response=result,
            model_name=self.model
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
        from prompts import build_vision_grading_prompt

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

    # ==================== GENERATE WITH CACHE METHOD ====================

    def generate_with_cache(
        self,
        cache_id: str,
        prompt: str,
        images: List[str] = None
    ) -> str:
        """
        Generate content using cached context.

        Unlike Chat API, this does NOT maintain conversation history.
        Each call is independent but references the same cached content.

        This is more token-efficient than Chat API because:
        - Chat API re-sends the entire conversation history on each call
        - generate_with_cache only sends the new prompt + cache reference

        Args:
            cache_id: Cache ID from create_cached_context()
            prompt: User prompt for this call
            images: Optional list of image paths (additional, not cached)

        Returns:
            Model response text
        """
        start_time = time.time()

        if self.mock_mode:
            return "Mock cached response"

        from google.genai import types

        # Build parts for the new content
        parts = [types.Part(text=prompt)]

        if images:
            for img_path in images:
                image_data = self._prepare_image(image_path=img_path)
                if image_data and 'inline_data' in image_data:
                    parts.append(types.Part(
                        inline_data=types.Blob(
                            mime_type=image_data['inline_data']['mime_type'],
                            data=image_data['inline_data']['data']
                        )
                    ))

        # Use generate_content with cached_content reference
        config = types.GenerateContentConfig(
            cached_content=cache_id
        )

        response = self.client.models.generate_content(
            model=self.vision_model,
            contents=[types.Content(role="user", parts=parts)],
            config=config
        )

        result = response.text or ""

        # Extract token usage including cached tokens
        prompt_tokens = None
        completion_tokens = None
        cached_tokens = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)
            completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)
            cached_tokens = getattr(response.usage_metadata, 'cached_content_token_count', None)

        # Log call
        duration = (time.time() - start_time) * 1000
        num_images = len(images) if isinstance(images, list) else (1 if images else 0)
        self._log_call(
            prompt_type="cached_generation",
            input_summary=f"Cached generation ({num_images} new images): {prompt[:100]}...",
            response_summary=result[:200],
            duration_ms=duration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            full_prompt=prompt,
            full_response=result,
            images=images,
            model_name=self.vision_model
        )

        if cached_tokens:
            logger.info(
                f"Context caching: Used {cached_tokens} cached tokens "
                f"(~${self._estimate_cache_savings(cached_tokens):.4f} saved)"
            )

        return result

    # ==================== CONTEXT CACHING (OPTIONAL) ====================

    # Gemini minimum tokens for context caching
    MIN_CACHE_TOKENS = 2048
    # Approximate tokens per image (varies by size, this is conservative)
    TOKENS_PER_IMAGE = 258

    def supports_context_caching(self) -> bool:
        """
        Check if context caching is available for this provider.

        Returns:
            True if context caching is supported and available
        """
        if self.mock_mode:
            return True  # Mock mode supports it for testing
        return HAS_GENAI_TYPES

    def _estimate_tokens(self, text: str, num_images: int) -> int:
        """
        Estimate token count for text + images.

        Args:
            text: Text content
            num_images: Number of images

        Returns:
            Estimated token count
        """
        # ~4 chars per token for most languages
        text_tokens = len(text) // 4 if text else 0
        image_tokens = num_images * self.TOKENS_PER_IMAGE
        return text_tokens + image_tokens

    def create_cached_context(
        self,
        system_prompt: str,
        images: List[str] = None,
        ttl_seconds: int = 3600
    ) -> Optional[str]:
        """
        Create a cached context for reuse across multiple calls.

        This is useful for reducing costs when the same context
        (system prompt + images) is used repeatedly.

        Note: Gemini requires minimum 2048 tokens for caching.
        If content is too small, returns None (no error).

        Args:
            system_prompt: System instruction
            images: Optional list of image paths to cache
            ttl_seconds: Time-to-live in seconds (default: 1 hour)

        Returns:
            Cache name/ID if successful, None otherwise
        """
        

        if self.mock_mode:
            logger.info("Context caching: Mock mode - returning mock cache ID")
            return "mock_cache_id"

        if not HAS_GENAI_TYPES:
            logger.warning(
                "Context caching NOT AVAILABLE: google.genai.types not found. "
                "Install the latest google-genai package for caching support. "
                "Falling back to regular API calls (higher cost)."
            )
            return None

        # Estimate token count before attempting to cache
        num_images = len(images) if images else 0
        estimated_tokens = self._estimate_tokens(system_prompt or "", num_images)
        

        if estimated_tokens < self.MIN_CACHE_TOKENS:
            logger.info(
                f"Context caching SKIPPED: Estimated {estimated_tokens} tokens "
                f"(minimum {self.MIN_CACHE_TOKENS} required). "
                f"Using regular API calls."
            )
            return None

        try:
            parts = []
            

            # Add system prompt only if provided
            if system_prompt:
                parts.append(genai_types.Part(text=system_prompt))

            # Add images
            if images:
                for img_path in images:
                    image_data = self._prepare_image(image_path=img_path)
                    if image_data and 'inline_data' in image_data:
                        parts.append(genai_types.Part(
                            inline_data=genai_types.Blob(
                                mime_type=image_data['inline_data']['mime_type'],
                                data=image_data['inline_data']['data']
                            )
                        ))

            cached_content = self.client.caches.create(
                model=self.vision_model,
                config=genai_types.CreateCachedContentConfig(
                    contents=[
                        genai_types.Content(
                            role="user",
                            parts=parts
                        )
                    ],
                    ttl=f"{ttl_seconds}s"
                )
            )
            

            logger.info(
                f"Context caching: Created cache '{cached_content.name}' "
                f"(~{estimated_tokens} tokens, {len(parts)} parts, TTL {ttl_seconds}s)"
            )

            return cached_content.name

        except Exception as e:
            logger.warning(
                f"Context caching FAILED: Could not create cached context: {e}. "
                "Falling back to regular API calls (higher cost)."
            )
            return None

    def delete_cached_context(self, cache_id: str) -> bool:
        """
        Delete a cached context from Gemini's server.

        Args:
            cache_id: The cache ID to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        if self.mock_mode:
            logger.info(f"Context caching: Mock mode - pretending to delete cache {cache_id}")
            return True

        if not cache_id:
            return False

        try:
            self.client.caches.delete(name=cache_id)
            logger.info(f"Context caching: Deleted cache '{cache_id}'")
            return True
        except Exception as e:
            logger.warning(f"Context caching: Failed to delete cache '{cache_id}': {e}")
            return False

    def use_cached_context(
        self,
        cache_name: str,
        new_prompt: str,
        images: List[str] = None
    ) -> str:
        """
        Use a cached context with a new prompt.

        Args:
            cache_name: Cache ID from create_cached_context()
            new_prompt: New user prompt
            images: Optional additional images (not cached)

        Returns:
            Model response text
        """
        if self.mock_mode:
            return "Mock cached response"

        start_time = time.time()

        content = [new_prompt]
        if images:
            for img_path in images:
                image_data = self._prepare_image(image_path=img_path)
                if image_data:
                    content.append(image_data)

        try:
            from google.genai import types
            response = self.client.models.generate_content(
                model=self.vision_model,
                contents=content,
                config=types.GenerateContentConfig(
                    cached_content=cache_name
                )
            )

            result = response.text or ""

            # Extract token usage - note: cached tokens may be reported differently
            prompt_tokens = None
            completion_tokens = None
            cached_tokens = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)
                completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)
                # Gemini may report cached tokens separately
                cached_tokens = getattr(response.usage_metadata, 'cached_content_token_count', None)

            # Log call with cached token info
            duration = (time.time() - start_time) * 1000
            self._log_call(
                prompt_type="cached_vision",
                input_summary=f"Cached context: {new_prompt[:100]}...",
                response_summary=result[:200],
                duration_ms=duration,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                full_prompt=new_prompt,
                full_response=result,
                images=images,
                model_name=self.vision_model
            )

            if cached_tokens:
                logger.info(
                    f"Context caching: Used {cached_tokens} cached tokens "
                    f"(~${self._estimate_cache_savings(cached_tokens):.4f} saved)"
                )

            return result

        except Exception as e:
            logger.warning(
                f"Context caching FAILED: Could not use cached context: {e}. "
                "Falling back to regular call (higher cost)."
            )
            # Fallback to regular vision call
            return self.call_vision(new_prompt, image_path=images)

    def _estimate_cache_savings(self, cached_tokens: int) -> float:
        """
        Estimate cost savings from using cached tokens.

        Args:
            cached_tokens: Number of tokens served from cache

        Returns:
            Estimated savings in USD
        """
        from config.constants import (
            GEMINI_25_FLASH_PRICING,
            GEMINI_3_PRO_PRICING,
            LONG_CONTEXT_THRESHOLD
        )

        model_lower = self.vision_model.lower()

        # Determine which model and pricing tier we're using
        if "flash" in model_lower:
            regular_price = GEMINI_25_FLASH_PRICING["input"]
            cached_price = GEMINI_25_FLASH_PRICING["cached"]
        elif "pro" in model_lower:
            # Use short context pricing as default for cache savings estimation
            regular_price = GEMINI_3_PRO_PRICING["input_short"]
            cached_price = GEMINI_3_PRO_PRICING["cached_short"]
        else:
            # Default to Flash pricing
            regular_price = GEMINI_25_FLASH_PRICING["input"]
            cached_price = GEMINI_25_FLASH_PRICING["cached"]

        # Calculate savings per 1M tokens
        savings_per_million = regular_price - cached_price
        return (cached_tokens / 1_000_000) * savings_per_million

    def get_effective_token_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0
    ) -> Dict[str, float]:
        """
        Calculate effective token costs for Gemini, accounting for cached tokens.

        Args:
            prompt_tokens: Total prompt/input tokens
            completion_tokens: Number of completion/output tokens
            cached_tokens: Number of tokens served from cache (lower cost)

        Returns:
            Dict with cost breakdown in USD
        """
        from config.constants import (
            GEMINI_25_FLASH_PRICING,
            GEMINI_3_PRO_PRICING,
            LONG_CONTEXT_THRESHOLD
        )

        model_lower = self.vision_model.lower()

        # Determine which model and pricing tier we're using
        if "flash" in model_lower:
            # Flash models have simple pricing
            input_price = GEMINI_25_FLASH_PRICING["input"]
            output_price = GEMINI_25_FLASH_PRICING["output"]
            cached_price = GEMINI_25_FLASH_PRICING["cached"]
        elif "pro" in model_lower:
            # Pro models have tiered pricing based on context size
            is_long_context = prompt_tokens > LONG_CONTEXT_THRESHOLD
            if is_long_context:
                input_price = GEMINI_3_PRO_PRICING["input_long"]
                output_price = GEMINI_3_PRO_PRICING["output_long"]
                cached_price = GEMINI_3_PRO_PRICING["cached_long"]
            else:
                input_price = GEMINI_3_PRO_PRICING["input_short"]
                output_price = GEMINI_3_PRO_PRICING["output_short"]
                cached_price = GEMINI_3_PRO_PRICING["cached_short"]
        else:
            # Default to Flash pricing
            input_price = GEMINI_25_FLASH_PRICING["input"]
            output_price = GEMINI_25_FLASH_PRICING["output"]
            cached_price = GEMINI_25_FLASH_PRICING["cached"]

        # Calculate costs
        # Non-cached portion of prompt tokens
        non_cached_tokens = max(0, prompt_tokens - cached_tokens)
        non_cached_cost = (non_cached_tokens / 1_000_000) * input_price
        cached_cost = (cached_tokens / 1_000_000) * cached_price
        output_cost = (completion_tokens / 1_000_000) * output_price

        # Calculate what it would have cost without caching
        full_input_cost = (prompt_tokens / 1_000_000) * input_price
        cached_savings = full_input_cost - (non_cached_cost + cached_cost)

        return {
            "input_cost": non_cached_cost + cached_cost,
            "output_cost": output_cost,
            "total_cost": non_cached_cost + cached_cost + output_cost,
            "cached_tokens": cached_tokens,
            "cached_savings": cached_savings
        }

    def get_estimated_cost(self) -> Dict[str, Any]:
        """
        Get estimated cost based on accumulated token usage.

        Returns:
            Dict with cost breakdown in USD
        """
        usage = self.get_token_usage()
        cost = self.get_effective_token_cost(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            cached_tokens=usage["cached_tokens"]
        )
        return {
            "provider": "Gemini",
            "model": self.vision_model,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "cached_tokens": usage["cached_tokens"],
            "total_tokens": usage["total_tokens"],
            "calls": usage["calls"],
            "estimated_cost_usd": round(cost["total_cost"], 4),
            "input_cost_usd": round(cost["input_cost"], 4),
            "output_cost_usd": round(cost["output_cost"], 4),
            "cached_savings_usd": round(cost["cached_savings"], 4)
        }
