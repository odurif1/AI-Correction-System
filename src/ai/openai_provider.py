"""
OpenAI-compatible API provider.

Works with any OpenAI-compatible API (OpenAI, GLM, OpenRouter, etc.)
"""

import base64
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import httpx

from openai import OpenAI, OpenAIError
from tenacity import retry, stop_after_attempt, retry_if_exception_type
from PIL import Image
import logging

from ai.base_provider import BaseProvider
from config.constants import MAX_RETRIES, MAX_TOKENS, TEMPERATURE, API_CONNECT_TIMEOUT, API_READ_TIMEOUT

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, bytes, Image.Image, List[Union[str, Path, bytes, Image.Image]]]


class OpenAIProvider(BaseProvider):
    """
    Provider for OpenAI-compatible APIs.

    Configuration is handled by the factory using config/providers.py registry.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = None,
        model: str = None,
        vision_model: str = None,
        embedding_model: str = None,
        name: str = None,
        mock_mode: bool = False,
        extra_headers: Dict[str, str] = None,
        **kwargs
    ):
        super().__init__(mock_mode=mock_mode)

        self.api_key = api_key
        self.model = model
        self.vision_model = vision_model or model
        self.embedding_model = embedding_model
        self._name = name or model or "OpenAI"
        self.base_url = base_url

        if not mock_mode:
            self.client = self._create_client(extra_headers, kwargs)

    def _create_client(self, extra_headers: Dict[str, str], kwargs: Dict) -> OpenAI:
        """Create OpenAI client."""
        timeout = httpx.Timeout(
            connect=API_CONNECT_TIMEOUT,
            read=API_READ_TIMEOUT,
            write=API_CONNECT_TIMEOUT,
            pool=API_CONNECT_TIMEOUT
        )

        client_kwargs = {
            "api_key": self.api_key,
            "timeout": timeout
        }

        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        if extra_headers:
            client_kwargs["default_headers"] = extra_headers

        client_kwargs.update(kwargs)
        return OpenAI(**client_kwargs)

    @property
    def name(self) -> str:
        return self._name

    # ==================== IMAGE HANDLING ====================

    def _prepare_image_content(self, image: Union[str, Path, bytes, Image.Image]) -> Dict[str, Any]:
        """Convert image to API format."""
        if isinstance(image, (str, Path)):
            b64 = self._image_to_base64(str(image))
        elif isinstance(image, bytes):
            b64 = base64.b64encode(image).decode("utf-8")
        elif isinstance(image, Image.Image):
            b64 = self._image_to_base64_from_pil(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}
        }

    def _prepare_images(self, images: ImageInput) -> List[Dict[str, Any]]:
        """Convert image(s) to API format."""
        if images is None:
            return []
        if not isinstance(images, list):
            return [self._prepare_image_content(images)]
        return [self._prepare_image_content(img) for img in images]

    # ==================== API CALLS ====================

    @retry(stop=stop_after_attempt(MAX_RETRIES), retry=retry_if_exception_type(OpenAIError))
    def call_vision(
        self,
        prompt: str,
        image_path=None,
        image_bytes: bytes = None,
        pil_image: Image.Image = None,
        images: ImageInput = None,
        response_format: str = "text"
    ) -> str:
        """Call vision API with image(s)."""
        start_time = time.time()

        # Handle legacy parameters
        if images is None:
            images = image_path or image_bytes or pil_image

        content = [{"type": "text", "text": prompt}]
        content.extend(self._prepare_images(images))

        response = self.client.chat.completions.create(
            model=self.vision_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )

        result = response.choices[0].message.content or ""

        # Prepare images list for debug capture
        images_list = None
        if images:
            if isinstance(images, list):
                images_list = [str(img) if isinstance(img, (str, Path)) else "<bytes>" for img in images]
            else:
                images_list = [str(images) if isinstance(images, (str, Path)) else "<bytes>"]

        self._log_call(
            prompt_type="vision",
            input_summary=f"Vision: {prompt[:80]}...",
            response_summary=result[:200],
            duration_ms=(time.time() - start_time) * 1000,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
            full_prompt=prompt,
            full_response=result,
            images=images_list,
            model_name=self.vision_model
        )

        return result

    @retry(stop=stop_after_attempt(MAX_RETRIES), retry=retry_if_exception_type(OpenAIError))
    def call_text(
        self,
        prompt: str,
        system_prompt: str = None,
        response_format: str = "text"
    ) -> str:
        """Call text API."""
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

        self._log_call(
            prompt_type="text",
            input_summary=f"Text: {prompt[:80]}...",
            response_summary=result[:200],
            duration_ms=(time.time() - start_time) * 1000,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
            full_prompt=prompt,
            full_response=result,
            model_name=self.model
        )

        return result

    @retry(stop=stop_after_attempt(MAX_RETRIES), retry=retry_if_exception_type(OpenAIError))
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        response = self.client.embeddings.create(model=self.embedding_model, input=text)
        return response.data[0].embedding

    @retry(stop=stop_after_attempt(MAX_RETRIES), retry=retry_if_exception_type(OpenAIError))
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        response = self.client.embeddings.create(model=self.embedding_model, input=texts)
        return [item.embedding for item in response.data]

    def supports_context_caching(self) -> bool:
        """
        Check if context caching is available.

        Note: OpenAI doesn't have native context caching like Gemini.
        Conversation history is re-sent with each call, so there's no
        cost savings from "caching".

        Returns:
            False - OpenAI doesn't support native context caching
        """
        return False

    def create_cached_context(
        self,
        system_prompt: str,
        images: List[str] = None,
        ttl_seconds: int = 3600
    ) -> Optional[str]:
        """
        Create a cached context - NOT SUPPORTED by OpenAI.

        OpenAI doesn't have native context caching. This method logs a warning
        and returns None to indicate caching is not available.

        Args:
            system_prompt: System instruction (ignored)
            images: Optional list of image paths (ignored)
            ttl_seconds: Time-to-live (ignored)

        Returns:
            None - caching not supported
        """
        logger.warning(
            "Context caching NOT AVAILABLE: OpenAI does not support native context caching. "
            "Conversation history is re-sent with each call, resulting in full token costs. "
            "Consider using Gemini for context caching support to reduce costs."
        )
        return None

    def get_effective_token_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int
    ) -> Dict[str, float]:
        """
        Calculate effective token costs for OpenAI.

        Note: OpenAI charges full price for all tokens in conversation history,
        which is re-sent with each call in chat mode.

        Args:
            prompt_tokens: Number of prompt/input tokens
            completion_tokens: Number of completion/output tokens

        Returns:
            Dict with cost breakdown in USD
        """
        from config.constants import OPENAI_PRICING

        # Determine which model we're using
        model_lower = self.model.lower() if self.model else ""
        if "mini" in model_lower:
            input_price = OPENAI_PRICING.get("gpt4o_mini_input", 0.15)
            output_price = OPENAI_PRICING.get("gpt4o_mini_output", 0.60)
        else:
            input_price = OPENAI_PRICING.get("gpt4o_input", 2.50)
            output_price = OPENAI_PRICING.get("gpt4o_output", 10.00)

        input_cost = (prompt_tokens / 1_000_000) * input_price
        output_cost = (completion_tokens / 1_000_000) * output_price

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "cached_savings": 0.0  # No caching for OpenAI
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
            completion_tokens=usage["completion_tokens"]
        )
        return {
            "provider": self._name,
            "model": self.model,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "cached_tokens": 0,  # No caching for OpenAI
            "total_tokens": usage["total_tokens"],
            "calls": usage["calls"],
            "estimated_cost_usd": round(cost["total_cost"], 4),
            "input_cost_usd": round(cost["input_cost"], 4),
            "output_cost_usd": round(cost["output_cost"], 4),
            "cached_savings_usd": 0.0,
            "warning": "OpenAI doesn't support context caching. "
                       "Conversation history is re-sent with each call."
        }
