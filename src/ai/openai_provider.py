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

from ai.base_provider import BaseProvider
from config.constants import MAX_RETRIES, MAX_TOKENS, TEMPERATURE, API_CONNECT_TIMEOUT, API_READ_TIMEOUT

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

        self._log_call(
            prompt_type="vision",
            input_summary=f"Vision: {prompt[:80]}...",
            response_summary=result[:200],
            duration_ms=(time.time() - start_time) * 1000,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None
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
            completion_tokens=response.usage.completion_tokens if response.usage else None
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
