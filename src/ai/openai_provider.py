"""
OpenAI-compatible API provider for vision and text interactions.

Supports:
- OpenAI (GPT-4o, etc.)
- OpenRouter (unified access to multiple LLMs)
- Any OpenAI-compatible API
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
from config.settings import get_settings
from config.constants import MAX_RETRIES, MAX_TOKENS, TEMPERATURE, API_CONNECT_TIMEOUT, API_READ_TIMEOUT


# Type alias for image input (single or multiple)
ImageInput = Union[str, Path, bytes, Image.Image, List[Union[str, Path, bytes, Image.Image]]]


class OpenAIProvider(BaseProvider):
    """
    Provider for OpenAI-compatible API interactions.

    Supports OpenAI, OpenRouter, and any OpenAI-compatible API.
    Handles vision, text, and embedding operations.
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        vision_model: str = None,
        embedding_model: str = None,
        name: str = None,
        mock_mode: bool = False,
        extra_headers: Dict[str, str] = None,
        **kwargs
    ):
        """
        Initialize the OpenAI-compatible provider.

        Args:
            api_key: API key (default: from settings)
            base_url: Custom base URL (e.g., https://openrouter.ai/api/v1)
            model: Text model name
            vision_model: Vision model name (defaults to model)
            embedding_model: Embedding model name
            name: Provider name for display (defaults to model name)
            mock_mode: Skip API key check for testing
            extra_headers: Additional headers for API calls
            **kwargs: Additional arguments passed to OpenAI client
        """
        super().__init__(mock_mode=mock_mode)

        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key

        if not self.api_key and not mock_mode:
            raise ValueError("API key required. Set AI_CORRECTION_OPENAI_API_KEY or pass api_key.")

        # Store model names
        self.model = model or settings.model
        self.vision_model = vision_model or settings.vision_model or self.model
        self.embedding_model = embedding_model or settings.embedding_model
        self._name = name or self.model

        if not mock_mode:
            self.client = self._create_client(base_url, extra_headers, kwargs, settings)

    def _create_client(
        self,
        base_url: Optional[str],
        extra_headers: Optional[Dict[str, str]],
        kwargs: Dict,
        settings
    ) -> OpenAI:
        """Create and configure the OpenAI client."""
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

        # Set base_url
        if base_url:
            client_kwargs["base_url"] = base_url
        elif settings.openai_organization:
            client_kwargs["base_url"] = settings.openai_organization

        # Add extra headers
        if extra_headers:
            client_kwargs["default_headers"] = extra_headers

        # Add any additional kwargs
        client_kwargs.update(kwargs)

        return OpenAI(**client_kwargs)

    @property
    def name(self) -> str:
        """Provider name for display."""
        return self._name

    # ==================== FACTORY METHODS ====================

    @classmethod
    def create_openrouter(
        cls,
        api_key: str = None,
        model: str = "anthropic/claude-3.5-sonnet",
        vision_model: str = None,
        app_name: str = "AI-Correction",
        app_url: str = "https://github.com/odurif1/AI-Correction-System"
    ) -> "OpenAIProvider":
        """
        Create a provider configured for OpenRouter.

        Args:
            api_key: OpenRouter API key
            model: Model name (e.g., "anthropic/claude-3.5-sonnet")
            vision_model: Vision model (defaults to model)
            app_name: App name for OpenRouter rankings
            app_url: App URL for OpenRouter rankings

        Popular models:
            - anthropic/claude-3.5-sonnet
            - openai/gpt-4o
            - google/gemini-2.0-flash-exp
            - deepseek/deepseek-r1
        """
        return cls(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model=model,
            vision_model=vision_model or model,
            name=f"OpenRouter/{model.split('/')[-1]}",
            extra_headers={
                "HTTP-Referer": app_url,
                "X-Title": app_name
            }
        )

    @classmethod
    def create_openai(
        cls,
        api_key: str = None,
        model: str = "gpt-4o",
        vision_model: str = "gpt-4o"
    ) -> "OpenAIProvider":
        """Create a provider for OpenAI's API."""
        return cls(
            api_key=api_key,
            model=model,
            vision_model=vision_model,
            name=f"OpenAI/{model}"
        )

    # ==================== IMAGE UTILITIES ====================

    def _prepare_image_content(self, image: Union[str, Path, bytes, Image.Image]) -> Dict[str, Any]:
        """Convert single image to API content format."""
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
            "image_url": {
                "url": f"data:image/png;base64,{b64}",
                "detail": "high"
            }
        }

    def _prepare_images(self, images: ImageInput) -> List[Dict[str, Any]]:
        """Convert image(s) to API content format (supports multiple images)."""
        if images is None:
            return []

        # Single image
        if not isinstance(images, list):
            return [self._prepare_image_content(images)]

        # Multiple images
        return [self._prepare_image_content(img) for img in images]

    # ==================== API CALLS ====================

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        retry=retry_if_exception_type(OpenAIError)
    )
    def call_vision(
        self,
        prompt: str,
        image_path=None,
        image_bytes: bytes = None,
        pil_image: Image.Image = None,
        images: ImageInput = None,
        response_format: str = "text"
    ) -> str:
        """
        Call the vision API with image(s).

        Args:
            prompt: Text prompt
            image_path: Single image path (legacy)
            image_bytes: Single image bytes (legacy)
            pil_image: Single PIL image (legacy)
            images: Single image or list of images (preferred)
            response_format: "text" or "json"

        Returns:
            Model response text
        """
        start_time = time.time()

        # Handle legacy single-image parameters
        if images is None:
            images = image_path or image_bytes or pil_image

        # Build content
        content = [{"type": "text", "text": prompt}]
        content.extend(self._prepare_images(images))

        messages = [{"role": "user", "content": content}]

        response = self.client.chat.completions.create(
            model=self.vision_model,
            messages=messages,
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

        self._log_call(
            prompt_type="text",
            input_summary=f"Text: {prompt[:80]}...",
            response_summary=result[:200],
            duration_ms=(time.time() - start_time) * 1000,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None
        )

        return result

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        retry=retry_if_exception_type(OpenAIError)
    )
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
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
        """Get embeddings for multiple texts."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [item.embedding for item in response.data]
