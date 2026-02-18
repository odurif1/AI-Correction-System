"""
OpenAI API provider for vision and text interactions.

Handles communication with OpenAI's GPT-4o and embedding models.
"""

import base64
import time
from typing import List, Dict, Any
import httpx

from openai import OpenAI, OpenAIError
from tenacity import retry, stop_after_attempt, retry_if_exception_type
from PIL import Image

from ai.base_provider import BaseProvider
from config.settings import get_settings
from config.constants import MAX_RETRIES, MAX_TOKENS, TEMPERATURE, API_CONNECT_TIMEOUT, API_READ_TIMEOUT


class OpenAIProvider(BaseProvider):
    """
    Provider for OpenAI API interactions.

    Inherits from BaseProvider for shared functionality.
    Implements OpenAI-specific API calls.
    """

    def __init__(self, api_key: str = None, base_url: str = None, mock_mode: bool = False):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key (default: from settings)
            base_url: Custom base URL (default: from settings)
            mock_mode: If True, skip API key check for testing
        """
        super().__init__(mock_mode=mock_mode)

        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key

        if not self.api_key and not mock_mode:
            raise ValueError("OpenAI API key is required. Set AI_CORRECTION_OPENAI_API_KEY.")

        if not mock_mode:
            # Configure timeout for API calls
            timeout = httpx.Timeout(
                connect=API_CONNECT_TIMEOUT,
                read=API_READ_TIMEOUT,
                write=API_CONNECT_TIMEOUT,
                pool=API_CONNECT_TIMEOUT
            )
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=base_url or settings.openai_organization,
                timeout=timeout
            )

        self.model = settings.model
        self.vision_model = settings.vision_model
        self.embedding_model = settings.embedding_model

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
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}
            }
        elif pil_image:
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self._image_to_base64_from_pil(pil_image)}",
                    "detail": "high"
                }
            }

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        if image_content:
            messages[0]["content"].append(image_content)

        response = self.client.chat.completions.create(
            model=self.vision_model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )

        result = response.choices[0].message.content or ""

        # Log call using base class method
        self._log_call(
            prompt_type="vision",
            input_summary=f"Vision prompt: {prompt[:100]}...",
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

        # Log call using base class method
        self._log_call(
            prompt_type="text",
            input_summary=f"Text prompt: {prompt[:100]}...",
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
