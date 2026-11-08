import base64
import io
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Type, TypeVar

from PIL import Image
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def encode_jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


def encode_jpeg_base64(image: Image.Image) -> str:
    return base64.b64encode(encode_jpeg(image)).decode("utf-8")


class VLMClient(ABC):
    """Asks a vision-language model about images and returns a report validated against a pydantic schema."""

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        max_retries: int = 3,
    ):
        self.model = model
        # None leaves the provider default, since some reasoning models reject a temperature.
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries

    def ask(self, images: List[Image.Image], prompt: str, schema: Type[T]) -> Optional[T]:
        for attempt in range(self.max_retries):
            try:
                return self._ask(images, prompt, schema)
            except Exception as e:
                print(f"[{self.model}] request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt + 1 < self.max_retries:
                    time.sleep(2**attempt)
        return None

    @abstractmethod
    def _ask(self, images: List[Image.Image], prompt: str, schema: Type[T]) -> T:
        """Send one request. Raise on failure so ask() can retry."""
