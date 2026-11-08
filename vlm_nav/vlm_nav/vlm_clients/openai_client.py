from typing import List, Optional, Type

from PIL import Image

from .base import T, VLMClient, encode_jpeg_base64


class OpenAIClient(VLMClient):
    """OpenAI Responses API with Structured Outputs. Reads OPENAI_API_KEY from the environment."""

    def __init__(self, model: str, image_detail: str = "auto", reasoning_effort: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)
        from openai import OpenAI

        self.client = OpenAI()
        self.image_detail = image_detail
        self.reasoning_effort = reasoning_effort

    def _ask(self, images: List[Image.Image], prompt: str, schema: Type[T]) -> T:
        content = [{"type": "input_text", "text": prompt}]
        for image in images:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encode_jpeg_base64(image)}",
                    "detail": self.image_detail,
                }
            )
        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens
        if self.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}

        response = self.client.responses.parse(
            model=self.model,
            input=[{"role": "user", "content": content}],
            text_format=schema,
            **kwargs,
        )
        if response.output_parsed is None:
            raise ValueError(f"No parsed output (status: {response.status})")
        return response.output_parsed
