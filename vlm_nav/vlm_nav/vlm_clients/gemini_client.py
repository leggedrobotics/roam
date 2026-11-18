import os
from typing import List, Optional, Type

from PIL import Image

from .base import T, VLMClient, encode_jpeg, encode_jpeg_base64

# Gemini's OpenAI-compatible endpoint, used where google-genai cannot be installed (Python 3.8, e.g. ROS Noetic).
OPENAI_COMPAT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiClient(VLMClient):
    """Google Gen AI SDK with a JSON response schema. Reads GEMINI_API_KEY or GOOGLE_API_KEY from the environment.

    thinking_budget caps the thinking tokens; the default 0 asks for no thinking so the labels come back quickly
    (some models, e.g. gemini-4-hydrogen, still think a little). None leaves the model's own default.
    Without google-genai (Python 3.8) the same request goes through the openai package and Gemini's
    OpenAI-compatible endpoint; thinking is then either off (thinking_budget=0) or the model default.
    """

    def __init__(self, model: str, thinking_budget: Optional[int] = 0, **kwargs):
        super().__init__(model, **kwargs)
        self.thinking_budget = thinking_budget
        try:
            from google import genai
        except ImportError:
            from openai import OpenAI

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            self.client = None
            self.compat_client = OpenAI(api_key=api_key, base_url=OPENAI_COMPAT_BASE_URL)
        else:
            self.client = genai.Client()
            self.compat_client = None

    def _ask(self, images: List[Image.Image], prompt: str, schema: Type[T]) -> T:
        if self.compat_client is not None:
            return self._ask_compat(images, prompt, schema)
        from google.genai import types

        contents = [prompt] + [types.Part.from_bytes(data=encode_jpeg(image), mime_type="image/jpeg") for image in images]
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            thinking_config=None
            if self.thinking_budget is None
            else types.ThinkingConfig(thinking_budget=self.thinking_budget),
        )
        response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
        if not response.text:
            raise ValueError(f"Empty response (prompt feedback: {response.prompt_feedback})")
        return schema.model_validate_json(response.text)

    def _ask_compat(self, images: List[Image.Image], prompt: str, schema: Type[T]) -> T:
        content = [{"type": "text", "text": prompt}]
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_jpeg_base64(image)}"}})
        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            kwargs["max_tokens"] = self.max_output_tokens
        if self.thinking_budget == 0:
            kwargs["reasoning_effort"] = "none"
        response = self.compat_client.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_format=schema,
            **kwargs,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError(f"No parsed output (finish reason: {response.choices[0].finish_reason})")
        return parsed
