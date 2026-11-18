from typing import List, Optional, Type

from PIL import Image

from .base import T, VLMClient, encode_jpeg


class GeminiClient(VLMClient):
    """Google Gen AI SDK with a JSON response schema. Reads GEMINI_API_KEY or GOOGLE_API_KEY from the environment.

    thinking_budget caps the thinking tokens; the default 0 asks for no thinking so the labels come back quickly
    (some models, e.g. gemini-4-hydrogen, still think a little). None leaves the model's own default.
    """

    def __init__(self, model: str, thinking_budget: Optional[int] = 0, **kwargs):
        super().__init__(model, **kwargs)
        self.thinking_budget = thinking_budget
        try:
            from google import genai
        except ImportError as e:
            raise ImportError('Gemini models need google-genai: pip install -e "vlm_nav[gemini]" (Python 3.9+)') from e

        self.client = genai.Client()

    def _ask(self, images: List[Image.Image], prompt: str, schema: Type[T]) -> T:
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
