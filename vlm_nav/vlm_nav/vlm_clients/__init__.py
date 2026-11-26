from .base import VLMClient
from .gemini_client import GeminiClient
from .openai_client import OpenAIClient


def create_client(model: str, **kwargs) -> VLMClient:
    """Pick the provider from the model ID, e.g. "gemini-4-hydrogen" or "gpt-6.1-sol".

    kwargs go to the client: api_key (default: the provider's environment variable), temperature,
    max_output_tokens, max_retries, and for OpenAI also image_detail and reasoning_effort; for Gemini
    also thinking_budget (default 0, no thinking).
    """
    if model.startswith("gemini"):
        return GeminiClient(model, **kwargs)
    if model.startswith("gpt"):
        return OpenAIClient(model, **kwargs)
    raise ValueError(f"Unknown model '{model}'. Use an OpenAI (gpt-*) or Gemini (gemini-*) model ID.")
