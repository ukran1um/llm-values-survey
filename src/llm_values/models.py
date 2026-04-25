from __future__ import annotations
import os
from typing import Protocol

from .types import ChatMessage, ChatResponse


class ChatClient(Protocol):
    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float = 1.0,
        max_tokens: int = 2000,
    ) -> ChatResponse: ...


# model_id → provider key
MODEL_TO_PROVIDER: dict[str, str] = {
    # Anthropic
    "claude-opus-4-7": "anthropic",
    "claude-sonnet-4-6": "anthropic",
    # OpenAI
    "gpt-5": "openai",
    # Google
    "gemini-2.5-pro": "google",
    # xAI (OpenAI-compatible)
    "grok-4": "xai",
    # DeepSeek (OpenAI-compatible)
    "deepseek-v3": "deepseek",
    # OpenRouter-served (OpenAI-compatible)
    "qwen3-max": "openrouter",
    "glm-4.5": "openrouter",
    # Together AI (OpenAI-compatible)
    "llama-3.3-70b": "together",
    "mistral-large-2": "together",
    "phi-4": "together",
    "qwen3-7b": "together",
}

# provider → (env var for API key, base URL or None)
PROVIDER_CONFIG: dict[str, tuple[str, str | None]] = {
    "anthropic": ("ANTHROPIC_API_KEY", None),
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1"),
    "google": ("GOOGLE_API_KEY", None),
    "xai": ("XAI_API_KEY", "https://api.x.ai/v1"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
    "together": ("TOGETHER_API_KEY", "https://api.together.xyz/v1"),
}

_CLIENT_CACHE: dict[str, ChatClient] = {}


def model_provider(model: str) -> str:
    if model not in MODEL_TO_PROVIDER:
        raise KeyError(f"unknown model: {model!r}")
    return MODEL_TO_PROVIDER[model]


def get_client(model: str) -> ChatClient:
    """Return a cached ChatClient for the given model, instantiating lazily."""
    provider = model_provider(model)
    if provider in _CLIENT_CACHE:
        return _CLIENT_CACHE[provider]

    env_var, base_url = PROVIDER_CONFIG[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise RuntimeError(f"missing env var: {env_var}")

    if provider == "anthropic":
        from .clients.anthropic_client import AnthropicChatClient
        client: ChatClient = AnthropicChatClient(api_key=api_key)
    elif provider == "google":
        from .clients.google_client import GoogleChatClient
        client = GoogleChatClient(api_key=api_key)
    else:
        from .clients.openai_compat import OpenAICompatClient
        assert base_url is not None
        client = OpenAICompatClient(base_url=base_url, api_key=api_key)

    _CLIENT_CACHE[provider] = client
    return client
