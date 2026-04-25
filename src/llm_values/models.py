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
# Roster locked April 2026. Update before pre-registration commit if newer
# snapshots become available, but do not update silently mid-run.
MODEL_TO_PROVIDER: dict[str, str] = {
    # Anthropic (bare names resolve to current snapshot per Anthropic API conventions)
    "claude-opus-4-7": "anthropic",
    "claude-sonnet-4-6": "anthropic",
    # OpenAI (dated snapshot for reproducibility)
    "gpt-5.5-2026-04-23": "openai",
    # Google
    "gemini-2.5-pro": "google",
    # xAI (OpenAI-compatible)
    "grok-4.20": "xai",
    # DeepSeek (OpenAI-compatible; deepseek-chat alias routes to V3.2; deprecates 2026-07-24)
    "deepseek-chat": "deepseek",
    # OpenRouter-served (OpenAI-compatible)
    "qwen/qwen3.6-plus": "openrouter",
    "z-ai/glm-4.7": "openrouter",
    # Together AI (OpenAI-compatible)
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "together",
    "mistralai/Mistral-Large-Instruct-2411": "together",
    "microsoft/Phi-4": "together",
    "Qwen/Qwen2.5-7B-Instruct-Turbo": "together",
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
