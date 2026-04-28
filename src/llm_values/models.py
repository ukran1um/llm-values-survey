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
        extras: dict | None = None,
    ) -> ChatResponse: ...


# model_id → provider key
# Roster locked April 2026. Update before pre-registration commit if newer
# snapshots become available, but do not update silently mid-run.
MODEL_TO_PROVIDER: dict[str, str] = {
    # Anthropic — frontier closed Western
    "claude-opus-4-7-20260416": "anthropic",
    "claude-sonnet-4-6-20260217": "anthropic",
    # OpenAI — frontier closed Western
    "gpt-5.5-2026-04-23": "openai",
    # Google — frontier closed Western
    "gemini-2.5-pro": "google",
    # xAI — alt-ideology Western
    "grok-4.20": "xai",
    # Groq — open-weight Western + Chinese
    "llama-3.3-70b-versatile": "groq",
    "meta-llama/llama-4-scout-17b-16e-instruct": "groq",
    "openai/gpt-oss-120b": "groq",
    "llama-3.1-8b-instant": "groq",
    "qwen/qwen3-32b": "groq",
    # Runware — Chinese frontier (MiniMax)
    "minimax-m2-7": "runware",
    # OpenRouter — European + Chinese frontier
    "mistralai/mistral-large-2411": "openrouter",
    "deepseek/deepseek-chat": "openrouter",
    "moonshotai/kimi-k2": "openrouter",
    "z-ai/glm-4.6": "openrouter",
}

# provider → (env var for API key, base URL or None for SDK-based)
PROVIDER_CONFIG: dict[str, tuple[str, str | None]] = {
    "anthropic": ("ANTHROPIC_API_KEY", None),
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1"),
    "google": ("GEMINI_API_KEY", None),
    "xai": ("XAI_API_KEY", "https://api.x.ai/v1"),
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1"),
    "runware": ("RUNWARE_API_KEY", "https://api.runware.ai/v1"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
}

# Per-provider request-body extras to disable thinking/reasoning modes for methodological consistency.
# Keys are provider names; values are dicts merged into the request body or config.
# anthropic: empty (claude-4.x default is no thinking unless explicitly requested)
# google: empty (handled directly in google_client.py — special config field, not body merge)
# runware: empty (no thinking parameter exposed in v1)
PROVIDER_EXTRAS: dict[str, dict] = {
    "anthropic": {},
    "openai": {"reasoning_effort": "low", "temperature": 1.0},
    "google": {},
    "xai": {"reasoning_effort": "low"},
    "groq": {"reasoning_format": "hidden"},
    "openrouter": {"reasoning": {"enabled": False}},
    "runware": {
        "max_completion_tokens": 1500,  # minimax-m2-7 needs ~1200 tokens for chain-of-thought before content
        "temperature": 1.0,  # minimax-m2-7 requires temperature > 0 (exclusive); override the verdict call's 0.0
    },
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
