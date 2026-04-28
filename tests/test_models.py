import os

import pytest

from llm_values.models import get_client, model_provider, MODEL_TO_PROVIDER, _CLIENT_CACHE


def setup_function():
    _CLIENT_CACHE.clear()


def test_model_provider_known():
    assert model_provider("claude-opus-4-7-20260416") == "anthropic"
    assert model_provider("gpt-5.5-2026-04-23") == "openai"
    assert model_provider("gemini-2.5-pro") == "google"
    assert model_provider("llama-3.3-70b-versatile") == "groq"
    assert model_provider("mistralai/mistral-large-2411") == "openrouter"


def test_model_provider_unknown_raises():
    with pytest.raises(KeyError):
        model_provider("does-not-exist")


def test_get_client_anthropic_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        get_client("claude-opus-4-7-20260416")


def test_get_client_anthropic_returns_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    c = get_client("claude-opus-4-7-20260416")
    assert c.__class__.__name__ == "AnthropicChatClient"


def test_get_client_memoizes(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    c1 = get_client("claude-opus-4-7-20260416")
    c2 = get_client("claude-opus-4-7-20260416")
    assert c1 is c2


def test_required_models_all_mapped():
    required = {"claude-opus-4-7-20260416", "claude-sonnet-4-6-20260217", "gpt-5.5-2026-04-23", "gemini-2.5-pro", "llama-3.3-70b-versatile", "mistralai/mistral-large-2411", "minimax-m2-7"}
    assert required.issubset(set(MODEL_TO_PROVIDER.keys()))


def test_provider_extras_has_required_keys():
    from llm_values.models import PROVIDER_EXTRAS
    required = {"anthropic", "openai", "google", "xai", "groq", "openrouter", "runware"}
    assert required == set(PROVIDER_EXTRAS.keys())


def test_provider_extras_openai_disables_thinking():
    from llm_values.models import PROVIDER_EXTRAS
    assert PROVIDER_EXTRAS["openai"] == {"reasoning_effort": "low", "temperature": 1.0}


def test_provider_extras_runware_token_budget():
    from llm_values.models import PROVIDER_EXTRAS
    assert PROVIDER_EXTRAS["runware"]["max_completion_tokens"] == 1500
    assert PROVIDER_EXTRAS["runware"]["temperature"] == 1.0


def test_provider_extras_groq_hides_reasoning():
    from llm_values.models import PROVIDER_EXTRAS
    assert PROVIDER_EXTRAS["groq"]["reasoning_format"] == "hidden"


def test_provider_extras_openrouter_disables_reasoning():
    from llm_values.models import PROVIDER_EXTRAS
    assert PROVIDER_EXTRAS["openrouter"]["reasoning"] == {"enabled": False}
