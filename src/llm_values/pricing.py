from __future__ import annotations

# USD per 1,000,000 tokens, (input_price, output_price).
# Prices are illustrative as of plan-write date; final values resolved against
# provider pages when the run executes. Update before pre-registration commit.
PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-7": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    # OpenAI
    "gpt-5.5-2026-04-23": (5.0, 20.0),
    # Google
    "gemini-2.5-pro": (3.5, 10.5),
    # xAI
    "grok-4.20": (5.0, 15.0),
    # Groq
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "meta-llama/llama-4-scout-17b-16e-instruct": (0.11, 0.34),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "qwen/qwen3-32b": (0.29, 0.59),
    # Runware
    "minimax-m2-7": (1.0, 4.0),
    # OpenRouter (~5% markup over provider direct)
    "mistralai/mistral-large-2411": (2.1, 6.3),
    "deepseek/deepseek-chat": (0.28, 1.16),
    "moonshotai/kimi-k2": (1.0, 4.0),
    "z-ai/glm-4.6": (0.55, 1.60),
}


def calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return USD cost for a single chat call. Returns 0.0 for unknown models."""
    if model not in PRICING:
        return 0.0
    in_price, out_price = PRICING[model]
    return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000
