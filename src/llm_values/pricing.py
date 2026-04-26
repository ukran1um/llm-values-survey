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
    "gpt-5.5-pro-2026-04-23": (15.0, 60.0),  # reasoning-tuned pro tier (estimated)
    # Google
    "gemini-2.5-pro": (3.5, 10.5),
    # xAI
    "grok-4.20": (5.0, 15.0),
    # Groq (cheap fast inference; prices per M tokens)
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "meta-llama/llama-4-scout-17b-16e-instruct": (0.11, 0.34),
    "meta-llama/llama-4-maverick-17b-128e-instruct": (0.50, 0.77),
    "openai/gpt-oss-120b": (0.15, 0.60),  # estimated
    "deepseek-r1-distill-llama-70b": (0.75, 0.99),
    "llama-3.1-8b-instant": (0.05, 0.08),
    # Runware (estimated — Runware emphasizes lowest-cost)
    "minimax-m2-7": (1.0, 4.0),  # estimated
    "runware:qwen3-thinking@1": (0.5, 2.0),  # estimated
    # OpenRouter (markup ~5% over provider direct; estimated)
    "mistralai/mistral-large-2411": (2.1, 6.3),
    "deepseek/deepseek-chat": (0.28, 1.16),
    "moonshotai/kimi-k2": (1.0, 4.0),  # estimated
    "z-ai/glm-4.6": (0.55, 1.60),  # estimated
}


def calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return USD cost for a single chat call. Returns 0.0 for unknown models."""
    if model not in PRICING:
        return 0.0
    in_price, out_price = PRICING[model]
    return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000
