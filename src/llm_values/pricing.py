from __future__ import annotations

# USD per 1,000,000 tokens, (input_price, output_price).
# Prices are illustrative as of plan-write date; final values resolved against
# provider pages when the run executes. Update before pre-registration commit.
PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-7": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    # OpenAI
    "gpt-5": (5.0, 20.0),
    # Google
    "gemini-2.5-pro": (3.5, 10.5),
    # xAI
    "grok-4": (5.0, 15.0),
    # OpenRouter / Together / DeepSeek (illustrative averages)
    "deepseek-v3": (0.27, 1.10),
    "qwen3-max": (1.5, 6.0),
    "glm-4.5": (0.50, 1.50),
    "llama-3.3-70b": (0.88, 0.88),
    "mistral-large-2": (3.0, 9.0),
    "phi-4": (0.30, 0.30),
    "qwen3-7b": (0.10, 0.30),
}


def calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return USD cost for a single chat call. Returns 0.0 for unknown models."""
    if model not in PRICING:
        return 0.0
    in_price, out_price = PRICING[model]
    return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000
