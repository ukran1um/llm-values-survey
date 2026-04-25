from __future__ import annotations

# USD per 1,000,000 tokens, (input_price, output_price).
# Prices are illustrative as of plan-write date; final values resolved against
# provider pages when the run executes. Update before pre-registration commit.
PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-7": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    # OpenAI (dated snapshot of GPT-5.5)
    "gpt-5.5-2026-04-23": (5.0, 20.0),
    # Google
    "gemini-2.5-pro": (3.5, 10.5),
    # xAI
    "grok-4.20": (5.0, 15.0),
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    # OpenRouter
    "qwen/qwen3.6-plus": (1.5, 6.0),
    "z-ai/glm-4.7": (0.50, 1.50),
    # Together AI
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (0.88, 0.88),
    "mistralai/Mistral-Large-Instruct-2411": (3.0, 9.0),
    "microsoft/Phi-4": (0.30, 0.30),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": (0.10, 0.30),
}


def calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return USD cost for a single chat call. Returns 0.0 for unknown models."""
    if model not in PRICING:
        return 0.0
    in_price, out_price = PRICING[model]
    return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000
