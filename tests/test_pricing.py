import pytest

from llm_values.pricing import calc_cost, PRICING


def test_calc_cost_known_model():
    # claude-opus-4-7-20260416: 5.0 in / 25.0 out per 1M (verified against Anthropic pricing 2026-04-26)
    cost = calc_cost("claude-opus-4-7-20260416", prompt_tokens=1_000_000, completion_tokens=0)
    assert cost == pytest.approx(5.0)
    cost = calc_cost("claude-opus-4-7-20260416", prompt_tokens=0, completion_tokens=1_000_000)
    assert cost == pytest.approx(25.0)


def test_calc_cost_small_amounts():
    cost = calc_cost("gpt-5.5-2026-04-23", prompt_tokens=1000, completion_tokens=500)
    # $5/M in, $30/M out → 1000*5/1e6 + 500*30/1e6 = 0.005 + 0.015 = 0.020
    assert cost == pytest.approx(0.020)


def test_calc_cost_unknown_model_returns_zero():
    cost = calc_cost("does-not-exist", prompt_tokens=1000, completion_tokens=1000)
    assert cost == 0.0


def test_pricing_table_has_required_models():
    required = {"claude-opus-4-7-20260416", "claude-sonnet-4-6-20260217", "gpt-5.5-2026-04-23", "gemini-2.5-pro", "mistralai/mistral-large-2411", "minimax-m2-7"}
    assert required.issubset(set(PRICING.keys()))
