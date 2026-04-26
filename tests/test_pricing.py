import pytest

from llm_values.pricing import calc_cost, PRICING


def test_calc_cost_known_model():
    # claude-opus-4-7: 15.0 in / 75.0 out per 1M
    cost = calc_cost("claude-opus-4-7", prompt_tokens=1_000_000, completion_tokens=0)
    assert cost == pytest.approx(15.0)
    cost = calc_cost("claude-opus-4-7", prompt_tokens=0, completion_tokens=1_000_000)
    assert cost == pytest.approx(75.0)


def test_calc_cost_small_amounts():
    cost = calc_cost("gpt-5.5-2026-04-23", prompt_tokens=1000, completion_tokens=500)
    # $5/M in, $20/M out → 1000*5/1e6 + 500*20/1e6 = 0.005 + 0.010 = 0.015
    assert cost == pytest.approx(0.015)


def test_calc_cost_unknown_model_returns_zero():
    cost = calc_cost("does-not-exist", prompt_tokens=1000, completion_tokens=1000)
    assert cost == 0.0


def test_pricing_table_has_required_models():
    required = {"claude-opus-4-7", "claude-sonnet-4-6", "gpt-5.5-2026-04-23", "gemini-2.5-pro", "mistralai/mistral-large-2411"}
    assert required.issubset(set(PRICING.keys()))
