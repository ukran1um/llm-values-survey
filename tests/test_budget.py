from pathlib import Path

import pytest

from llm_values.budget import Budget, BudgetExceeded


def test_starts_empty(tmp_path: Path):
    b = Budget(state_path=tmp_path / "budget.json", cap_usd=10.0)
    assert b.spent_usd == 0.0
    assert b.remaining_usd == 10.0


def test_add_persists(tmp_path: Path):
    state = tmp_path / "budget.json"
    b1 = Budget(state_path=state, cap_usd=10.0)
    b1.add(2.5)
    assert b1.spent_usd == 2.5

    b2 = Budget(state_path=state, cap_usd=10.0)
    assert b2.spent_usd == 2.5


def test_raises_when_cap_exceeded(tmp_path: Path):
    b = Budget(state_path=tmp_path / "budget.json", cap_usd=1.0)
    b.add(0.9)
    with pytest.raises(BudgetExceeded):
        b.add(0.2)
    assert b.spent_usd == 0.9  # not incremented on failure
