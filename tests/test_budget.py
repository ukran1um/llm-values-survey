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


def test_budget_thread_safe_add(tmp_path: Path):
    """Under concurrent add(), the cap should hold even with thread races."""
    import threading
    state = tmp_path / "budget.json"
    cap = 1.0
    b = Budget(state_path=state, cap_usd=cap)

    successes = []
    failures = []

    def worker():
        try:
            b.add(0.30)  # 4 successful adds = 1.20 > cap; at least one must fail
            successes.append(1)
        except BudgetExceeded:
            failures.append(1)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()

    # If lock works correctly, exactly 3 should succeed (3 × 0.30 = 0.90 ≤ 1.0; 4th would push to 1.20 > 1.0).
    # Without a lock, all 8 could race past the check and we'd see > 3 successes.
    assert len(successes) == 3, f"expected exactly 3 successes (lock failed: {len(successes)} succeeded)"
    assert len(failures) == 5
    assert b.spent_usd == pytest.approx(0.90)


def test_multi_instance_interleave(tmp_path: Path):
    state = tmp_path / "budget.json"
    b1 = Budget(state_path=state, cap_usd=10.0)
    b2 = Budget(state_path=state, cap_usd=10.0)
    b1.add(1.0)
    # b2.add() re-reads disk first, so it sees b1's 1.0 before adding 2.0 → 3.0 on disk
    b2.add(2.0)
    assert b2.spent_usd == 3.0
    # b1.add() re-reads disk (3.0) then adds 0.5 → 3.5 on disk
    b1.add(0.5)
    assert b1.spent_usd == 3.5
    # b2._spent is now stale (still 3.0) — the spent_usd property reflects last add(), not disk
    assert b2.spent_usd == 3.0
    # Reconstructing reads canonical disk state
    b3 = Budget(state_path=state, cap_usd=10.0)
    assert b3.spent_usd == 3.5
