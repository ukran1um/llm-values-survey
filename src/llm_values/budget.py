from __future__ import annotations
import json
from pathlib import Path


class BudgetExceeded(Exception):
    pass


class Budget:
    """Persistent cost tracker. Reads/writes state on every add()."""

    def __init__(self, state_path: Path, cap_usd: float):
        self.state_path = Path(state_path)
        self.cap_usd = cap_usd
        self._spent = self._load()

    def _load(self) -> float:
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return float(data.get("spent_usd", 0.0))
        return 0.0

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({"spent_usd": self._spent}, indent=2),
            encoding="utf-8",
        )

    def add(self, cost_usd: float) -> None:
        if self._spent + cost_usd > self.cap_usd:
            raise BudgetExceeded(
                f"would exceed cap (${self.cap_usd:.2f}); "
                f"spent ${self._spent:.4f}, attempted +${cost_usd:.4f}"
            )
        self._spent += cost_usd
        self._save()

    @property
    def spent_usd(self) -> float:
        return self._spent

    @property
    def remaining_usd(self) -> float:
        return self.cap_usd - self._spent
