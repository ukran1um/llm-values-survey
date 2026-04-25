from __future__ import annotations
import json
from pathlib import Path

from .types import Axis


def load_probe_file(path: Path) -> list[Axis]:
    """Load and validate a probe JSON file. Raises pydantic ValidationError on schema problems."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"probe file {path} must contain a JSON list at top level")
    return [Axis.model_validate(item) for item in raw]


def load_battery(probes_dir: Path, battery: str) -> list[Axis]:
    """Load all axes for a given battery from <probes_dir>/<battery>.json."""
    path = Path(probes_dir) / f"{battery}.json"
    return load_probe_file(path)
