import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_values.probes import load_probe_file, load_battery
from llm_values.types import Axis


def test_load_probe_file_returns_axes(tmp_path: Path):
    data = [
        {
            "id": "beatles_vs_stones",
            "battery": "pilot",
            "description": "Music canon: Beatles vs Rolling Stones.",
            "verdict_format": {"type": "binary", "options": ["beatles", "stones"]},
        }
    ]
    p = tmp_path / "pilot.json"
    p.write_text(json.dumps(data))

    axes = load_probe_file(p)
    assert len(axes) == 1
    assert isinstance(axes[0], Axis)
    assert axes[0].id == "beatles_vs_stones"
    assert axes[0].verdict_format.options == ["beatles", "stones"]


def test_load_probe_file_rejects_invalid_schema(tmp_path: Path):
    # binary verdict_format requires exactly 2 options — only one given here
    bad = [
        {
            "id": "x",
            "battery": "pilot",
            "description": "d",
            "verdict_format": {"type": "binary", "options": ["only_one"]},
        }
    ]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))

    with pytest.raises(ValidationError):
        load_probe_file(p)


def test_load_battery_finds_file(tmp_path: Path):
    data = [
        {
            "id": "a",
            "battery": "pilot",
            "description": "d",
            "verdict_format": {"type": "binary", "options": ["x", "y"]},
        }
    ]
    (tmp_path / "pilot.json").write_text(json.dumps(data))
    axes = load_battery(tmp_path, "pilot")
    assert len(axes) == 1
    assert axes[0].battery == "pilot"


def test_load_probe_file_rejects_non_list(tmp_path: Path):
    p = tmp_path / "obj.json"
    p.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError, match="must contain a JSON list"):
        load_probe_file(p)
