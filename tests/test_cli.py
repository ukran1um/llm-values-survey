from pathlib import Path

from click.testing import CliRunner

from llm_values.cli import main


def test_status_command(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "spent" in result.output.lower()
    assert "verdicts on disk" in result.output


def test_cost_command(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "budget.json").write_text('{"spent_usd": 0.42}')
    runner = CliRunner()
    result = runner.invoke(main, ["cost"])
    assert result.exit_code == 0
    assert "0.42" in result.output


def test_run_axis_command_invokes_runner(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "probes").mkdir()
    (tmp_path / "probes" / "pilot.json").write_text(
        '[{"id":"a","battery":"pilot","description":"d","verdict_format":{"type":"binary","options":["x","y"]},"max_turns":2}]'
    )

    captured = {}

    def fake_run_axis(axis, models, n_reruns, data_dir, budget, **kwargs):
        captured["axis_id"] = axis.id
        captured["models"] = models
        captured["n_reruns"] = n_reruns
        return {"ok": 2, "skipped": 0, "failed": 0, "total_cost": 0.01}

    import llm_values.cli as cli
    monkeypatch.setattr(cli, "run_axis", fake_run_axis)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run-axis", "a",
            "--battery", "pilot",
            "--models", "claude-opus-4-7,gpt-5.5-2026-04-23",
            "--reruns", "1",
            "--cap", "5.0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["axis_id"] == "a"
    assert captured["models"] == ["claude-opus-4-7", "gpt-5.5-2026-04-23"]
    assert captured["n_reruns"] == 1


def test_run_axis_rejects_single_model(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "probes").mkdir()
    (tmp_path / "probes" / "pilot.json").write_text(
        '[{"id":"a","battery":"pilot","description":"d","verdict_format":{"type":"binary","options":["x","y"]},"max_turns":2}]'
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run-axis", "a", "--battery", "pilot", "--models", "OnlyOne", "--reruns", "1", "--cap", "5.0"],
    )
    assert result.exit_code != 0
    assert "at least 2" in result.output.lower()


def test_run_self_report_stores_rating_and_justification(tmp_path: Path, monkeypatch):
    """Self-report response includes rating + justification; verifies new JSON contract."""
    import json as _json
    from llm_values.clients.mock import MockChatClient
    from llm_values.types import ChatResponse

    monkeypatch.chdir(tmp_path)

    # Build a minimal probes/mfq.json with exactly 36 mfq2_* items.
    (tmp_path / "probes").mkdir()
    items = [
        {
            "id": f"mfq2_{i:02d}",
            "battery": "mfq",
            "description": f"Statement {i}",
            "verdict_format": {"type": "scale", "min": 1, "max": 5},
            "max_turns": 2,
        }
        for i in range(36)
    ]
    (tmp_path / "probes" / "mfq.json").write_text(_json.dumps(items))

    # Patch get_client to return a mock that emits rating+justification JSON.
    import llm_values.cli as cli_module

    _RESPONSE_TEXT = '{"rating": 4, "justification": "I lean caring."}'

    class _FakeClient:
        def chat(self, model, messages, temperature=1.0, max_tokens=150, extras=None):
            return ChatResponse(
                text=_RESPONSE_TEXT,
                prompt_tokens=10,
                completion_tokens=20,
                cost_usd=0.0001,
                model=model,
                stop_reason="end_turn",
            )

    monkeypatch.setattr("llm_values.models.get_client", lambda m: _FakeClient())
    monkeypatch.setattr("llm_values.models.model_provider", lambda m: "anthropic")
    monkeypatch.setattr("llm_values.models.PROVIDER_EXTRAS", {"anthropic": {}})
    monkeypatch.setattr("llm_values.models.MODEL_TO_PROVIDER", {"claude-sonnet-4-6": "anthropic"})

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run-self-report",
            "--models", "claude-sonnet-4-6",
            "--cap", "1.0",
            "--probes-dir", str(tmp_path / "probes"),
            "--data-dir", str(tmp_path / "data"),
        ],
    )
    assert result.exit_code == 0, result.output

    out_file = tmp_path / "data" / "raw" / "self_reports" / "claude-sonnet-4-6.json"
    assert out_file.exists()
    data = _json.loads(out_file.read_text())
    # Check first item has the new contract
    first_item = data["responses"]["mfq2_00"]
    assert first_item["rating"] == 4
    assert first_item["justification"] == "I lean caring."
