from pathlib import Path

from click.testing import CliRunner

from llm_values.cli import main


def test_status_command(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "spent" in result.output.lower()


def test_cost_command(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # write a budget file
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
        '[{"id":"a","battery":"pilot","description":"d","labels":["x","y"]}]'
    )

    captured = {}

    def fake_run_axis(axis, models, judges, n_reruns, data_dir, budget, **kwargs):
        captured["axis_id"] = axis.id
        captured["models"] = models
        captured["judges"] = judges
        captured["n_reruns"] = n_reruns

    import llm_values.cli as cli
    monkeypatch.setattr(cli, "run_axis", fake_run_axis)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run-axis", "a",
            "--battery", "pilot",
            "--models", "M1,M2",
            "--judges", "J1",
            "--reruns", "1",
            "--cap", "5.0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["axis_id"] == "a"
    assert captured["models"] == ["M1", "M2"]
    assert captured["judges"] == ["J1"]
    assert captured["n_reruns"] == 1
