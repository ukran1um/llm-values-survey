from __future__ import annotations
import json
from pathlib import Path

import click

from .budget import Budget
from .probes import load_battery
from .runner import run_axis as _run_axis


DATA_DIR_DEFAULT = Path("data")
BUDGET_FILE_DEFAULT = DATA_DIR_DEFAULT / "budget.json"


# Re-exported for tests to monkeypatch
run_axis = _run_axis


@click.group()
def main():
    """llm-values-survey CLI."""


@main.command("run-axis")
@click.argument("axis_id")
@click.option("--battery", required=True, help="probe battery name (e.g. pilot, mft, anglophone)")
@click.option("--models", required=True, help="comma-separated model ids")
@click.option("--judges", required=True, help="comma-separated judge model ids")
@click.option("--reruns", default=5, show_default=True, type=int)
@click.option("--cap", default=2500.0, show_default=True, type=float, help="hard budget cap in USD")
@click.option("--probes-dir", default="probes", show_default=True, type=click.Path(path_type=Path))
@click.option("--data-dir", default=str(DATA_DIR_DEFAULT), show_default=True, type=click.Path(path_type=Path))
def run_axis_cmd(axis_id, battery, models, judges, reruns, cap, probes_dir, data_dir):
    """Run all directed pairs × reruns for one axis."""
    axes = load_battery(probes_dir, battery)
    axis = next((a for a in axes if a.id == axis_id), None)
    if axis is None:
        raise click.ClickException(f"axis {axis_id!r} not found in battery {battery!r}")
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    judge_list = [j.strip() for j in judges.split(",") if j.strip()]
    budget = Budget(state_path=data_dir / "budget.json", cap_usd=cap)

    click.echo(f"running axis={axis_id} pairs={len(model_list) * (len(model_list) - 1)} reruns={reruns}")
    run_axis(axis, models=model_list, judges=judge_list, n_reruns=reruns, data_dir=data_dir, budget=budget)
    click.echo(f"done. spent ${budget.spent_usd:.4f} (cap ${cap:.2f})")


@main.command("status")
@click.option("--data-dir", default=str(DATA_DIR_DEFAULT), show_default=True, type=click.Path(path_type=Path))
def status_cmd(data_dir):
    """Print run status: number of transcripts and judgments on disk."""
    interviews = data_dir / "raw" / "interviews"
    judgments = data_dir / "raw" / "judgments"
    n_interviews = sum(1 for _ in interviews.rglob("*.json")) if interviews.exists() else 0
    n_judgments = sum(1 for _ in judgments.rglob("*.json")) if judgments.exists() else 0

    state = data_dir / "budget.json"
    spent = json.loads(state.read_text())["spent_usd"] if state.exists() else 0.0

    click.echo(f"interviews on disk: {n_interviews}")
    click.echo(f"judgments on disk:  {n_judgments}")
    click.echo(f"budget spent:       ${spent:.4f}")


@main.command("cost")
@click.option("--data-dir", default=str(DATA_DIR_DEFAULT), show_default=True, type=click.Path(path_type=Path))
def cost_cmd(data_dir):
    """Print current spend."""
    state = data_dir / "budget.json"
    spent = json.loads(state.read_text())["spent_usd"] if state.exists() else 0.0
    click.echo(f"{spent:.4f}")


if __name__ == "__main__":
    main()
