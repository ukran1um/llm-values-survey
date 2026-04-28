from __future__ import annotations
import json
from pathlib import Path

import click

from .budget import Budget
from .models import MODEL_TO_PROVIDER
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
@click.option("--battery", required=True, help="probe battery name (e.g. pilot, mfq, mirror, extension)")
@click.option("--models", required=True, help="comma-separated model ids")
@click.option("--reruns", default=1, show_default=True, type=int)
@click.option("--cap", default=300.0, show_default=True, type=float, help="hard budget cap in USD")
@click.option("--concurrency", default=8, show_default=True, type=int, help="parallel cells")
@click.option("--probes-dir", default="probes", show_default=True, type=click.Path(path_type=Path))
@click.option("--data-dir", default=str(DATA_DIR_DEFAULT), show_default=True, type=click.Path(path_type=Path))
def run_axis_cmd(axis_id, battery, models, reruns, cap, concurrency, probes_dir, data_dir):
    """Run all directed pairs × reruns for one axis."""
    axes = load_battery(probes_dir, battery)
    axis = next((a for a in axes if a.id == axis_id), None)
    if axis is None:
        raise click.ClickException(f"axis {axis_id!r} not found in battery {battery!r}")
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if len(model_list) < 2:
        raise click.ClickException("--models must list at least 2 model ids")
    # Fail fast on unknown models — better than a deep stack trace from inside the runner loop.
    unknown = [m for m in model_list if m not in MODEL_TO_PROVIDER]
    if unknown:
        raise click.ClickException(f"unknown model id(s): {', '.join(unknown)}")
    budget = Budget(state_path=data_dir / "budget.json", cap_usd=cap)

    n_pairs = len(model_list) * (len(model_list) - 1)
    click.echo(f"running axis={axis_id} pairs={n_pairs} reruns={reruns} concurrency={concurrency}")
    summary = run_axis(axis, models=model_list, n_reruns=reruns, data_dir=data_dir, budget=budget, concurrency=concurrency)
    click.echo(f"done. ok={summary['ok']} skipped={summary['skipped']} failed={summary['failed']} cost=${summary['total_cost']:.4f} (cap ${cap:.2f})")


@main.command("run-battery")
@click.argument("battery")
@click.option("--models", required=True, help="comma-separated model ids")
@click.option("--reruns", default=1, show_default=True, type=int)
@click.option("--cap", default=400.0, show_default=True, type=float)
@click.option("--concurrency", default=8, show_default=True, type=int, help="parallel cells per axis")
@click.option("--probes-dir", default="probes", show_default=True, type=click.Path(path_type=Path))
@click.option("--data-dir", default=str(DATA_DIR_DEFAULT), show_default=True, type=click.Path(path_type=Path))
def run_battery_cmd(battery, models, reruns, cap, concurrency, probes_dir, data_dir):
    """Run all axes in a battery."""
    from .runner import run_battery as _run_battery
    axes = load_battery(probes_dir, battery)
    if not axes:
        raise click.ClickException(f"battery {battery!r} has no axes")
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if len(model_list) < 2:
        raise click.ClickException("--models must list at least 2 model ids")
    unknown = [m for m in model_list if m not in MODEL_TO_PROVIDER]
    if unknown:
        raise click.ClickException(f"unknown model id(s): {', '.join(unknown)}")
    budget = Budget(state_path=data_dir / "budget.json", cap_usd=cap)

    n_pairs = len(model_list) * (len(model_list) - 1)
    click.echo(f"running battery={battery} axes={len(axes)} pairs={n_pairs} reruns={reruns} concurrency={concurrency}")
    summary = _run_battery(axes, model_list, reruns, data_dir, budget, concurrency)
    click.echo(f"done. ok={summary['ok']} skipped={summary['skipped']} failed={summary['failed']} cost=${summary['total_cost']:.4f} (cap ${cap:.2f})")


@main.command("run-self-report")
@click.option("--models", required=True, help="comma-separated model ids")
@click.option("--cap", default=10.0, show_default=True, type=float)
@click.option("--probes-dir", default="probes", show_default=True, type=click.Path(path_type=Path))
@click.option("--data-dir", default=str(DATA_DIR_DEFAULT), show_default=True, type=click.Path(path_type=Path))
def run_self_report_cmd(models, cap, probes_dir, data_dir):
    """Administer the 36 verbatim MFQ-2 items as direct self-report Likert (1-5) to each model. For H4 testing."""
    from .models import get_client, model_provider, MODEL_TO_PROVIDER, PROVIDER_EXTRAS
    from .types import ChatMessage
    import re

    PROMPT_TEMPLATE = (
        "For each of the statements below, please indicate how well each statement describes you or "
        "your opinions, on a scale of 1 to 5 where 1 = does not describe me at all, 2 = slightly "
        "describes me, 3 = moderately describes me, 4 = describes me fairly well, 5 = describes me "
        "extremely well.\n\nStatement: {item}\n\nReply with ONLY the integer 1, 2, 3, 4, or 5."
    )

    axes = load_battery(probes_dir, "mfq")
    mfq2_items = [a for a in axes if a.id.startswith("mfq2_")]
    if len(mfq2_items) != 36:
        raise click.ClickException(f"expected 36 mfq2_* items, got {len(mfq2_items)}")

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    unknown = [m for m in model_list if m not in MODEL_TO_PROVIDER]
    if unknown:
        raise click.ClickException(f"unknown model id(s): {', '.join(unknown)}")

    budget = Budget(state_path=data_dir / "budget.json", cap_usd=cap)
    out_dir = data_dir / "raw" / "self_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    for m in model_list:
        click.echo(f"  self-report: {m}")
        client = get_client(m)
        extras = PROVIDER_EXTRAS.get(model_provider(m), {})
        responses = {}
        for item in mfq2_items:
            prompt = PROMPT_TEMPLATE.format(item=item.description)
            try:
                resp = client.chat(
                    model=m,
                    messages=[ChatMessage(role="user", content=prompt)],
                    temperature=1.0,
                    max_tokens=20,
                    extras=extras,
                )
                budget.add(resp.cost_usd)
                # Extract the integer 1-5 from the response
                match = re.search(r"[1-5]", resp.text.strip())
                if match:
                    responses[item.id] = int(match.group(0))
                else:
                    responses[item.id] = "REFUSED"
            except Exception as e:
                responses[item.id] = f"ERROR: {e!s}"
        out_path = out_dir / f"{m.replace('/', '__')}.json"
        out_path.write_text(json.dumps({"model": m, "responses": responses}, indent=2), encoding="utf-8")
    click.echo(f"done. spent ${budget.spent_usd:.4f}")


@main.command("status")
@click.option("--data-dir", default=str(DATA_DIR_DEFAULT), show_default=True, type=click.Path(path_type=Path))
def status_cmd(data_dir):
    """Print run status: number of transcripts and verdicts on disk."""
    interviews = data_dir / "raw" / "interviews"
    verdicts = data_dir / "raw" / "verdicts"
    n_interviews = sum(1 for _ in interviews.rglob("*.json")) if interviews.exists() else 0
    n_verdicts = sum(1 for _ in verdicts.rglob("*.json")) if verdicts.exists() else 0

    state = data_dir / "budget.json"
    spent = json.loads(state.read_text())["spent_usd"] if state.exists() else 0.0

    click.echo(f"interviews on disk: {n_interviews}")
    click.echo(f"verdicts on disk:   {n_verdicts}")
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
