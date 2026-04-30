"""H1 — validate peer-interview MFQ-2 means against published Atari 2023 norms.

Output structure (saved to data/derived/ and figures/):

  data/derived/h1_per_model_per_foundation.csv
      One row per (model, foundation) — mean across 6 items × 11 interviewers.
      This is what each model "looks like" through the eyes of the other 11.

  data/derived/h1_aggregate_per_foundation.csv
      One row per foundation — mean and SD across the 12 models.
      This is the row that goes into the H1 scatter as one point per foundation.

  figures/h1_foundation_distributions.png
      Boxplot of per-model means by foundation. Sanity check for the canonical
      MFQ pattern (Care/Equality typically high in WEIRD samples; Purity low).
      Doesn't claim a Pearson r yet — just shows the shape.

  figures/h1_atari_scatter.png   (only if data/external/atari_2023_norms.csv exists)
      Six points (one per foundation): x = Atari 2023 published mean,
      y = our 12-model mean. Pearson r reported on the figure.

To produce the scatter, drop a CSV at data/external/atari_2023_norms.csv with
columns: foundation, atari_mean, atari_sd, n_populations. Foundation labels
must match {care, equality, proportionality, loyalty, authority, purity}.
The Atari et al. 2023 (JPSP, doi 10.1037/pspp0000470) supplementary tables have
the per-foundation means averaged across 19 populations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import pandas as pd

V1_ROSTER = [
    "claude-opus-4-7", "claude-sonnet-4-6", "gpt-5.5-2026-04-23", "gemini-2.5-pro",
    "grok-4.20", "meta-llama/llama-4-scout-17b-16e-instruct", "openai/gpt-oss-120b",
    "qwen/qwen3-32b", "minimax-m2-7", "mistralai/mistral-large-2411",
    "deepseek/deepseek-chat", "z-ai/glm-4.6",
]
FOUNDATIONS = ["care", "equality", "proportionality", "loyalty", "authority", "purity"]
FOUNDATION_COLORS = {
    "care": "#ef4444",
    "equality": "#f59e0b",
    "proportionality": "#eab308",
    "loyalty": "#10b981",
    "authority": "#3b82f6",
    "purity": "#8b5cf6",
}


def foundation_for_axis(axis_id: str) -> str | None:
    # mfq2_<foundation>_<NN> per the probes file convention
    if not axis_id.startswith("mfq2_"):
        return None
    parts = axis_id.split("_")
    if len(parts) < 3:
        return None
    return parts[1] if parts[1] in FOUNDATIONS else None


def compute_per_model_per_foundation(verdicts: pd.DataFrame) -> pd.DataFrame:
    """For each model (as interviewee) × foundation, mean across 6 items × all interviewers."""
    scale = verdicts[verdicts["verdict_type"] == "scale"].copy()
    scale["scale_value"] = pd.to_numeric(scale["scale_value"], errors="coerce")
    scale["foundation"] = scale["axis_id"].apply(foundation_for_axis)
    scale = scale.dropna(subset=["foundation", "scale_value"])

    agg = (
        scale.groupby(["interviewee", "foundation"])["scale_value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"interviewee": "model"})
    )
    return agg


def plot_foundation_distributions(per_model: pd.DataFrame, out_path: Path) -> None:
    """Boxplot of per-model means by foundation, ordered by canonical Atari ranking."""
    fig, ax = plt.subplots(figsize=(9, 5))
    data_by_foundation = []
    for foundation in FOUNDATIONS:
        vals = per_model[per_model["foundation"] == foundation]["mean"].dropna().tolist()
        data_by_foundation.append(vals)

    bp = ax.boxplot(
        data_by_foundation,
        labels=FOUNDATIONS,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, foundation in zip(bp["boxes"], FOUNDATIONS):
        patch.set_facecolor(FOUNDATION_COLORS[foundation])
        patch.set_alpha(0.7)

    # Overlay individual model points
    for i, foundation in enumerate(FOUNDATIONS):
        sub = per_model[per_model["foundation"] == foundation]
        x = [i + 1] * len(sub)
        ax.scatter(x, sub["mean"], color="black", s=8, alpha=0.6, zorder=3)

    ax.set_ylabel("Per-model mean (averaged over 6 items × 11 interviewers)")
    ax.set_title(
        "MFQ-2 foundation distributions (peer-interview, n=12 models)\n"
        "Atari 2023 WEIRD pattern: Care ≈ Equality > Proportionality ≈ Loyalty > Authority > Purity"
    )
    ax.set_ylim(1, 5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"  wrote {out_path}")


def plot_atari_scatter(
    aggregate: pd.DataFrame,
    atari: pd.DataFrame,
    out_path: Path,
) -> None:
    """Per-foundation scatter: x=Atari mean, y=our mean. Pearson r reported."""
    merged = aggregate.merge(atari, on="foundation", validate="one_to_one")
    if len(merged) < 3:
        print(f"  skipping H1 scatter (only {len(merged)} foundations matched in atari CSV)")
        return

    r = merged[["our_mean", "atari_mean"]].corr().iloc[0, 1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for _, row in merged.iterrows():
        ax.scatter(
            row["atari_mean"], row["our_mean"],
            color=FOUNDATION_COLORS[row["foundation"]],
            s=160, edgecolor="black", lw=1.0, zorder=3,
        )
        ax.annotate(
            row["foundation"],
            (row["atari_mean"], row["our_mean"]),
            xytext=(8, 8), textcoords="offset points", fontsize=10,
        )
    # Identity line
    lo = min(merged["atari_mean"].min(), merged["our_mean"].min()) - 0.2
    hi = max(merged["atari_mean"].max(), merged["our_mean"].max()) + 0.2
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", alpha=0.5, label="y = x")
    ax.set_xlabel("Atari 2023 published mean (19-population aggregate)")
    ax.set_ylabel("Peer-interview mean (12 LLMs, this study)")
    ax.set_title(f"H1 — MFQ-2 foundation validation\nPearson r = {r:.3f}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"  wrote {out_path}")
    print(f"  Pearson r = {r:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()

    derived = Path(args.data_dir) / "derived"
    external = Path(args.data_dir) / "external"
    figures = Path(args.figures_dir)
    figures.mkdir(parents=True, exist_ok=True)

    verdicts_path = derived / "verdicts_long.csv"
    if not verdicts_path.exists():
        print(f"ERROR: run scripts/aggregate.py first (no {verdicts_path})", file=sys.stderr)
        return 1

    verdicts = pd.read_csv(verdicts_path)
    per_model = compute_per_model_per_foundation(verdicts)

    # Per-model × per-foundation
    per_model_out = derived / "h1_per_model_per_foundation.csv"
    per_model.to_csv(per_model_out, index=False, float_format="%.4f")
    print(f"  wrote {per_model_out} ({len(per_model)} rows)")

    # Aggregate (one row per foundation)
    aggregate = (
        per_model.groupby("foundation")["mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "our_mean", "std": "our_sd", "count": "n_models"})
    )
    # canonical foundation order
    aggregate["foundation"] = pd.Categorical(aggregate["foundation"], categories=FOUNDATIONS, ordered=True)
    aggregate = aggregate.sort_values("foundation").reset_index(drop=True)
    agg_out = derived / "h1_aggregate_per_foundation.csv"
    aggregate.to_csv(agg_out, index=False, float_format="%.4f")
    print(f"  wrote {agg_out}")

    # Distribution plot (works without Atari)
    plot_foundation_distributions(per_model, figures / "h1_foundation_distributions.png")

    # H1 scatter — only if external Atari norms exist
    atari_path = external / "atari_2023_norms.csv"
    if atari_path.exists():
        atari = pd.read_csv(atari_path)
        atari["foundation"] = atari["foundation"].str.lower().str.strip()
        plot_atari_scatter(aggregate, atari, figures / "h1_atari_scatter.png")
    else:
        print()
        print(f"  NOTE: {atari_path} not found — H1 scatter not generated.")
        print(f"        Drop a CSV with columns [foundation, atari_mean, atari_sd, n_populations]")
        print(f"        from Atari 2023 (JPSP, doi 10.1037/pspp0000470) supplementary materials.")

    print()
    print("H1 outputs:")
    print(aggregate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
