"""Per-axis consensus and disagreement (exploratory; NOT pre-registered).

For each scale axis, compute the SD of all 132 cells (across all interviewer-interviewee
permutations). High SD = the axis is contested, LLMs read each other inconsistently;
low SD = consensus axis, LLMs broadly agree.

The contested axes are where between-model variation lives; the consensus axes are
boring uniformity. Use this to pick which axes are worth spotlighting in the essay.
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()

    derived = Path(args.data_dir) / "derived"
    figures = Path(args.figures_dir)
    figures.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(derived / "verdicts_long.csv")
    df = df[df["verdict_type"] == "scale"].copy()
    df["scale_value"] = pd.to_numeric(df["scale_value"], errors="coerce")
    df = df.dropna(subset=["scale_value"])

    summary = (
        df.groupby(["axis_id", "battery"])["scale_value"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n_cells", "mean": "axis_mean", "std": "axis_sd"})
    )
    summary["range"] = summary["max"] - summary["min"]
    summary = summary.sort_values("axis_sd", ascending=False)

    out_csv = derived / "explore_axis_consensus.csv"
    summary.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"  wrote {out_csv}")

    # Top contested + top consensus
    print("\nTop 10 MOST CONTESTED axes (highest SD across 132 cells):")
    print(summary.head(10)[["axis_id", "battery", "n_cells", "axis_mean", "axis_sd"]].to_string(index=False))
    print("\nTop 10 MOST CONSENSUS axes (lowest SD):")
    print(summary.tail(10)[["axis_id", "battery", "n_cells", "axis_mean", "axis_sd"]].to_string(index=False))
    print("\nBy battery — mean SD per battery (axis-level disagreement, averaged within battery):")
    print(summary.groupby("battery")["axis_sd"].agg(["count", "mean", "std"]).round(3))

    # Figure: every axis bar-charted by SD, colored by battery
    fig, ax = plt.subplots(figsize=(10, 12))
    s = summary.sort_values("axis_sd")
    ax.barh(s["axis_id"], s["axis_sd"], color=s["battery"].map({
        "mfq": "#3b82f6", "mirror": "#10b981", "extension": "#f59e0b", "philosophy": "#8b5cf6",
    }), edgecolor="black", lw=0.3)
    ax.set_xlabel("SD of scale_value across all cells (1-5 Likert range)")
    ax.set_title("Per-axis disagreement spectrum (right = contested, left = consensus)")
    fig.tight_layout()
    fig.savefig(figures / "explore_axis_consensus.png", dpi=140)
    print(f"  wrote {figures / 'explore_axis_consensus.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
