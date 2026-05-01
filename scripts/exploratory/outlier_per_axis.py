"""Per-axis outlier-model analysis (exploratory; NOT pre-registered).

For each scale axis: compute consensus mean across the 12 models (each model's mean
verdict as interviewee, averaged over all 11 interviewers). Then for each (model, axis)
compute distance from consensus. The model with the largest |distance| on that axis is
the outlier — the LLM that other LLMs read most differently from the rest.

Aggregate over axes: which models are most often the outlier, and on which axis types?
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

V1_ROSTER = [
    "claude-opus-4-7", "claude-sonnet-4-6", "gpt-5.5-2026-04-23", "gemini-2.5-pro",
    "grok-4.20", "meta-llama/llama-4-scout-17b-16e-instruct", "openai/gpt-oss-120b",
    "qwen/qwen3-32b", "minimax-m2-7", "mistralai/mistral-large-2411",
    "deepseek/deepseek-chat", "z-ai/glm-4.6",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()

    derived = Path(args.data_dir) / "derived"
    figures = Path(args.figures_dir)
    figures.mkdir(parents=True, exist_ok=True)

    fp = pd.read_csv(derived / "per_model_fingerprints_long.csv")
    fp = fp[fp["verdict_type"] == "scale"].copy()
    fp["mean_scale"] = pd.to_numeric(fp["mean_scale"], errors="coerce")
    fp = fp.dropna(subset=["mean_scale"])

    # consensus mean per axis (across the 12 models)
    consensus = fp.groupby("axis_id")["mean_scale"].agg(["mean", "std", "count"]).reset_index()
    consensus = consensus.rename(columns={"mean": "consensus_mean", "std": "consensus_sd", "count": "n_models"})
    fp = fp.merge(consensus, on="axis_id")
    fp["distance"] = fp["mean_scale"] - fp["consensus_mean"]
    fp["abs_distance"] = fp["distance"].abs()

    # outlier per axis
    idx = fp.groupby("axis_id")["abs_distance"].idxmax()
    outliers = fp.loc[idx, ["axis_id", "battery", "interviewee", "interviewee_region",
                             "mean_scale", "consensus_mean", "distance", "consensus_sd"]].rename(
        columns={"interviewee": "outlier_model", "interviewee_region": "outlier_region"}
    ).sort_values("axis_id")
    outliers["distance_in_sd"] = outliers["distance"] / outliers["consensus_sd"]

    out_csv = derived / "explore_axis_outliers.csv"
    outliers.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"  wrote {out_csv} ({len(outliers)} axes)")

    # Aggregate: how often is each model the outlier?
    counts = outliers.groupby(["outlier_model", "outlier_region"]).size().reset_index(name="n_axes_as_outlier")
    counts = counts.sort_values("n_axes_as_outlier", ascending=False)

    print("\nOutlier-frequency per model (how often this model is the most-distant from consensus):")
    print(counts.to_string(index=False))

    # By battery breakdown
    by_battery = outliers.groupby(["outlier_model", "battery"]).size().unstack(fill_value=0)
    by_battery = by_battery.reindex(V1_ROSTER, fill_value=0)
    print("\nOutlier counts per model × battery:")
    print(by_battery)

    # Direction analysis: does the outlier read HIGHER or LOWER than consensus?
    direction = outliers.groupby(["outlier_model"]).agg(
        n_above=("distance", lambda x: (x > 0).sum()),
        n_below=("distance", lambda x: (x < 0).sum()),
        max_above=("distance", "max"),
        min_below=("distance", "min"),
    )
    direction["net_above_minus_below"] = direction["n_above"] - direction["n_below"]
    direction = direction.sort_values("net_above_minus_below", ascending=False)
    print("\nDirection — when this model is the outlier, does it read HIGHER or LOWER than consensus?")
    print(direction)

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    counts_indexed = counts.set_index("outlier_model")
    counts_full = counts_indexed.reindex(V1_ROSTER)
    counts_full["n_axes_as_outlier"] = counts_full["n_axes_as_outlier"].fillna(0).astype(int)
    counts_full["outlier_region"] = counts_full["outlier_region"].fillna("")
    counts_full = counts_full.sort_values("n_axes_as_outlier")
    ax1.barh(counts_full.index, counts_full["n_axes_as_outlier"], color="#3b82f6", edgecolor="black", lw=0.5)
    ax1.set_xlabel("Number of axes where this model is the outlier")
    ax1.set_title(f"Frequency as outlier ({len(outliers)} scale axes)")

    bb = by_battery.loc[counts_full.index].copy()
    bb.plot.barh(stacked=True, ax=ax2, color={"mfq": "#3b82f6", "mirror": "#10b981", "extension": "#f59e0b", "philosophy": "#8b5cf6"})
    ax2.set_xlabel("Outlier counts by battery (stacked)")
    ax2.set_title("Outlier breakdown by battery")
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "explore_outlier_per_axis.png", dpi=140)
    print(f"\n  wrote {figures / 'explore_outlier_per_axis.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
