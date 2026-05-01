"""H4 — self-report vs peer-interpretation divergence.

The most novel finding of the survey: do models say what their peers say they say?

Method:
- For each model, the SELF-REPORT fingerprint is the 36-axis Likert vector from Task 4.5
  (direct administration of verbatim Atari 2023 MFQ-2 items).
- The PEER-INTERPRETATION fingerprint is the 36-axis Likert vector where each axis is the
  mean across the 11 reading-interviewers. (We aggregate over interviewers to wash out the
  per-interviewer personality effect we identified separately.)
- For each model: Pearson r between self-report and peer-interpretation across the 36 items.
- Headline: distribution of those 12 r values, plus a per-model scatter showing the divergence.

Outputs:
  figures/h4_self_vs_peer_scatter.png      All 12 models × 36 axes points; color = model.
                                            Diagonal = perfect agreement.
  figures/h4_per_model_r_distribution.png  Bar chart of the 12 per-model Pearson r values.
  figures/h4_delta_heatmap.png             12 × 36 heatmap of (self - peer) deltas.

Pre-registered thresholds (from PRE_REG):
  Strong agreement:  mean per-model r ≥ 0.6 → self-report and peer-interpretation align.
  Strong divergence: mean per-model r < 0.3 → headline methodology paper finding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Color palette per model — try to make regions visually distinguishable
MODEL_COLORS = {
    "claude-opus-4-7": "#1d4ed8",
    "claude-sonnet-4-6": "#3b82f6",
    "gpt-5.5-2026-04-23": "#0d9488",
    "gemini-2.5-pro": "#10b981",
    "grok-4.20": "#84cc16",
    "meta-llama/llama-4-scout-17b-16e-instruct": "#eab308",
    "openai/gpt-oss-120b": "#f59e0b",
    "qwen/qwen3-32b": "#dc2626",
    "minimax-m2-7": "#ef4444",
    "mistralai/mistral-large-2411": "#a855f7",
    "deepseek/deepseek-chat": "#ec4899",
    "z-ai/glm-4.6": "#f43f5e",
}

SHORT_NAMES = {
    "claude-opus-4-7": "claude-opus",
    "claude-sonnet-4-6": "claude-sonnet",
    "gpt-5.5-2026-04-23": "gpt-5.5",
    "gemini-2.5-pro": "gemini-2.5",
    "grok-4.20": "grok-4.20",
    "meta-llama/llama-4-scout-17b-16e-instruct": "llama-4-scout",
    "openai/gpt-oss-120b": "gpt-oss-120b",
    "qwen/qwen3-32b": "qwen3-32b",
    "minimax-m2-7": "minimax-m2",
    "mistralai/mistral-large-2411": "mistral-large",
    "deepseek/deepseek-chat": "deepseek-chat",
    "z-ai/glm-4.6": "glm-4.6",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()

    derived = Path(args.data_dir) / "derived"
    figures = Path(args.figures_dir)
    figures.mkdir(parents=True, exist_ok=True)

    h4_path = derived / "h4_self_vs_peer_correlations.csv"
    if not h4_path.exists():
        print(f"ERROR: run scripts/aggregate.py first (no {h4_path})", file=sys.stderr)
        return 1

    df = pd.read_csv(h4_path)
    if len(df) == 0:
        print("ERROR: h4 CSV is empty — no self-reports yet", file=sys.stderr)
        return 1

    # Per-model summary rows
    summary = df[df["scope"] == "model_summary"].copy()
    summary["per_model_pearson_r"] = pd.to_numeric(summary["per_model_pearson_r"], errors="coerce")
    summary["per_model_n_axes"] = pd.to_numeric(summary["per_model_n_axes"], errors="coerce")
    summary_valid = summary.dropna(subset=["per_model_pearson_r"])

    # Per-axis detail rows
    detail = df[df["scope"] == "model_axis"].copy()
    for c in ("self_report", "peer_mean", "delta_self_minus_peer"):
        detail[c] = pd.to_numeric(detail[c], errors="coerce")
    detail = detail.dropna(subset=["self_report", "peer_mean"])

    print(f"  models with self-reports: {len(summary_valid)}")
    print(f"  per-model Pearson r distribution:")
    print(summary_valid[["model", "per_model_pearson_r", "per_model_n_axes"]].to_string(index=False))
    print(f"\n  mean across models:   r = {summary_valid['per_model_pearson_r'].mean():.4f}")
    print(f"  median across models: r = {summary_valid['per_model_pearson_r'].median():.4f}")

    # --- Figure 1: scatter of all (self, peer) points ---
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for model, sub in detail.groupby("model"):
        ax.scatter(
            sub["self_report"], sub["peer_mean"],
            color=MODEL_COLORS.get(model, "gray"),
            label=SHORT_NAMES.get(model, model),
            alpha=0.7, s=40, edgecolor="white", lw=0.5,
        )
    ax.plot([1, 5], [1, 5], ls="--", color="gray", alpha=0.5, label="self = peer")
    ax.set_xlabel("Self-report rating (Likert 1-5)")
    ax.set_ylabel("Peer-interpretation mean (mean across 11 interviewers)")
    ax.set_title(
        f"H4 — Self-report vs peer-interpretation across 36 MFQ-2 items × {len(summary_valid)} models\n"
        f"Mean per-model Pearson r = {summary_valid['per_model_pearson_r'].mean():.3f}"
    )
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=7, ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(figures / "h4_self_vs_peer_scatter.png", dpi=140)
    print(f"  wrote {figures / 'h4_self_vs_peer_scatter.png'}")

    # --- Figure 2: per-model r bar ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    s = summary_valid.sort_values("per_model_pearson_r")
    colors = [MODEL_COLORS.get(m, "gray") for m in s["model"]]
    ax.barh(
        [SHORT_NAMES.get(m, m) for m in s["model"]],
        s["per_model_pearson_r"],
        color=colors, edgecolor="black", lw=0.5,
    )
    ax.axvline(0.3, ls=":", color="red", alpha=0.6, label="weak (PRE_REG)")
    ax.axvline(0.6, ls=":", color="green", alpha=0.6, label="strong (PRE_REG)")
    ax.set_xlabel("Pearson r (self-report vs peer-interpretation, 36 MFQ-2 items)")
    ax.set_title("H4 — Per-model agreement between what a model says about itself and how peers read it")
    ax.set_xlim(min(0, s["per_model_pearson_r"].min() - 0.1), 1.0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures / "h4_per_model_r_distribution.png", dpi=140)
    print(f"  wrote {figures / 'h4_per_model_r_distribution.png'}")

    # --- Figure 3: delta heatmap (model × axis) ---
    pivot = detail.pivot_table(
        index="model", columns="axis_id", values="delta_self_minus_peer", aggfunc="mean"
    )
    # Order columns canonically (foundation grouped)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    pivot = pivot.reindex([m for m in summary_valid["model"]])

    fig, ax = plt.subplots(figsize=(14, max(4, 0.4 * len(pivot))))
    abs_max = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 1.0)
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-abs_max, vmax=abs_max)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([SHORT_NAMES.get(m, m) for m in pivot.index], fontsize=9)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=6)
    ax.set_title(
        "H4 — Per-(model, axis) delta = self-report minus peer-interpretation\n"
        "Red = self claims MORE than peers attribute. Blue = self claims LESS than peers attribute."
    )
    plt.colorbar(im, ax=ax, label="Δ (Likert points)", shrink=0.7)
    fig.tight_layout()
    fig.savefig(figures / "h4_delta_heatmap.png", dpi=140)
    print(f"  wrote {figures / 'h4_delta_heatmap.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
