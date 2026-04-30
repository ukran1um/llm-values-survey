"""H3 — hierarchical clustering of the 12 × 44 fingerprint matrix, colored by training region.

Method:
- Build the 12 × N matrix from data/derived/per_model_fingerprints.csv (scale axes only).
- Drop axes with NaN values (gemini-quota-affected coverage holes will leave some).
- Compute pairwise cosine distance between models.
- Hierarchical clustering with average linkage.
- Dendrogram with leaves colored by training region (US / China / EU).
- Bonus: distribution of pairwise distances within-region vs across-region.

Outputs:
  data/derived/h3_pairwise_distances.csv     12×12 cosine distance matrix
  data/derived/h3_within_vs_across_region.csv  per-pair distance + within-region flag
  figures/h3_dendrogram.png                  dendrogram with region-colored leaf labels
  figures/h3_within_across_distance_dist.png distribution of distances by within/across region

H3 thresholds (from PRE_REGISTRATION):
  Strong support:  median(within-region distance) significantly < median(across-region distance),
                   Mann-Whitney U p < 0.05.
  Sibling pair (Anthropic Opus vs Sonnet) distance < median cross-lab distance.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu

REGION_COLORS = {
    "US": "#3b82f6",      # blue
    "China": "#ef4444",   # red
    "EU": "#10b981",      # green
}

# Display labels (shortened)
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


def load_fingerprint_matrix(path: Path) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Returns (matrix, model_ids, regions, axis_ids). Drops axes with any NaN."""
    df = pd.read_csv(path)
    axis_cols = [c for c in df.columns if c not in ("model", "region")]
    # Convert to numeric, drop axes with any NaN (so we have a clean dense matrix for distance)
    for c in axis_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    keep_axes = [c for c in axis_cols if df[c].notna().all()]
    dropped = set(axis_cols) - set(keep_axes)
    if dropped:
        print(f"  dropping {len(dropped)} axes with NaN values: {sorted(dropped)[:5]}...")
    matrix = df[keep_axes].values
    return matrix, df["model"].tolist(), df["region"].tolist(), keep_axes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()

    derived = Path(args.data_dir) / "derived"
    figures = Path(args.figures_dir)
    figures.mkdir(parents=True, exist_ok=True)

    fp_path = derived / "per_model_fingerprints.csv"
    if not fp_path.exists():
        print(f"ERROR: run scripts/aggregate.py first (no {fp_path})", file=sys.stderr)
        return 1

    matrix, models, regions, axes_used = load_fingerprint_matrix(fp_path)
    print(f"  matrix shape: {matrix.shape} ({len(models)} models × {len(axes_used)} clean axes)")

    # Pairwise cosine distance
    dist_vec = pdist(matrix, metric="cosine")
    dist_mat = squareform(dist_vec)
    pd.DataFrame(dist_mat, index=models, columns=models).to_csv(
        derived / "h3_pairwise_distances.csv", float_format="%.4f"
    )
    print(f"  wrote {derived / 'h3_pairwise_distances.csv'}")

    # Within- vs across-region pairs
    rows = []
    for i, j in combinations(range(len(models)), 2):
        within = (regions[i] == regions[j])
        rows.append({
            "model_a": models[i], "region_a": regions[i],
            "model_b": models[j], "region_b": regions[j],
            "within_region": int(within),
            "cosine_distance": dist_mat[i, j],
        })
    pairs = pd.DataFrame(rows).sort_values("cosine_distance")
    pairs.to_csv(derived / "h3_within_vs_across_region.csv", index=False, float_format="%.4f")
    print(f"  wrote {derived / 'h3_within_vs_across_region.csv'}")

    # Statistical test: median within vs median across
    within_d = pairs[pairs["within_region"] == 1]["cosine_distance"].values
    across_d = pairs[pairs["within_region"] == 0]["cosine_distance"].values
    print()
    print(f"  median within-region distance:  {np.median(within_d):.4f}  (n={len(within_d)} pairs)")
    print(f"  median across-region distance:  {np.median(across_d):.4f}  (n={len(across_d)} pairs)")
    if len(within_d) >= 1 and len(across_d) >= 1:
        u, p = mannwhitneyu(within_d, across_d, alternative="less")
        print(f"  Mann-Whitney U (within < across): U={u:.0f}, p={p:.4f}")
    sibling_pair = pairs[
        (pairs["model_a"].isin(["claude-opus-4-7", "claude-sonnet-4-6"]))
        & (pairs["model_b"].isin(["claude-opus-4-7", "claude-sonnet-4-6"]))
    ]
    if not sibling_pair.empty:
        sib_d = sibling_pair["cosine_distance"].iloc[0]
        cross_lab = pairs[pairs["within_region"] == 0]["cosine_distance"]
        print(f"  Anthropic sibling pair distance: {sib_d:.4f}")
        print(f"  median cross-lab distance:       {cross_lab.median():.4f}")
        print(f"  sibling < median cross-lab? {sib_d < cross_lab.median()}")

    # --- Dendrogram ---
    Z = linkage(dist_vec, method="average")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    short_labels = [SHORT_NAMES.get(m, m) for m in models]
    dendro = dendrogram(
        Z,
        labels=short_labels,
        leaf_rotation=45,
        leaf_font_size=10,
        color_threshold=0,
        above_threshold_color="#888",
        ax=ax,
    )
    # Color the leaf labels by region
    leaf_order = dendro["leaves"]
    for idx, leaf_idx in enumerate(leaf_order):
        region = regions[leaf_idx]
        ax.get_xticklabels()[idx].set_color(REGION_COLORS.get(region, "black"))
    ax.set_ylabel("Cosine distance")
    ax.set_title(
        f"H3 — Hierarchical clustering of LLMs by 44-axis fingerprint (cosine, average linkage)\n"
        f"Leaf color = training region. Within-region median = {np.median(within_d):.3f}, "
        f"across = {np.median(across_d):.3f}"
    )
    # Region legend
    handles = [plt.Line2D([], [], marker="s", color="white", markerfacecolor=c, markersize=12, label=r)
               for r, c in REGION_COLORS.items()]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "h3_dendrogram.png", dpi=140)
    print(f"  wrote {figures / 'h3_dendrogram.png'}")

    # --- Within vs across distribution ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, max(pairs["cosine_distance"].max(), 0.1), 25)
    ax.hist(across_d, bins=bins, alpha=0.6, color="#888", label=f"across-region (n={len(across_d)})", edgecolor="k", lw=0.5)
    ax.hist(within_d, bins=bins, alpha=0.7, color="#3b82f6", label=f"within-region (n={len(within_d)})", edgecolor="k", lw=0.5)
    ax.axvline(np.median(within_d), ls="--", color="#3b82f6", label=f"within median = {np.median(within_d):.3f}")
    ax.axvline(np.median(across_d), ls="--", color="#444", label=f"across median = {np.median(across_d):.3f}")
    ax.set_xlabel("Cosine distance between model pairs")
    ax.set_ylabel("Pair count")
    ax.set_title("H3 — Pairwise distance distribution: within-region vs across-region")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "h3_within_across_distance_dist.png", dpi=140)
    print(f"  wrote {figures / 'h3_within_across_distance_dist.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
