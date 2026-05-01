"""H2 — cross-cultural asymmetric reading on the mirror battery.

For each of the 6 mirror axes (Schwartz/Hofstede/GLOBE-derived value dimensions),
do American-trained models read Chinese-trained models differently than Chinese-
trained models read each other?

Method:
  Group all scale verdicts on mirror_* axes by (interviewer_region, interviewee_region):
    - "US → CN" — US interviewer reads Chinese interviewee
    - "CN → CN" — Chinese interviewer reads Chinese interviewee
  For each mirror axis, compare these two distributions of scale_value:
    - Mean each
    - Cohen's d (pooled SD denominator)
  PRE_REG threshold: |d| ≥ 0.5 on ≥ 3 of 6 axes = evidence of asymmetric reading.

Bonus: also report the symmetric counter-asymmetry "CN → US" vs "US → US".

Outputs:
  data/derived/h2_mirror_asymmetry.csv      one row per axis × asymmetry-direction with d, means, ns.
  figures/h2_mirror_asymmetry.png           grouped bar chart per axis showing the 4 cell means.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MIRROR_AXES = (
    "mirror_individualism_vs_collectivism",
    "mirror_universalism_vs_particularism",
    "mirror_harmony_vs_mastery",
    "mirror_face_vs_directness",
    "mirror_present_vs_future_orientation",
    "mirror_sacred_vs_instrumental_nature",
)

# Comparison directions to report (interviewer_region → interviewee_region)
DIRECTIONS = (
    ("US", "CN", "China", "US"),  # ASYMMETRY 1: how US reads Chinese vs how Chinese read Chinese
    ("CN", "US", "China", "US"),  # ASYMMETRY 2: how Chinese read US vs how US read US (mirror direction)
)


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Pooled-SD Cohen's d. Returns NaN if either group has < 2 samples or zero variance."""
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = group_a.var(ddof=1)
    var_b = group_b.var(ddof=1)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled_var <= 0:
        return float("nan")
    return (group_a.mean() - group_b.mean()) / np.sqrt(pooled_var)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()

    derived = Path(args.data_dir) / "derived"
    figures = Path(args.figures_dir)
    figures.mkdir(parents=True, exist_ok=True)

    verdicts_path = derived / "verdicts_long.csv"
    if not verdicts_path.exists():
        print(f"ERROR: run scripts/aggregate.py first (no {verdicts_path})", file=sys.stderr)
        return 1

    df = pd.read_csv(verdicts_path)
    df = df[df["axis_id"].isin(MIRROR_AXES)].copy()
    df = df[df["verdict_type"] == "scale"].copy()
    df["scale_value"] = pd.to_numeric(df["scale_value"], errors="coerce")
    df = df.dropna(subset=["scale_value"])
    # Map to short region keys
    df["i_region"] = df["interviewer_region"].map({"US": "US", "China": "CN", "EU": "EU"})
    df["e_region"] = df["interviewee_region"].map({"US": "US", "China": "CN", "EU": "EU"})

    rows = []
    print("\nH2 — cross-cultural asymmetric reading on the mirror battery\n")
    print(f"{'axis':<42} {'compare':<14} {'mean_a':>7} {'mean_b':>7} {'n_a':>4} {'n_b':>4} {'d':>7}")
    print("-" * 88)

    for axis in MIRROR_AXES:
        sub = df[df["axis_id"] == axis]
        for i_a, e_a, e_b_label_a, e_b_label_b in DIRECTIONS:
            # Group A: i_region == i_a, e_region == e_a   (e.g. US → CN)
            # Group B: i_region == e_a (mirror), e_region == e_a   (e.g. CN → CN)
            # Translate the asymmetry pair clearly:
            #   (i_a="US", e_a="CN") → group A = US→CN, group B = CN→CN
            #   (i_a="CN", e_a="US") → group A = CN→US, group B = US→US
            ga = sub[(sub["i_region"] == i_a) & (sub["e_region"] == e_a)]["scale_value"].values
            gb = sub[(sub["i_region"] == e_a) & (sub["e_region"] == e_a)]["scale_value"].values
            d = cohens_d(ga, gb)
            label_a = f"{i_a}->{e_a}"
            label_b = f"{e_a}->{e_a}"
            compare = f"{label_a} vs {label_b}"
            print(f"{axis:<42} {compare:<14} {ga.mean():>7.3f} {gb.mean():>7.3f} {len(ga):>4} {len(gb):>4} {d:>+7.3f}")
            rows.append({
                "axis_id": axis,
                "compare_a": label_a,
                "compare_b": label_b,
                "mean_a": float(ga.mean()) if len(ga) else float("nan"),
                "mean_b": float(gb.mean()) if len(gb) else float("nan"),
                "sd_a": float(ga.std(ddof=1)) if len(ga) >= 2 else float("nan"),
                "sd_b": float(gb.std(ddof=1)) if len(gb) >= 2 else float("nan"),
                "n_a": int(len(ga)),
                "n_b": int(len(gb)),
                "cohens_d": float(d) if not np.isnan(d) else "",
            })

    out = pd.DataFrame(rows)
    out.to_csv(derived / "h2_mirror_asymmetry.csv", index=False, float_format="%.4f")
    print(f"\n  wrote {derived / 'h2_mirror_asymmetry.csv'}")

    # PRE_REG threshold check on the canonical asymmetry (US→CN vs CN→CN)
    canonical = out[(out["compare_a"] == "US->CN") & (out["compare_b"] == "CN->CN")].copy()
    canonical["cohens_d"] = pd.to_numeric(canonical["cohens_d"], errors="coerce")
    canonical = canonical.dropna(subset=["cohens_d"])
    n_significant = int((canonical["cohens_d"].abs() >= 0.5).sum())
    print(f"\nPRE_REG threshold (US→CN vs CN→CN canonical asymmetry):")
    print(f"  axes with |d| >= 0.5: {n_significant} of {len(canonical)} mirror axes")
    print(f"  H2 supported (>= 3 of 6)? {n_significant >= 3}")
    if n_significant >= 1:
        print(f"  Direction of the asymmetric reads:")
        for _, row in canonical.iterrows():
            d = row["cohens_d"]
            if abs(d) >= 0.5:
                direction = "US reads CN HIGHER" if d > 0 else "US reads CN LOWER"
                print(f"    {row['axis_id']}: d = {d:+.3f}  ({direction} than CN reads CN)")

    # --- Figure: grouped bars per axis showing all 4 region cells ---
    fig, axes_grid = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    axes_flat = axes_grid.flatten()
    cells = [("US", "US"), ("US", "CN"), ("CN", "US"), ("CN", "CN")]
    cell_labels = ["US→US", "US→CN", "CN→US", "CN→CN"]
    cell_colors = ["#3b82f6", "#8b5cf6", "#f59e0b", "#ef4444"]

    for i, axis in enumerate(MIRROR_AXES):
        ax = axes_flat[i]
        sub = df[df["axis_id"] == axis]
        means = []
        sems = []
        for ir, er in cells:
            vals = sub[(sub["i_region"] == ir) & (sub["e_region"] == er)]["scale_value"].values
            if len(vals):
                means.append(vals.mean())
                sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) >= 2 else 0.0)
            else:
                means.append(float("nan"))
                sems.append(0.0)
        ax.bar(cell_labels, means, yerr=sems, color=cell_colors, edgecolor="black", lw=0.5, capsize=3)
        ax.set_ylim(1, 5)
        ax.axhline(3.0, ls=":", color="gray", lw=0.5)
        ax.set_title(axis.replace("mirror_", "").replace("_", " "), fontsize=10)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(
        "H2 — Region cell means on the 6 mirror axes\n"
        f"PRE_REG: US→CN vs CN→CN |d| ≥ 0.5 on ≥ 3 of 6 axes  →  observed: {n_significant} of {len(canonical)}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(figures / "h2_mirror_asymmetry.png", dpi=140)
    print(f"  wrote {figures / 'h2_mirror_asymmetry.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
