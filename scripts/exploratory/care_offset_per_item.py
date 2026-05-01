"""Care-offset decomposition per MFQ-2 item (exploratory; NOT pre-registered).

H1 found that LLMs match Atari 2023's foundation rank order exactly but with a
systematic offset — LLMs sit BELOW human means on every foundation EXCEPT Care,
where they sit ABOVE. This script asks: which specific Care items drive the offset?

Atari 2023 reports per-foundation means but not (in Table 7) per-item means, so we
work with two reference points:
  (a) the Atari foundation-level mean (3.96 for Care, ~3.7 for Loyalty/Authority,
      etc.) — used as the per-foundation baseline.
  (b) the LLM grand-mean per axis — what the 12 LLMs say collectively on each item.

We rank all 36 MFQ-2 items by LLM-grand-mean and surface:
  - Top 5 most strongly endorsed items (by LLM consensus)
  - Bottom 5 least endorsed
  - Per-item delta from the Atari foundation mean (LLM_item_mean - Atari_foundation_mean)
  - Display the verbatim Atari item text alongside, to inspect language patterns
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

FOUNDATIONS = ["care", "equality", "proportionality", "loyalty", "authority", "purity"]
FOUNDATION_COLORS = {
    "care": "#ef4444", "equality": "#f59e0b", "proportionality": "#eab308",
    "loyalty": "#10b981", "authority": "#3b82f6", "purity": "#8b5cf6",
}


def foundation_for(axis_id: str) -> str | None:
    parts = axis_id.split("_")
    return parts[1] if len(parts) >= 3 and parts[0] == "mfq2" and parts[1] in FOUNDATIONS else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--probes-dir", default="probes")
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()

    derived = Path(args.data_dir) / "derived"
    figures = Path(args.figures_dir)
    figures.mkdir(parents=True, exist_ok=True)

    # Load LLM verdicts
    df = pd.read_csv(derived / "verdicts_long.csv")
    df = df[df["verdict_type"] == "scale"].copy()
    df["scale_value"] = pd.to_numeric(df["scale_value"], errors="coerce")
    df = df.dropna(subset=["scale_value"])
    df["foundation"] = df["axis_id"].apply(foundation_for)
    df = df[df["foundation"].notna()].copy()

    # LLM mean per item across all interviewer reads × all interviewees
    item_means = df.groupby(["axis_id", "foundation"])["scale_value"].agg(
        ["count", "mean", "std"]
    ).reset_index().rename(columns={"count": "n_cells", "mean": "llm_grand_mean", "std": "llm_grand_sd"})

    # Load Atari foundation means
    atari = pd.read_csv(Path(args.data_dir) / "external" / "atari_2023_norms.csv")
    atari_dict = atari.set_index("foundation")["atari_mean"].to_dict()
    item_means["atari_foundation_mean"] = item_means["foundation"].map(atari_dict)
    item_means["delta_vs_atari"] = item_means["llm_grand_mean"] - item_means["atari_foundation_mean"]

    # Load probe text (the verbatim Atari item statements)
    probe_axes = json.loads((Path(args.probes_dir) / "mfq.json").read_text())
    desc = {a["id"]: a["description"] for a in probe_axes}
    item_means["item_text"] = item_means["axis_id"].map(desc)

    item_means = item_means.sort_values("llm_grand_mean", ascending=False).reset_index(drop=True)
    out_csv = derived / "explore_care_offset_per_item.csv"
    item_means.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"  wrote {out_csv}")

    # Top 5 / Bottom 5
    print("\n=== TOP 5 most-endorsed MFQ-2 items (LLM grand mean) ===")
    for _, r in item_means.head(5).iterrows():
        print(f"  [{r['foundation']:<15}] {r['axis_id']:<25} mean={r['llm_grand_mean']:.3f}  Δ={r['delta_vs_atari']:+.3f}  \"{r['item_text']}\"")
    print("\n=== BOTTOM 5 least-endorsed MFQ-2 items ===")
    for _, r in item_means.tail(5).iterrows():
        print(f"  [{r['foundation']:<15}] {r['axis_id']:<25} mean={r['llm_grand_mean']:.3f}  Δ={r['delta_vs_atari']:+.3f}  \"{r['item_text']}\"")

    # Care drill-down — within Care, which items show the biggest positive offset?
    care_items = item_means[item_means["foundation"] == "care"].sort_values("delta_vs_atari", ascending=False)
    print("\n=== CARE items, ranked by delta vs Atari foundation mean ===")
    for _, r in care_items.iterrows():
        print(f"  Δ={r['delta_vs_atari']:+.3f}  {r['axis_id']:<18} mean={r['llm_grand_mean']:.3f}  \"{r['item_text']}\"")

    # Cross-foundation: which items have the biggest POSITIVE offset (LLM > Atari)?
    most_positive = item_means.nlargest(7, "delta_vs_atari")
    most_negative = item_means.nsmallest(7, "delta_vs_atari")
    print("\n=== TOP 7 items where LLMs ENDORSE more than humans (Δ > 0) ===")
    for _, r in most_positive.iterrows():
        print(f"  Δ={r['delta_vs_atari']:+.3f}  [{r['foundation']:<15}] \"{r['item_text']}\"")
    print("\n=== TOP 7 items where LLMs ENDORSE less than humans (Δ < 0) ===")
    for _, r in most_negative.iterrows():
        print(f"  Δ={r['delta_vs_atari']:+.3f}  [{r['foundation']:<15}] \"{r['item_text']}\"")

    # Figure: per-item delta colored by foundation
    fig, ax = plt.subplots(figsize=(10, 12))
    sorted_items = item_means.sort_values("delta_vs_atari")
    colors = [FOUNDATION_COLORS[f] for f in sorted_items["foundation"]]
    ax.barh(sorted_items["axis_id"], sorted_items["delta_vs_atari"], color=colors, edgecolor="black", lw=0.3)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("LLM grand mean − Atari foundation mean (Likert points)\nNegative = LLMs endorse less than humans. Positive = LLMs endorse more.")
    ax.set_title("Per-item LLM-vs-Atari offset (color = foundation)")
    fig.tight_layout()
    fig.savefig(figures / "explore_care_offset_per_item.png", dpi=140)
    print(f"\n  wrote {figures / 'explore_care_offset_per_item.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
