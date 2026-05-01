"""Key-quote mining (exploratory; NOT pre-registered).

Each verdict pulls a verbatim short quote that the interviewer cites as the basis for
its score. With 11 interviewers reading each interviewee on each axis, we can ask:

  - CONSENSUS QUOTES: which (interviewee, axis) cells have multiple interviewers
    extracting near-identical short passages? Those are the "moment of truth" quotes —
    where everyone agrees something revealing happened.

  - HIGH-DISAGREEMENT CELLS: where do the 11 interviewers extract very different
    quotes from the same interviewee answer? Those are interviewer-personality cells.

Useful narrative content for the essay (and for the readgrounded.com explorer).
"""

from __future__ import annotations
import argparse, re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

STOPWORDS = set("""
a an the and or but if then so as is are was were be been being have has had do does did
not no this that these those of in on at to from by for with about into through during
above below between after before under over again further once i you he she it we they
their them his her its our your my me him us very just also so as more less most all
any some other another such only own same than too can will may might could should would
""".split())


def normalize(text: str) -> list[str]:
    """Tokenize a quote down to lowercase content words for similarity comparison."""
    if not isinstance(text, str):
        return []
    cleaned = re.sub(r"[^a-z0-9' ]", " ", text.lower())
    return [t for t in cleaned.split() if t and t not in STOPWORDS and len(t) > 2]


def overlap(a: list[str], b: list[str]) -> float:
    """Jaccard overlap between two normalized token lists."""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--top-consensus", type=int, default=20)
    ap.add_argument("--top-disagreement", type=int, default=15)
    args = ap.parse_args()

    derived = Path(args.data_dir) / "derived"
    df = pd.read_csv(derived / "verdicts_long.csv")
    df = df[df["rerun"] == 0].copy()  # main run only — consistent comparison set
    df = df[df["key_quote"].fillna("").str.len() > 5]
    df["tokens"] = df["key_quote"].apply(normalize)

    # Group: same (interviewee, axis) — the 11 different reads
    cell_groups = defaultdict(list)
    for _, row in df.iterrows():
        cell_groups[(row["interviewee"], row["axis_id"], row["battery"])].append({
            "interviewer": row["interviewer"],
            "scale_value": row.get("scale_value"),
            "categorical_choice": row.get("categorical_choice"),
            "confidence": row.get("confidence"),
            "key_quote": row["key_quote"],
            "tokens": row["tokens"],
        })

    # CONSENSUS: find cells where most interviewers extracted overlapping passages.
    # Score = mean pairwise Jaccard among the cell's quotes; needs >= 4 interviewers.
    consensus_rows = []
    for (interviewee, axis_id, battery), reads in cell_groups.items():
        if len(reads) < 4:
            continue
        pair_scores = []
        for i in range(len(reads)):
            for j in range(i + 1, len(reads)):
                pair_scores.append(overlap(reads[i]["tokens"], reads[j]["tokens"]))
        if not pair_scores:
            continue
        mean_overlap = sum(pair_scores) / len(pair_scores)
        consensus_rows.append({
            "interviewee": interviewee, "axis_id": axis_id, "battery": battery,
            "n_reads": len(reads),
            "mean_pairwise_overlap": round(mean_overlap, 4),
            "sample_quotes": " || ".join(r["key_quote"][:100] for r in reads[:3]),
        })

    cdf = pd.DataFrame(consensus_rows).sort_values("mean_pairwise_overlap", ascending=False)
    cdf.head(args.top_consensus * 2).to_csv(derived / "explore_consensus_quotes.csv", index=False)
    print(f"  wrote data/derived/explore_consensus_quotes.csv ({len(cdf)} cells scored)")
    print()
    print(f"=== TOP {args.top_consensus} consensus cells (interviewers extract similar passages) ===")
    print(cdf.head(args.top_consensus)[["interviewee", "axis_id", "n_reads", "mean_pairwise_overlap"]].to_string(index=False))

    print()
    print(f"=== TOP 5 consensus cells, with sample quotes ===")
    for _, row in cdf.head(5).iterrows():
        print(f"\n  {row['interviewee']} on {row['axis_id']} (overlap={row['mean_pairwise_overlap']:.3f}):")
        cell_reads = cell_groups[(row["interviewee"], row["axis_id"], row["battery"])]
        for r in cell_reads[:5]:
            print(f"    [{r['interviewer'][:24]:<24}] \"{r['key_quote'][:140]}\"")

    # DISAGREEMENT: low overlap AND scale verdicts that span a wide range
    disagreement_rows = []
    for (interviewee, axis_id, battery), reads in cell_groups.items():
        if len(reads) < 4:
            continue
        scales = [r["scale_value"] for r in reads if pd.notna(r["scale_value"])]
        if len(scales) < 4:
            continue
        try:
            scales_f = [float(s) for s in scales]
        except (ValueError, TypeError):
            continue
        sd = pd.Series(scales_f).std()
        rng = max(scales_f) - min(scales_f)
        pair_scores = [overlap(reads[i]["tokens"], reads[j]["tokens"])
                       for i in range(len(reads)) for j in range(i + 1, len(reads))]
        mean_overlap = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
        # Disagreement metric: high SD on scale + low quote overlap
        disagreement_rows.append({
            "interviewee": interviewee, "axis_id": axis_id, "battery": battery,
            "n_reads": len(reads),
            "scale_sd": round(sd, 3),
            "scale_range": round(rng, 1),
            "mean_quote_overlap": round(mean_overlap, 4),
            "sample_quotes": " || ".join(r["key_quote"][:80] for r in reads[:3]),
        })

    ddf = pd.DataFrame(disagreement_rows)
    ddf["disagreement_score"] = ddf["scale_sd"] - 2.0 * ddf["mean_quote_overlap"]
    ddf = ddf.sort_values("disagreement_score", ascending=False)
    ddf.head(args.top_disagreement * 2).to_csv(derived / "explore_disagreement_quotes.csv", index=False)
    print(f"\n  wrote data/derived/explore_disagreement_quotes.csv ({len(ddf)} cells scored)")
    print()
    print(f"=== TOP {args.top_disagreement} most-contested cells (high scale-SD AND low quote overlap) ===")
    print(ddf.head(args.top_disagreement)[
        ["interviewee", "axis_id", "n_reads", "scale_sd", "scale_range", "mean_quote_overlap"]
    ].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
