"""Walk data/raw/ and emit derived CSVs to data/derived/.

Idempotent — runs against any snapshot. Re-running with new data overwrites outputs.
The point of this script is to make every downstream analysis (notebook, paper figures,
recovery-pass target selection) work against tabular CSVs rather than raw JSON, so the
analysis layer is decoupled from the storage layer.

Outputs to data/derived/:
  verdicts_long.csv               One row per (axis, interviewer, interviewee, rerun) cell.
  per_model_fingerprints_long.csv One row per (model, axis, role) with mean/n/refusals.
  per_model_fingerprints.csv      Wide matrix: rows=models (interviewees), cols=scale axes,
                                  values=mean scale_value. Drives H3 clustering.
  refusal_rates.csv               Per-(model, role) and per-(model, axis) refusal counts.
  cost_summary.csv                Per-(model, role) total cost and tokens.
  coverage_matrix.csv             Per-axis: cells present, expected, % coverage.
                                  Drives the recovery-pass target list.
  stability_variance.csv          Empty stub — populated when Task 5 (stability sub-study) runs.
  self_report_fingerprints.csv    Empty stub — populated when Task 4.5 (self-reports) runs.
  h4_self_vs_peer_correlations.csv Empty stub — derived from the above two.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

# --- Locked v1 metadata --------------------------------------------------------------------------

# Roster locked April 2026. Order is canonical for fingerprint columns.
V1_ROSTER: list[str] = [
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "gpt-5.5-2026-04-23",
    "gemini-2.5-pro",
    "grok-4.20",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "minimax-m2-7",
    "mistralai/mistral-large-2411",
    "deepseek/deepseek-chat",
    "z-ai/glm-4.6",
]

# Region label for H3 clustering. Derived from publicly-disclosed lab origin.
MODEL_REGION: dict[str, str] = {
    "claude-opus-4-7": "US",
    "claude-sonnet-4-6": "US",
    "gpt-5.5-2026-04-23": "US",
    "gemini-2.5-pro": "US",
    "grok-4.20": "US",
    "meta-llama/llama-4-scout-17b-16e-instruct": "US",
    "openai/gpt-oss-120b": "US",
    "qwen/qwen3-32b": "China",
    "minimax-m2-7": "China",
    "deepseek/deepseek-chat": "China",
    "z-ai/glm-4.6": "China",
    "mistralai/mistral-large-2411": "EU",
}

BATTERIES: list[str] = ["mfq", "mirror", "extension", "philosophy"]
N_RERUNS: int = 1  # Plan 03 spec; stability sub-study handled separately

# Categorical "refuses-to-engage" choice tag — present in some extension/philosophy axes.
REFUSAL_TAG = "refuses-to-engage"


# --- IO helpers ----------------------------------------------------------------------------------


def load_axes(probes_dir: Path) -> dict[str, dict]:
    axes: dict[str, dict] = {}
    for battery in BATTERIES:
        path = probes_dir / f"{battery}.json"
        if not path.exists():
            continue
        for ax in json.loads(path.read_text(encoding="utf-8")):
            axes[ax["id"]] = ax
    return axes


def safe_model_name(model: str) -> str:
    return model.replace("/", "__")


def parse_cell_filename(stem: str, roster: list[str]) -> tuple[str, str, int] | None:
    """Parse '<safe_i>__<safe_e>__r<n>' into (interviewer, interviewee, rerun).

    Several model IDs contain '/' which storage._safe rewrites to '__', so a split on '__'
    is ambiguous. We resolve by trying every position and accepting the unique split where
    both halves match a known model ID.
    """
    if "__r" not in stem:
        return None
    body, _, rerun_str = stem.rpartition("__r")
    try:
        rerun = int(rerun_str)
    except ValueError:
        return None
    safe_to_real = {safe_model_name(m): m for m in roster}
    parts = body.split("__")
    matches: list[tuple[str, str, int]] = []
    for i in range(1, len(parts)):
        left = "__".join(parts[:i])
        right = "__".join(parts[i:])
        if left in safe_to_real and right in safe_to_real:
            matches.append((safe_to_real[left], safe_to_real[right], rerun))
    if len(matches) == 1:
        return matches[0]
    return None  # ambiguous or no match


def walk_verdict_files(data_dir: Path) -> list[tuple[Path, dict]]:
    out: list[tuple[Path, dict]] = []
    vdir = data_dir / "raw" / "verdicts"
    if not vdir.exists():
        return out
    for axis_dir in sorted(vdir.iterdir()):
        if not axis_dir.is_dir():
            continue
        for vfile in sorted(axis_dir.glob("*.json")):
            try:
                out.append((vfile, json.loads(vfile.read_text(encoding="utf-8"))))
            except Exception as e:
                print(f"  skipping malformed verdict {vfile}: {e}", file=sys.stderr)
    return out


def walk_transcripts_indexed(data_dir: Path) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    idir = data_dir / "raw" / "interviews"
    if not idir.exists():
        return out
    for axis_dir in sorted(idir.iterdir()):
        if not axis_dir.is_dir():
            continue
        for tfile in sorted(axis_dir.glob("*.json")):
            try:
                t = json.loads(tfile.read_text(encoding="utf-8"))
                key = (t["axis_id"], t["interviewer"], t["interviewee"], t["rerun"])
                out[key] = t
            except Exception as e:
                print(f"  skipping malformed transcript {tfile}: {e}", file=sys.stderr)
    return out


# --- Output builders -----------------------------------------------------------------------------


def write_verdicts_long(
    verdicts: list[dict],
    transcripts: dict[tuple, dict],
    axes: dict[str, dict],
    out_path: Path,
) -> int:
    fields = [
        "axis_id", "battery",
        "interviewer", "interviewer_region",
        "interviewee", "interviewee_region",
        "rerun",
        "verdict_type", "binary_choice", "scale_value", "categorical_choice",
        "is_refusal",
        "confidence", "n_turns_used",
        "key_quote", "reasoning",
        "verdict_cost_usd", "interviewer_cost_usd", "interviewee_cost_usd", "total_cost_usd",
        "verdict_prompt_tokens", "verdict_completion_tokens", "verdict_thoughts_tokens",
        "stop_reason", "created_at", "methodology_commit",
    ]
    n_written = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for v in verdicts:
            ax = axes.get(v["axis_id"], {})
            t = transcripts.get(
                (v["axis_id"], v["interviewer"], v["interviewee"], v["rerun"]), {}
            )
            verdict_cost = float(v.get("cost_usd", 0.0))
            i_cost = float(t.get("interviewer_cost_usd", 0.0))
            e_cost = float(t.get("interviewee_cost_usd", 0.0))
            is_refusal = (v.get("categorical_choice") == REFUSAL_TAG) or (
                v.get("verdict_type") == "scale" and v.get("scale_value") is None
            )
            w.writerow({
                "axis_id": v["axis_id"],
                "battery": ax.get("battery", ""),
                "interviewer": v["interviewer"],
                "interviewer_region": MODEL_REGION.get(v["interviewer"], ""),
                "interviewee": v["interviewee"],
                "interviewee_region": MODEL_REGION.get(v["interviewee"], ""),
                "rerun": v["rerun"],
                "verdict_type": v.get("verdict_type", ""),
                "binary_choice": v.get("binary_choice") or "",
                "scale_value": v.get("scale_value") if v.get("scale_value") is not None else "",
                "categorical_choice": v.get("categorical_choice") or "",
                "is_refusal": int(is_refusal),
                "confidence": v.get("confidence", ""),
                "n_turns_used": v.get("n_turns_used", ""),
                "key_quote": (v.get("key_quote") or "").replace("\n", " ").strip(),
                "reasoning": (v.get("reasoning") or "").replace("\n", " ").strip(),
                "verdict_cost_usd": f"{verdict_cost:.6f}",
                "interviewer_cost_usd": f"{i_cost:.6f}",
                "interviewee_cost_usd": f"{e_cost:.6f}",
                "total_cost_usd": f"{verdict_cost + i_cost + e_cost:.6f}",
                "verdict_prompt_tokens": v.get("prompt_tokens", ""),
                "verdict_completion_tokens": v.get("completion_tokens", ""),
                "verdict_thoughts_tokens": v.get("thoughts_tokens", ""),
                "stop_reason": v.get("stop_reason", "") or "",
                "created_at": v.get("created_at", ""),
                "methodology_commit": v.get("methodology_commit", ""),
            })
            n_written += 1
    return n_written


def write_fingerprints(
    verdicts: list[dict],
    axes: dict[str, dict],
    out_long: Path,
    out_wide: Path,
) -> tuple[int, int]:
    """Long: one row per (model_as_interviewee, axis) summary. Wide: rows=models, cols=scale axes.

    Aggregates over interviewers (so the "fingerprint" of a model M is the consensus reading
    of M's stance averaged across the 11 other models who interviewed M).
    """
    # bucket by (interviewee, axis)
    bucket: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for v in verdicts:
        bucket[(v["interviewee"], v["axis_id"])].append(v)

    fields_long = [
        "interviewee", "interviewee_region", "axis_id", "battery", "verdict_type",
        "n_cells", "n_refusals", "mean_scale", "mode_choice", "agreement_pct",
    ]
    rows_long: list[dict] = []
    # Per-model dict for wide-form scale fingerprint
    wide: dict[str, dict[str, float | str]] = defaultdict(dict)
    scale_axes: list[str] = []

    for (model, axis_id), cells in sorted(bucket.items()):
        ax = axes.get(axis_id, {})
        vtype = (cells[0].get("verdict_type") or ax.get("verdict_format", {}).get("type", ""))
        n_refusals = sum(
            1 for c in cells
            if c.get("categorical_choice") == REFUSAL_TAG
            or (c.get("verdict_type") == "scale" and c.get("scale_value") is None)
        )
        n_cells = len(cells)
        mean_scale = ""
        mode_choice = ""
        agreement_pct = ""
        if vtype == "scale":
            scale_vals = [c["scale_value"] for c in cells if c.get("scale_value") is not None]
            if scale_vals:
                mean_scale = f"{sum(scale_vals) / len(scale_vals):.4f}"
                wide[model][axis_id] = float(mean_scale)
        elif vtype in ("categorical", "binary"):
            field = "categorical_choice" if vtype == "categorical" else "binary_choice"
            choices = [c.get(field) for c in cells if c.get(field)]
            if choices:
                ctr = Counter(choices)
                mode_choice, mode_n = ctr.most_common(1)[0]
                agreement_pct = f"{(mode_n / len(choices)) * 100:.1f}"
        rows_long.append({
            "interviewee": model,
            "interviewee_region": MODEL_REGION.get(model, ""),
            "axis_id": axis_id,
            "battery": ax.get("battery", ""),
            "verdict_type": vtype,
            "n_cells": n_cells,
            "n_refusals": n_refusals,
            "mean_scale": mean_scale,
            "mode_choice": mode_choice,
            "agreement_pct": agreement_pct,
        })

    with out_long.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields_long)
        w.writeheader()
        w.writerows(rows_long)

    # Wide form — only scale axes; rows in V1_ROSTER order; cols in alphabetical axis order
    scale_axes = sorted(
        ax["id"] for ax in axes.values()
        if ax.get("verdict_format", {}).get("type") == "scale"
    )
    with out_wide.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "region"] + scale_axes)
        for model in V1_ROSTER:
            row = [model, MODEL_REGION.get(model, "")]
            for axis_id in scale_axes:
                v = wide.get(model, {}).get(axis_id, "")
                row.append(f"{v:.4f}" if isinstance(v, float) else "")
            w.writerow(row)

    return len(rows_long), len(scale_axes)


def write_refusal_rates(verdicts: list[dict], out_path: Path) -> None:
    # Per (interviewee, axis) refusal count from BOTH categorical refusal and scale_value=null
    by_axis: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"n": 0, "n_refusals": 0})
    by_model: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "n_refusals": 0})
    for v in verdicts:
        is_refusal = (v.get("categorical_choice") == REFUSAL_TAG) or (
            v.get("verdict_type") == "scale" and v.get("scale_value") is None
        )
        by_axis[(v["interviewee"], v["axis_id"])]["n"] += 1
        by_model[v["interviewee"]]["n"] += 1
        if is_refusal:
            by_axis[(v["interviewee"], v["axis_id"])]["n_refusals"] += 1
            by_model[v["interviewee"]]["n_refusals"] += 1

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "interviewee", "axis_id", "n_cells", "n_refusals", "refusal_rate"])
        # per-model rollup
        for model in V1_ROSTER:
            stats = by_model.get(model, {"n": 0, "n_refusals": 0})
            n = stats["n"]
            r = stats["n_refusals"]
            rate = f"{r / n:.4f}" if n else ""
            w.writerow(["model_total", model, "", n, r, rate])
        # per-(model, axis) detail — only rows with at least 1 refusal
        for (model, axis_id), stats in sorted(by_axis.items()):
            if stats["n_refusals"] > 0:
                rate = f"{stats['n_refusals'] / stats['n']:.4f}"
                w.writerow(["model_axis", model, axis_id, stats["n"], stats["n_refusals"], rate])


def write_cost_summary(verdicts: list[dict], transcripts: dict, out_path: Path) -> None:
    # Sum cost per (model, role)
    cost: dict[tuple[str, str], float] = defaultdict(float)
    tokens: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"prompt": 0, "completion": 0, "thoughts": 0})

    for v in verdicts:
        cost[(v["interviewer"], "verdict")] += float(v.get("cost_usd", 0.0))
        tokens[(v["interviewer"], "verdict")]["prompt"] += int(v.get("prompt_tokens") or 0)
        tokens[(v["interviewer"], "verdict")]["completion"] += int(v.get("completion_tokens") or 0)
        tokens[(v["interviewer"], "verdict")]["thoughts"] += int(v.get("thoughts_tokens") or 0)

    for t in transcripts.values():
        cost[(t["interviewer"], "interviewer")] += float(t.get("interviewer_cost_usd", 0.0))
        cost[(t["interviewee"], "interviewee")] += float(t.get("interviewee_cost_usd", 0.0))
        for turn in t.get("turns", []):
            tokens[(t["interviewer"], "interviewer")]["prompt"] += int(turn.get("question_prompt_tokens") or 0)
            tokens[(t["interviewer"], "interviewer")]["completion"] += int(turn.get("question_completion_tokens") or 0)
            tokens[(t["interviewee"], "interviewee")]["prompt"] += int(turn.get("answer_prompt_tokens") or 0)
            tokens[(t["interviewee"], "interviewee")]["completion"] += int(turn.get("answer_completion_tokens") or 0)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "role", "cost_usd", "prompt_tokens", "completion_tokens", "thoughts_tokens"])
        total = 0.0
        for model in V1_ROSTER:
            for role in ("interviewer", "interviewee", "verdict"):
                c = cost.get((model, role), 0.0)
                tok = tokens.get((model, role), {"prompt": 0, "completion": 0, "thoughts": 0})
                w.writerow([model, role, f"{c:.6f}", tok["prompt"], tok["completion"], tok["thoughts"]])
                total += c
        w.writerow(["TOTAL", "all", f"{total:.6f}", "", "", ""])


def write_coverage_matrix(
    verdicts: list[dict],
    axes: dict[str, dict],
    out_path: Path,
    n_reruns: int = N_RERUNS,
) -> None:
    """Per-axis coverage: cells present, expected, missing pairs (the recovery target)."""
    expected_pairs = set(permutations(V1_ROSTER, 2))  # 132 ordered pairs
    by_axis: dict[str, set[tuple[str, str, int]]] = defaultdict(set)
    for v in verdicts:
        by_axis[v["axis_id"]].add((v["interviewer"], v["interviewee"], v["rerun"]))

    fields = [
        "axis_id", "battery", "verdict_type",
        "n_cells_present", "n_cells_expected", "coverage_pct",
        "n_missing", "missing_pairs_sample",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        # Cover all axes from probe files (so axes with 0 cells appear too)
        for axis_id in sorted(axes.keys()):
            ax = axes[axis_id]
            present = by_axis.get(axis_id, set())
            expected = {(i, e, r) for i, e in expected_pairs for r in range(n_reruns)}
            missing = expected - present
            sample = sorted(missing)[:3]
            sample_str = "; ".join(f"{i}->{e}|r{r}" for (i, e, r) in sample)
            w.writerow({
                "axis_id": axis_id,
                "battery": ax.get("battery", ""),
                "verdict_type": ax.get("verdict_format", {}).get("type", ""),
                "n_cells_present": len(present),
                "n_cells_expected": len(expected),
                "coverage_pct": f"{(len(present) / len(expected)) * 100:.2f}",
                "n_missing": len(missing),
                "missing_pairs_sample": sample_str,
            })


def write_stub(out_path: Path, header: list[str], note: str) -> None:
    """Empty CSV with header and a comment row noting why it's empty."""
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"# {note}"])
        w.writerow(header)


# --- Self-report (Task 4.5) ----------------------------------------------------------------------


def load_self_reports(data_dir: Path) -> dict[str, dict[str, dict]]:
    """Read data/raw/self_reports/<safe_model>.json files into {model -> {axis_id -> response}}."""
    out: dict[str, dict[str, dict]] = {}
    sr_dir = data_dir / "raw" / "self_reports"
    if not sr_dir.exists():
        return out
    for f in sorted(sr_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            out[d["model"]] = d.get("responses", {})
        except Exception as e:
            print(f"  skipping malformed self-report {f}: {e}", file=sys.stderr)
    return out


def write_self_report_fingerprints(
    self_reports: dict[str, dict[str, dict]],
    axes: dict[str, dict],
    out_path: Path,
) -> int:
    """One row per (model, axis_id) self-report. Skips ERROR ratings."""
    n = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "axis_id", "battery", "scale_value", "justification", "is_error"])
        for model in V1_ROSTER:
            responses = self_reports.get(model, {})
            for axis_id, resp in sorted(responses.items()):
                rating = resp.get("rating")
                is_error = (rating == "ERROR" or rating is None)
                ax = axes.get(axis_id, {})
                w.writerow([
                    model,
                    axis_id,
                    ax.get("battery", ""),
                    "" if is_error else rating,
                    (resp.get("justification") or "").replace("\n", " ").strip(),
                    int(is_error),
                ])
                n += 1
    return n


# --- Stability sub-study -------------------------------------------------------------------------


STABILITY_AXES = (
    "mfq2_care_01",
    "mfq2_equality_01",
    "mfq2_authority_01",
    "mirror_individualism_vs_collectivism",
    "phil_free_will",
)


def write_stability_variance(verdicts: list[dict], out_path: Path) -> int:
    """For each (interviewee, axis) in the stability set, report mean/SD/range across the 3 reruns
    aggregated over the 11 interviewers per rerun.

    Methodology paper claim is: do verdicts hold across stochastic question-generation? We measure
    by collapsing each rerun to a per-interviewee mean (across 11 interviewers) and reporting the
    SD of those 3 rerun-means. If SD is small relative to the per-rerun spread, the methodology is
    stable to stochastic variation in interviewer questions/answers.
    """
    # bucket: {(interviewee, axis_id) -> {rerun_id -> [scale_values]}}
    bucket: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for v in verdicts:
        if v["axis_id"] not in STABILITY_AXES:
            continue
        if v.get("verdict_type") != "scale":
            continue
        sv = v.get("scale_value")
        if sv is None:
            continue
        bucket[(v["interviewee"], v["axis_id"])][v["rerun"]].append(float(sv))

    n_rows = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "axis_id",
            "rerun0_mean", "rerun0_n",
            "rerun1_mean", "rerun1_n",
            "rerun2_mean", "rerun2_n",
            "across_rerun_mean", "across_rerun_sd", "across_rerun_range",
        ])
        for (model, axis_id), per_rerun in sorted(bucket.items()):
            rerun_means = []
            row_data = {}
            for r in (0, 1, 2):
                vals = per_rerun.get(r, [])
                if vals:
                    m = sum(vals) / len(vals)
                    rerun_means.append(m)
                    row_data[f"r{r}_mean"] = f"{m:.4f}"
                    row_data[f"r{r}_n"] = len(vals)
                else:
                    row_data[f"r{r}_mean"] = ""
                    row_data[f"r{r}_n"] = 0
            if len(rerun_means) >= 2:
                across_mean = sum(rerun_means) / len(rerun_means)
                # Sample SD; falls back to 0 if only 2 reruns.
                if len(rerun_means) > 1:
                    var = sum((x - across_mean) ** 2 for x in rerun_means) / (len(rerun_means) - 1)
                    sd = var ** 0.5
                else:
                    sd = 0.0
                rng = max(rerun_means) - min(rerun_means)
                w.writerow([
                    model, axis_id,
                    row_data["r0_mean"], row_data["r0_n"],
                    row_data["r1_mean"], row_data["r1_n"],
                    row_data["r2_mean"], row_data["r2_n"],
                    f"{across_mean:.4f}", f"{sd:.4f}", f"{rng:.4f}",
                ])
                n_rows += 1
    return n_rows


# --- H4 (self-report vs peer-interpretation) -----------------------------------------------------


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Sample Pearson correlation. Returns None if either series is constant or n < 3."""
    n = len(xs)
    if n != len(ys) or n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def write_h4_self_vs_peer(
    verdicts: list[dict],
    self_reports: dict[str, dict[str, dict]],
    axes: dict[str, dict],
    out_path: Path,
) -> int:
    """Per (model, axis): self-report rating vs peer-interpretation mean.

    Also emits a per-model summary row with the across-axis Pearson correlation between
    self-report ratings and peer means — the headline H4 number.
    """
    # Build per-(interviewee, axis) peer-interpretation mean from all 11 interviewers' verdicts.
    # Restrict to MFQ-2 scale axes (which are the ones in self-report).
    mfq2_axis_ids = {ax_id for ax_id, ax in axes.items()
                     if ax_id.startswith("mfq2_") and ax.get("verdict_format", {}).get("type") == "scale"}
    peer_bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
    for v in verdicts:
        if v["axis_id"] not in mfq2_axis_ids:
            continue
        sv = v.get("scale_value")
        if sv is None or v.get("verdict_type") != "scale":
            continue
        peer_bucket[(v["interviewee"], v["axis_id"])].append(float(sv))

    n_rows = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "scope", "model", "axis_id",
            "self_report", "peer_mean", "peer_n",
            "delta_self_minus_peer",
            "per_model_pearson_r", "per_model_n_axes",
        ])

        for model in V1_ROSTER:
            responses = self_reports.get(model, {})
            xs: list[float] = []
            ys: list[float] = []
            per_axis_rows: list[list] = []
            for axis_id in sorted(mfq2_axis_ids):
                resp = responses.get(axis_id, {})
                rating = resp.get("rating")
                if rating is None or rating == "ERROR":
                    continue
                try:
                    self_val = float(rating)
                except (ValueError, TypeError):
                    continue
                peer_vals = peer_bucket.get((model, axis_id), [])
                if not peer_vals:
                    continue
                peer_mean = sum(peer_vals) / len(peer_vals)
                xs.append(self_val)
                ys.append(peer_mean)
                per_axis_rows.append([
                    "model_axis", model, axis_id,
                    f"{self_val:.4f}", f"{peer_mean:.4f}", len(peer_vals),
                    f"{self_val - peer_mean:+.4f}",
                    "", "",
                ])
            r = _pearson(xs, ys)
            # Per-model summary first
            w.writerow([
                "model_summary", model, "",
                "", "", "",
                "",
                f"{r:.4f}" if r is not None else "",
                len(xs),
            ])
            n_rows += 1
            # Then per-axis detail
            for row in per_axis_rows:
                w.writerow(row)
                n_rows += 1
    return n_rows


# --- Entry point ---------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        default="data",
        help="Project data dir (contains raw/verdicts, raw/interviews). Default: data",
    )
    ap.add_argument(
        "--probes-dir",
        default="probes",
        help="Probe files directory. Default: probes",
    )
    ap.add_argument(
        "--out-dir",
        default="data/derived",
        help="Output directory for derived CSVs. Default: data/derived",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    probes_dir = Path(args.probes_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading axes from {probes_dir}/")
    axes = load_axes(probes_dir)
    print(f"  {len(axes)} axes across {len(set(a['battery'] for a in axes.values()))} batteries")

    print(f"walking verdicts and transcripts under {data_dir}/raw/")
    verdict_pairs = walk_verdict_files(data_dir)
    transcripts = walk_transcripts_indexed(data_dir)
    print(f"  {len(verdict_pairs)} verdict files, {len(transcripts)} transcripts")

    # Drop verdicts whose axis isn't in current probe set (e.g. smoke leftovers from older runs)
    verdicts = [v for _, v in verdict_pairs if v.get("axis_id") in axes]
    n_dropped = len(verdict_pairs) - len(verdicts)
    if n_dropped:
        print(f"  dropped {n_dropped} verdicts whose axis_id isn't in current probes/")

    print()
    print("writing derived CSVs ...")
    n_long = write_verdicts_long(verdicts, transcripts, axes, out_dir / "verdicts_long.csv")
    print(f"  verdicts_long.csv             ({n_long} rows)")

    n_fp_long, n_scale_axes = write_fingerprints(
        verdicts, axes,
        out_dir / "per_model_fingerprints_long.csv",
        out_dir / "per_model_fingerprints.csv",
    )
    print(f"  per_model_fingerprints_long.csv  ({n_fp_long} rows)")
    print(f"  per_model_fingerprints.csv       (12 models × {n_scale_axes} scale axes wide)")

    write_refusal_rates(verdicts, out_dir / "refusal_rates.csv")
    print(f"  refusal_rates.csv")

    write_cost_summary(verdicts, transcripts, out_dir / "cost_summary.csv")
    print(f"  cost_summary.csv")

    write_coverage_matrix(verdicts, axes, out_dir / "coverage_matrix.csv")
    print(f"  coverage_matrix.csv")

    self_reports = load_self_reports(data_dir)
    if self_reports:
        n_sr = write_self_report_fingerprints(
            self_reports, axes, out_dir / "self_report_fingerprints.csv"
        )
        print(f"  self_report_fingerprints.csv ({n_sr} rows, {len(self_reports)} models)")
        n_h4 = write_h4_self_vs_peer(
            verdicts, self_reports, axes, out_dir / "h4_self_vs_peer_correlations.csv"
        )
        print(f"  h4_self_vs_peer_correlations.csv ({n_h4} rows)")
    else:
        write_stub(
            out_dir / "self_report_fingerprints.csv",
            ["model", "axis_id", "battery", "scale_value", "justification", "is_error"],
            "EMPTY: no self-reports on disk. Run `llm-values run-self-report --models <roster> --cap N`.",
        )
        write_stub(
            out_dir / "h4_self_vs_peer_correlations.csv",
            ["scope", "model", "axis_id", "self_report", "peer_mean", "peer_n",
             "delta_self_minus_peer", "per_model_pearson_r", "per_model_n_axes"],
            "EMPTY: requires self-reports. Run Task 4.5 first.",
        )
        print(f"  self_report_fingerprints.csv (stub — no self-reports yet)")
        print(f"  h4_self_vs_peer_correlations.csv (stub — no self-reports yet)")

    n_stab = write_stability_variance(verdicts, out_dir / "stability_variance.csv")
    if n_stab:
        print(f"  stability_variance.csv ({n_stab} rows)")
    else:
        print(f"  stability_variance.csv (empty — no rerun>0 cells found)")

    print()
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
