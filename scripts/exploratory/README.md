# Exploratory analyses

Post-hoc analyses that were **not** in `PRE_REGISTRATION.md`. They surface findings the
pre-registered hypotheses (H1-H4) didn't capture.

These analyses inform discussion / future hypotheses but **do not** constitute confirmatory
evidence. The methodology paper reports them as exploratory, with clear flagging.

| Script | What it computes |
|---|---|
| `outlier_per_axis.py` | Per-axis: which model is most distant from the consensus mean? Aggregate: which models are most often outliers, and on which axes? |
| `axis_consensus.py` | Per-axis: SD across all 132 cells. Sorted output identifies the consensus axes (LLMs all read each other the same way) vs. the contested axes (high inter-model disagreement). |
| `quote_mining.py` | Walks the 8,170 `key_quote` fields. Finds (a) consensus quotes — passages multiple interviewers extract from the same interviewee, (b) high-disagreement cells where the quotes diverge. Useful narrative content for the essay. |
| `care_offset_per_item.py` | Decomposes the H1 "LLMs are hyper-Care vs humans" foundation-level finding into per-item analysis. Identifies which Care items (and their language) drive the offset. |

Re-runnable against any snapshot — just rerun `scripts/aggregate.py` first.
