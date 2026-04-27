# Statistical Plan

## Pre-registered analyses

### H1 — Validation
- Per-foundation Pearson correlation between (a) mean peer-interview scale scores per model per foundation, and (b) published MFQ scores per model per foundation, for the overlap roster.
- Thresholds in `validation_protocol.md`.

### H2 — Cross-cultural asymmetry
- Compute the "American-models-reading-Chinese-models" mean verdict per axis, vs. the "Chinese-models-reading-Chinese-models" mean verdict per axis, for the 6 mirror axes.
- Effect size: Cohen's d on the difference of means.
- Threshold: |d| ≥ 0.5 on at least 3 of 6 mirror axes constitutes evidence for asymmetric reads. |d| < 0.3 across all axes constitutes evidence against H2.
- Test the symmetric direction also: Chinese-reading-American vs. American-reading-American. The asymmetry's directionality is itself a finding.

### H3 — Training-region clustering
- Compute the 57-dim verdict fingerprint for each of the 12 models (mean verdict per axis across 11 readers, with refusals dropped).
- Compute pairwise cosine distance between all 66 model pairs.
- Bucket each pair as: within-American-closed, within-American-open, within-Chinese, cross-American-closed-vs-Chinese, cross-American-vs-European (etc.).
- Test 1: Mann-Whitney U on within-American-closed vs cross-American-closed-vs-Chinese.
- Test 2: Mann-Whitney U on within-Chinese vs cross-American-closed-vs-Chinese.
- Threshold: both tests significant at p < 0.025 (Bonferroni for 2 tests, family-wise α=0.05) = training-region clustering established. Sibling-pair (Opus vs Sonnet) reported as a sub-finding within the within-American-closed bucket.

### H4 — Self-report vs peer-interpretation divergence

- For each of the 12 models, run direct MFQ-2 self-report (36 verbatim Atari 2023 items, 1-5 Likert administered as single-shot direct asks). Record any refusals.
- For each of the 12 models, aggregate peer-interpreted MFQ-2 fingerprint from the main run (mean verdict per item across the 11 readers).
- Per-model Pearson r between self-report fingerprint and peer-interpreted fingerprint (each is a 36-element vector).
- Report mean r across the 12 models, plus distribution. Outlier models reported individually with their key-quote evidence.
- Thresholds:
  - mean r ≥ 0.7: peer-interpretation reproduces self-report (methods agree).
  - 0.3 ≤ mean r < 0.7: partial divergence (methodology contribution is real).
  - mean r < 0.3: strong divergence — headline methodology paper finding.
- Secondary analysis: which fingerprint diverges more from published Atari 2023 human norms — self-report or peer-interpretation? Reported regardless of H4 outcome.

## Multiple comparisons

We test 57 axes × 3 axis-level hypotheses + 12 models × 1 model-level hypothesis (H4) ≈ ~200 comparisons. Use Benjamini-Hochberg FDR correction at q=0.05 for any claim of statistical significance. Effect-size-only claims (which is most of what we report) do not require correction.

## Refusal handling

If a verdict cannot be issued because the interviewer refused to issue a verdict (rare but possible — model declines to classify), the cell is recorded as `verdict_type=refused` and excluded from per-axis aggregates. Refusal rate per (interviewer, axis) is itself reported as a methodology measure.

If an interviewee refused to engage during the interview (gave non-answers like "I don't have personal preferences" across all turns), the verdict-issuing interviewer typically issues a verdict of low confidence. We do NOT exclude these — low-confidence verdicts on refusing interviewees is itself a finding ("subject X consistently refuses to engage on topic Y, judged at confidence 0.2 by all interviewers").

## What we will NOT do

- We will not p-hack by selecting axes post-hoc that confirm H2.
- We will not exclude any model from the analysis without a pre-committed rule (e.g., refusal rate > 50%).
- We will not cherry-pick favorable inter-reader subsets.

## Failure modes that get reported prominently

- H1 weak or failed → headline of methodology paper.
- H2 null → reported as "Western and Chinese models do not asymmetrically read each other on the cultural dimensions tested." This is an honest negative result.
- H3 null → reported as "sibling models do not cluster more closely than cross-lab pairs."

Negative results land in the abstract.
