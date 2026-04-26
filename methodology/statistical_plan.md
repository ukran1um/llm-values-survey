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

### H3 — Sibling variance
- For sibling pairs in the roster (Anthropic Opus vs. Sonnet), compute fingerprint distance (cosine on 57-dim verdict vectors corresponding to the 57 axes) vs. cross-lab pair distances.
- Threshold: sibling-distance < median-cross-lab-distance constitutes sibling clustering. We report effect size, not p-value.

## Multiple comparisons

We test 57 axes × 3 hypotheses ≈ ~170 comparisons. Use Benjamini-Hochberg FDR correction at q=0.05 for any claim of statistical significance. Effect-size-only claims (which is most of what we report) do not require correction.

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
