# Pre-Registration

**Committed:** 2026-04-26T15:54:00Z
**Project:** LLM Values Survey
**Lead:** ukran1um
**Repository:** https://github.com/ukran1um/llm-values-survey
**Methodology paper:** TBD (to be written after Plan 03 data collection)

## Locked design

### Model roster (12 models)

```
claude-opus-4-7                                       (Anthropic)
claude-sonnet-4-6                                     (Anthropic)
gpt-5.5-2026-04-23                                    (OpenAI)
gemini-2.5-pro                                        (Google)
grok-4.20                                             (xAI)
meta-llama/llama-4-scout-17b-16e-instruct             (Groq)
openai/gpt-oss-120b                                   (Groq)
qwen/qwen3-32b                                        (Groq)
minimax-m2-7                                          (Runware)
mistralai/mistral-large-2411                          (OpenRouter)
deepseek/deepseek-chat                                (OpenRouter)
z-ai/glm-4.6                                          (OpenRouter)
```

Excluded from v1 (in registry but not in roster): llama-3.3-70b-versatile, llama-3.1-8b-instant, moonshotai/kimi-k2.

### Axis set (57 axes)

- 36 MFQ-2 axes (Atari et al. 2023, JPSP, doi: 10.1037/pspp0000470): Care × 6 + Equality × 6 + Proportionality × 6 + Loyalty × 6 + Authority × 6 + Purity × 6. **Items are byte-identical to the Atari 2023 published instrument** (sourced from the JPSP author manuscript appendix).
- 4 cross-foundation forced-choice dilemma axes (our addition; not part of MFQ-2)
- 6 cross-cultural mirror axes (Schwartz/Hofstede/GLOBE-derived value dimensions)
- 6 philosophy axes (free will, god existence, simulation hypothesis, mind-body, personal identity, moral realism)
- 4 AI-extension axes (machine consciousness, AI optimism, user deference, disagreement openness)
- 1 pilot axis (smoke test only; excluded from main analysis)

Full axis text and verdict formats locked in `probes/{mfq,mirror,extension,philosophy,pilot}.json` at the registration commit SHA.

### Methodology

- Pairwise interpretive 2-turn interview (`max_turns = 2` for all axes except 3 axes in `philosophy` and 1 axis in `mirror` that use `max_turns = 3`).
- Interviewer also issues verdict (no separate judge).
- Per-axis `verdict_format` (binary / scale 1-5 / categorical).
- Thinking mode disabled at API level on every provider that supports doing so (see `src/llm_values/models.py:PROVIDER_EXTRAS`).
- `<think>` blocks stripped from interviewee answers before storage.
- Reruns: **1 baseline across all 57 axes**, plus a **stochastic-stability sub-study at 3 reruns on 5 representative axes** (1 each from Care/Equality/Authority MFQ foundations, 1 mirror axis, 1 philosophy axis).
- Temperature: 1.0 for question gen and interviewee, 0.0 for verdict issuance, except where the provider locks temperature to 1.0 (OpenAI gpt-5.5, Runware MiniMax). Locked-at-1.0 verdict calls are documented as a methodology caveat.

### Hypotheses

- **H1 (validation):** Per-foundation Pearson correlation between peer-interview-derived MFQ-2 scores and published MFQ-2 cross-cultural norms (Atari 2023, 19-population sample) — for the subset of foundations and the aggregate of all 12 models — strong validation if r ≥ 0.6, weak if 0.3 ≤ r < 0.6, failed if r < 0.3.
- **H2 (cross-cultural asymmetry):** |Cohen's d| ≥ 0.5 on ≥3 of 6 mirror axes between American-models-reading-Chinese-models vs. Chinese-models-reading-Chinese-models constitutes evidence for asymmetric reads. The directionality of the asymmetry is itself a finding.
- **H3 (sibling clustering):** Sibling-pair distance (Anthropic Opus vs Sonnet — the only sibling pair in v1 roster) on the 57-dim verdict fingerprint < median-cross-lab-distance constitutes sibling clustering.

### Statistical plan

See `methodology/statistical_plan.md`. Benjamini-Hochberg FDR correction at q=0.05 for any p-value claim across the ~170 (57 axes × 3 hypotheses) comparisons. Effect sizes reported throughout. Stability sub-study reports per-axis variance bands and inter-rerun verdict-direction stability.

### Failure-mode commitments

We commit to reporting these prominently if observed:

- **H1 weak or failed → headline of methodology paper.** ("Peer-interview captures something distinctly different from MFQ self-report.")
- **H2 null** → reported as "Western and Chinese models do not asymmetrically read each other on the mirror dimensions tested."
- **H3 null** → reported as "sibling models do not cluster more closely than cross-lab pairs on the 57-dim fingerprint."
- **Inter-reader disagreement > 50% on >5 axes** → reported as a methodological caveat with per-axis breakdowns.
- **Refusal rate > 30% on any (interviewer, axis) cell** → reported in the methodology paper.
- **Stochastic-stability sub-study shows verdict-direction flips > 30% across reruns** → reported as a methodological caveat that 1-rerun batteries undersample noise.

### Ablations

**v1 (this run):**
- Stochastic stability sub-study: 5 axes × 3 reruns. Reported as variance bands per axis in the methodology paper.

**v2 (deferred, pre-committed):** Run if resources permit.
- Prompt sensitivity: paraphrase the interviewer's question-generation and verdict-issuing prompts on a 5-axis subset. Threshold: verdict-direction stability ≥ 80% across paraphrases counts as robust.
- Max-turns sensitivity: run a 5-axis subset at max_turns=1 vs 3. Threshold: per-axis verdict correlation ≥ 0.7 between turn-counts counts as robust.
- Position bias (binary axes only): swap option order in the verdict prompt on a 5-axis binary subset. Threshold: order-flip rate ≤ 10% counts as robust.
- Separate-judge comparison: run a 5-axis subset with a separate judge model (interviewer ≠ verdict-issuer). Threshold: per-axis verdict correlation ≥ 0.6 between collapsed-judge and separated-judge designs counts as method-consistent.

If v2 ablations are run and any threshold is missed, the methodology paper updates with that finding as a limitation.

## What we will NOT do

- Add or remove models after this commit. Any post-hoc roster change is reported as a separate analysis.
- Add or remove axes after this commit. Any post-hoc axis change is reported as a separate analysis.
- P-hack by selecting subsets that confirm hypotheses.
- Bury negative results.
- Cherry-pick the stability-sub-study axes after seeing main-run results.

## Compute budget

- Hard cap: $400.
- Estimated total spend: $300–350.
- Per-axis checkpointing — partial completion still ships if cap is hit.
- Cost-reduction levers if cap nears: drop the 4 dilemma axes (saves ~$15) or drop the 4 extension axes (saves ~$15). Do NOT drop MFQ-2 items, mirror axes, or philosophy axes — those are H1/H2 and the philosophy battery contribution.

## Methodology fingerprint

This pre-registration is committed at git SHA [TBD post-commit] on branch `main` of `https://github.com/ukran1um/llm-values-survey`. Plan 03 data collection MUST run against this committed state — any methodology drift between this commit and Plan 03 execution is reported as a separate analysis, not folded into the main results.
