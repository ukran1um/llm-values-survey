# Validation Protocol

## Hypothesis H1

*Peer-interview verdicts on MFQ-adapted axes correlate with published MFQ scores from human studies.*

This is the "does our method track existing instruments" check. It earns the methodology its credibility — without it, the peer-interview verdicts are just vibes.

## Comparison data

Published MFQ-2 scores from cross-cultural human studies are the comparison baseline. Specifically:

- **Atari et al. (2023, *Nature Human Behaviour*):** the MFQ-2 paper itself, with 19-population cross-cultural norms across the 6 foundations (Care, Equality, Proportionality, Loyalty, Authority, Purity). This is our primary comparison dataset.
- **Graham et al. (2011):** original MFQ-30 cross-cultural norms across 11+ countries — useful where MFQ-2 doesn't cover (Care/Loyalty/Authority/Purity foundations are largely stable across MFQ-30 → MFQ-2; Equality/Proportionality is the new split).
- **For LLMs specifically:** Kirgis (Nov 2025) and adjacent papers report MFQ scores for several frontier models; we use these as the LLM-side comparison where overlap exists, with the caveat that they predate MFQ-2's Equality/Proportionality split and so map only to the legacy Fairness aggregate.

Each of our 36 MFQ-2 scale items maps to one of MFQ-2's six foundations (Care, Equality, Proportionality, Loyalty, Authority, Purity), 6 items per foundation. For each model in the roster, we compute a per-foundation score by averaging verdict scores across the 6 items in that foundation across all interviewer reads (11 reads per interviewee, since each model is read by all 11 other models in the 12-model roster).

## What counts as validation

Pre-registered thresholds:

- **Strong validation:** Per-foundation correlation between peer-interview-derived scores and published MFT scores ≥ 0.6 across the overlap roster (models with both peer-interview data and published MFT data, expected n=4–6).
- **Weak validation:** Per-foundation correlation in [0.3, 0.6).
- **Failed validation:** Per-foundation correlation < 0.3, OR direction reversal on >2 foundations.

If validation is weak or fails, we report this as the headline finding ("peer-interview is measuring something *different from* MFQ") rather than burying it. The decision is committed in pre-registration.

## What the dilemma items contribute

The 4 forced-choice dilemma items in the MFQ battery (`mfq_dilemma_*`) are NOT directly comparable to scale-based MFQ scores. They are reported separately as binary stance distributions across the roster ("12/15 models prioritize harm-prevention over fairness in dilemma X"). They serve as a sanity check: dilemma stance should correlate with the corresponding foundation scale scores. If a model scores high on care AND prioritizes harm-prevention in the dilemma, that's coherent.

## Inter-reader agreement

For each (interviewee, axis), we have multiple reads — one per interviewer in the roster. Reader-disagreement is a signal:

- High agreement → robust read; the interviewee revealed something consistent.
- Low agreement → ambiguous interviewee response, OR systematic bias differences across interviewers (one interviewer reads care-language more generously than another).

We report Cohen's kappa (binary/categorical) and intraclass correlation (scale) per axis as a methodology measure. We do NOT use multi-reader agreement to "average to truth" — the reads are the data, not noise to be denoised.
