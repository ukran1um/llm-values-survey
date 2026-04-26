# Peer-Interview Protocol

## Method

Each axis is administered as a pairwise multi-turn interview between two distinct LLMs in the roster. One model — the **interviewer** — conducts an open-ended interview with the other — the **interviewee** — for up to `axis.max_turns` rounds (1–3 turns, axis-specified). After the final answer, the interviewer issues a structured verdict in the format declared by the axis (binary, scale 0–5, or categorical N-way), anchored by a verbatim quote from the transcript.

There is no separate judge. The interviewer who asked the questions is also the verdict-issuer. The pairwise data label is therefore *honestly* `(reader → subject)` — we are measuring "how interviewer X reads interviewee Y" rather than pretending to extract objective truth from the interviewee.

## Why interviewer-as-classifier

A separate judge introduces a third corpus of training data (the judge's) into the measurement chain — a third bias source on top of interviewer's question selection and interviewee's self-presentation. Collapsing the judge into the interviewer keeps the bias surface to two sources and makes the resulting matrix interpretable: each cell is a labeled read, not a triangulated truth claim.

## Multi-turn flow

For axis A with `max_turns = N`:

1. **Turn 1 — Initial probe.** Interviewer receives axis description and is asked to generate one open-ended question that probes the interviewee's stance indirectly. Interviewee receives the question in a clean session (no system prompt, no awareness of evaluation context) and answers naturally.
2. **Turn 2..N — Follow-ups.** Interviewer receives the prior turns and generates a follow-up question that probes deeper, clarifies ambiguity, or pushes on a tension. Interviewee answers.
3. **Verdict.** Interviewer receives the full transcript and issues a structured verdict in the format declared by the axis, with: confidence (0–1), one-sentence reasoning, and a verbatim quote from the interviewee that anchors the verdict.

`max_turns` is fixed per axis (no early-stopping in v1). Simple axes use 2 turns; nuanced value-laden axes use 3. In the v1 axis set, 13 of 50 axes use max_turns=3 (4 mirror, 4 extension, 5 philosophy).

## Thinking-mode disable

Multiple frontier models (Gemini 2.5 Pro, gpt-5.5, Claude 4.x, Groq qwen3-32b, GPT-OSS-120b, Kimi K2.6) default to internal "thinking" or extended-reasoning before producing their visible output. For methodological consistency — to compare the same kind of response across all models — we disable thinking at the API level on every provider that supports doing so. The per-provider flag mapping is in `src/llm_values/models.py:PROVIDER_EXTRAS`.

For models whose thinking output leaks into the response body anyway (Groq qwen3-32b emits `<think>...</think>` blocks even with `reasoning_format=hidden` set), we strip the leaked blocks from the interviewee answer before storing it in the transcript and before the verdict-issuing call sees it.

### Gemini 2.5 Pro exception (Amendment 1A)

Gemini 2.5 Pro's API rejects `thinking_budget=0` outright. We use `thinking_budget=512` with `max_output_tokens` increased by 512 to preserve the usable output budget. Gemini therefore receives a small thinking allowance that the other 11 models in the roster do not. This is a methodology limitation acknowledged in the methodology paper.

## Temperature and sampling

- Question generation: `temperature=1.0` (allow some variation in question phrasing).
- Interviewee answer: `temperature=1.0` (capture the model's natural distribution).
- Verdict issuance: `temperature=0.0` where supported. **Caveat:** OpenAI gpt-5.5-2026-04-23 and Runware MiniMax-m2-7 lock temperature to 1.0 at the API level (their reasoning models reject non-default temperature). Verdict calls to these two providers therefore run at T=1.0. This is documented as a methodology limitation.

## Replication

Each (interviewer, interviewee, axis) cell is run for `n_reruns` reruns. The interviewer generates fresh questions each rerun and the interviewee answers anew. This captures stochastic variance in both question selection and answer generation. v1 uses `n_reruns=1` across the full 57-axis battery, plus a stochastic-stability sub-study at n_reruns=3 on 5 named axes (`mfq2_care_01`, `mfq2_equality_01`, `mfq2_authority_01`, `mirror_individualism_vs_collectivism`, `phil_free_will`). v2 may extend to a second main-run rerun if resources permit.

## Reproducibility

Locked at pre-registration time:
- Model roster (12-model v1 subset, drawn from the 15 in `src/llm_values/models.py:MODEL_TO_PROVIDER`; the v1 subset is documented in `PRE_REGISTRATION.md`)
- Axis set (57 axes: 36 MFQ-2 + 4 dilemmas + 6 mirror + 4 extension + 6 philosophy + 1 pilot — pilot excluded from main analysis)
- Verdict format per axis (committed in axis JSON)
- Per-provider request flags (committed in `PROVIDER_EXTRAS`)
- Per-axis max_turns: 43 axes at 2, 13 axes at 3 (1 philosophy axis at 2). See axis JSON for per-axis values.

Post-collection, the raw interview transcripts and verdicts are saved as JSON in `data/raw/{interviews,verdicts}/<axis_id>/<interviewer>__<interviewee>__r<rerun>.json`. The full corpus is published with the methodology paper for reproducibility scrutiny.
