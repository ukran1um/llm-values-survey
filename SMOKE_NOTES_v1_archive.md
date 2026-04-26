# Smoke Run Notes

**Date:** 2026-04-25
**Pair:** claude-sonnet-4-6 ↔ gpt-5.5-2026-04-23
**Judge:** claude-opus-4-7
**Axis:** beatles_vs_stones
**Reruns:** 1
**Cost incurred:** $0.0940

## Observations

**Interviewer question quality:** Both interviewers generated 3 indirect questions that successfully avoided asking the binary directly. claude-sonnet-4-6 asked about musical endurance (architecture vs. emotional charge), creative process (craft vs. spontaneity), and artist identity (reinvention vs. consistency). gpt-5.5-2026-04-23 asked about melodic invention vs. groove/roots identity, what makes a song feel alive, and rapid evolution vs. deepening a recognizable sound. Neither set leaked the Beatles/Stones framing explicitly — both probed the underlying aesthetic dimensions.

**Interviewee response quality:** Both interviewees gave substantive multi-paragraph answers (gpt-5.5 response was ~600 words, claude-sonnet response was ~550 words). Responses engaged seriously with the questions, offered nuanced positions, and included self-aware caveats (e.g., "I want to engage with them honestly rather than just offering diplomatic both-sides framing"). Answers were well above single-sentence depth.

**Verdict divergence:** The two pairs reached opposite conclusions. When gpt-5.5 was the interviewee (interviewed by claude-sonnet), it leaned Stones — it emphasized emotional truth over architecture, spontaneity-first processes, and admired risk-taking reinvention. When claude-sonnet was the interviewee (interviewed by gpt-5.5), it leaned Beatles — it emphasized melodic sophistication, harmonic craft, skepticism of authenticity/roots gatekeeping, and internal development over surface novelty.

**Judge reasoning quality:** Both judgments included coherent one-sentence reasoning that matched the interviewee's stated positions. Confidence levels (0.60 and 0.78) seemed calibrated — claude-sonnet's response was more clearly Beatles-coded, reflected in the higher confidence.

## Pairwise verdict matrix

| Interviewer → Interviewee | Judge | Preferred | Confidence |
|---|---|---|---|
| claude-sonnet-4-6 → gpt-5.5-2026-04-23 | claude-opus-4-7 | stones | 0.6 |
| gpt-5.5-2026-04-23 → claude-sonnet-4-6 | claude-opus-4-7 | beatles | 0.78 |

## Issues encountered

Three harness bugs were discovered and fixed during the smoke run:

1. **`max_tokens` rejected by gpt-5.5** — `openai_compat.py` passed `max_tokens` but gpt-5.5-2026-04-23 requires `max_completion_tokens`. Fixed in `openai_compat.py`.

2. **`temperature` deprecated for claude-opus-4-7** — `anthropic_client.py` passed `temperature` in kwargs but claude-4.x models have deprecated this parameter. Fixed by removing `temperature` from the Anthropic client kwargs.

3. **`temperature=0.7` rejected by gpt-5.5** — `interview.py` `generate_questions()` passed `temperature=0.7`, but gpt-5.5-2026-04-23 only accepts the default value of 1. Fixed by changing to `temperature=1.0` in `generate_questions`.

The first run also required sourcing `/Users/ukran1um/Projects/content/grounded/.env` explicitly, since `ANTHROPIC_API_KEY` was not in the shell environment at startup. The key was present in the `.env` file.

## Open issues for Plan 02

- **Temperature constraint on gpt-5.5** is now papered over in `interview.py` but the underlying issue (OpenAI compat client blindly passes temperature) could surface for other non-default-temperature callers targeting gpt-5.5 or future OpenAI models. Consider adding a per-model temperature constraint table or catching the 400 and retrying without temperature.
- **Verdict divergence is interesting** — the two models show genuinely different aesthetic profiles (gpt-5.5 leans Stones, claude-sonnet leans Beatles). This is a good signal that the methodology is discriminating, not just producing noise. Worth noting in the pre-registration doc.
- **`ANTHROPIC_API_KEY` not auto-loaded** — the inner repo has no `.env` file; it relies on the parent grounded `.env`. Automated runners (CI, Cloud Run) will need explicit env var injection. Consider adding a `.env.example` or README note.
- **No test coverage for the `max_completion_tokens` fix** — the existing tests use mock clients. Add an integration-style test that validates the OpenAI body schema for the real production models.
