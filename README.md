# llm-values-survey

A reproducible study of how language models classify each other's preferences across moral, cultural, and pop-culture dimensions, using a peer-interview methodology.

This is the **receipts layer** — methodology, code, probes, raw data, derived CSVs, the pre-registration, and the methodology paper. Use this repo to verify, reproduce, or audit.

If you want to *see* the data — the explorable dashboard, the essay, and the video — go to [readgrounded.com/studies/cross-examination](https://readgrounded.com/studies/cross-examination). The interactive charts (per-model fingerprints, Atari validation scatter, per-axis browsers, interviewer personality matrix) live there, hydrated from the same `data/derived/*.csv` files committed here.

## Status

Pre-data-collection. Pre-registration: see `PRE_REGISTRATION.md` (committed before any data is collected).

## Setup

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
uv run pytest
```

## License

MIT for code; CC-BY-4.0 for data and methodology docs (see `LICENSE`).
