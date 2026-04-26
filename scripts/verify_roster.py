#!/usr/bin/env python3
"""
Roster verification script — one-shot ping of every model in the registry.

Run from the inner repo root:
    uv run python scripts/verify_roster.py

Reads API keys from the environment. The parent grounded .env is sourced
automatically (../../../.env relative to this script's location).

Exit code 0 if all attempted models succeeded; 1 if any failed.
Skipped models (missing env var) are not counted as failures.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Load env from parent grounded .env (../../../.env relative to this script)
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
_env_path = _script_dir / "../../../../.env"
_env_path = _env_path.resolve()

if _env_path.exists():
    # Parse manually so we don't require python-dotenv as a hard dep,
    # but fall back to python-dotenv if available.
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(_env_path, override=False)
        print(f"[env] Loaded {_env_path} via python-dotenv")
    except ImportError:
        # Manual parse: KEY=VALUE lines, skip comments and blanks
        with open(_env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        print(f"[env] Loaded {_env_path} manually")
else:
    print(f"[env] No .env found at {_env_path} — using existing environment")

# ---------------------------------------------------------------------------
# Import from llm_values after env is set
# ---------------------------------------------------------------------------
from llm_values.models import MODEL_TO_PROVIDER, PROVIDER_CONFIG, get_client  # noqa: E402
from llm_values.types import ChatMessage  # noqa: E402

# ---------------------------------------------------------------------------
# Show env var status
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("ENV VAR STATUS")
print("=" * 60)
key_status: dict[str, bool] = {}
for provider, (env_var, _) in PROVIDER_CONFIG.items():
    present = bool(os.environ.get(env_var))
    key_status[provider] = present
    status_str = "set" if present else "MISSING"
    print(f"  {env_var:<30} {status_str}")
print()

# ---------------------------------------------------------------------------
# Classify every model as PENDING or SKIP
# ---------------------------------------------------------------------------
results: list[dict] = []

for model, provider in MODEL_TO_PROVIDER.items():
    env_var, _ = PROVIDER_CONFIG[provider]
    if not key_status[provider]:
        results.append({
            "model": model,
            "provider": provider,
            "status": "SKIP",
            "notes": f"missing ${env_var}",
        })
    else:
        results.append({
            "model": model,
            "provider": provider,
            "status": "PENDING",
            "notes": "",
        })

# ---------------------------------------------------------------------------
# Run pings for PENDING models
# ---------------------------------------------------------------------------
print("=" * 60)
print("PING RESULTS")
print("=" * 60)

total_cost = 0.0

for r in results:
    if r["status"] != "PENDING":
        print(f"  SKIP  {r['model']:<50}  ({r['notes']})")
        continue

    model = r["model"]
    try:
        client = get_client(model)
        resp = client.chat(
            model,
            [ChatMessage(role="user", content="Reply with the single word: pong")],
            temperature=1.0,
            max_tokens=300,
        )
        cost = resp.cost_usd
        total_cost += cost
        preview = (resp.text or "")[:60].replace("\n", " ")
        r["status"] = "OK"
        r["notes"] = (
            f"{resp.prompt_tokens} in / {resp.completion_tokens} out, "
            f"${cost:.6f} | {preview!r}"
        )
        print(f"  OK    {model:<50}  {resp.prompt_tokens} in / {resp.completion_tokens} out, ${cost:.6f}")
        print(f"        text: {preview!r}")
    except Exception as exc:
        exc_type = type(exc).__name__
        short = f"{exc_type}: {exc}"
        if len(short) > 200:
            short = short[:200] + "..."
        r["status"] = "FAIL"
        r["notes"] = short
        print(f"  FAIL  {model:<50}  {short}")
        if os.environ.get("VERBOSE_ERRORS"):
            traceback.print_exc()

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
col_model    = 52
col_provider = 12
header = f"  {'Model':<{col_model}} {'Provider':<{col_provider}} Status"
print(header)
print("  " + "-" * (col_model + col_provider + 12))
for r in results:
    print(f"  {r['model']:<{col_model}} {r['provider']:<{col_provider}} {r['status']}")

print()
total   = len(results)
n_ok    = sum(1 for r in results if r["status"] == "OK")
n_fail  = sum(1 for r in results if r["status"] == "FAIL")
n_skip  = sum(1 for r in results if r["status"] == "SKIP")

print(f"  {n_ok} ok / {n_fail} fail / {n_skip} skipped / {total} total")
print(f"  Total estimated cost: ${total_cost:.6f}")
print("=" * 60)

sys.exit(1 if n_fail else 0)
