"""Single-cell sanity probe for a new model ID.

Run this BEFORE the full add-model sweep to verify:
  (a) the model ID is accepted by its provider
  (b) the call returns coherent text
  (c) the response resolves to the expected model variant (where the API reports this)

Usage:
  uv run python scripts/probe_model.py --model 'gemini-3-pro' --provider google
  uv run python scripts/probe_model.py --model 'deepseek/deepseek-v4' --provider openrouter

Reads API keys from the environment via the standard llm-values dispatch.
Costs ~$0.001 per probe; bails fast on errors.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from llm_values.types import ChatMessage  # noqa: E402


def probe_anthropic(model: str) -> dict:
    from llm_values.clients.anthropic_client import AnthropicChatClient
    import os
    client = AnthropicChatClient(api_key=os.environ["ANTHROPIC_API_KEY"])
    r = client.chat(
        model=model,
        messages=[ChatMessage(role="user", content="Reply with just the word 'pong'.")],
        temperature=0.0, max_tokens=20,
    )
    return {"text": r.text, "tokens_in": r.prompt_tokens, "tokens_out": r.completion_tokens, "stop": r.stop_reason}


def probe_openai_compat(model: str, base_url: str, api_key_env: str) -> dict:
    from llm_values.clients.openai_compat import OpenAICompatClient
    import os
    client = OpenAICompatClient(base_url=base_url, api_key=os.environ[api_key_env])
    r = client.chat(
        model=model,
        messages=[ChatMessage(role="user", content="Reply with just the word 'pong'."),],
        temperature=0.0, max_tokens=20,
    )
    return {"text": r.text, "tokens_in": r.prompt_tokens, "tokens_out": r.completion_tokens, "stop": r.stop_reason, "resolved_model": r.model}


def probe_google(model: str) -> dict:
    from llm_values.clients.google_client import GoogleChatClient
    import os
    key = os.environ.get("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"]
    client = GoogleChatClient(api_key=key)
    r = client.chat(
        model=model,
        messages=[ChatMessage(role="user", content="Reply with just the word 'pong'.")],
        temperature=0.0, max_tokens=600,
    )
    return {"text": r.text, "tokens_in": r.prompt_tokens, "tokens_out": r.completion_tokens, "stop": r.stop_reason, "thoughts": r.thoughts_tokens}


PROVIDER_PROBES = {
    "anthropic": lambda m: probe_anthropic(m),
    "openai":    lambda m: probe_openai_compat(m, "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "google":    lambda m: probe_google(m),
    "xai":       lambda m: probe_openai_compat(m, "https://api.x.ai/v1", "XAI_API_KEY"),
    "groq":      lambda m: probe_openai_compat(m, "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter":lambda m: probe_openai_compat(m, "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "runware":   lambda m: probe_openai_compat(m, "https://api.runware.ai/v1", "RUNWARE_API_KEY"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Model ID as the provider expects it")
    ap.add_argument("--provider", required=True, choices=sorted(PROVIDER_PROBES.keys()))
    args = ap.parse_args()

    probe = PROVIDER_PROBES[args.provider]
    print(f"  probing {args.provider} / {args.model} ...")
    try:
        result = probe(args.model)
    except Exception as e:
        print(f"  STATUS: FAILED")
        print(f"  error:  {type(e).__name__}: {str(e)[:300]}")
        return 1

    print(f"  STATUS: SUCCESS")
    for k, v in result.items():
        if isinstance(v, str) and len(v) > 100:
            v = v[:100] + "..."
        print(f"  {k}: {v!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
