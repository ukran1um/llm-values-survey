from __future__ import annotations
from typing import Optional

import httpx

from ..types import ChatMessage, ChatResponse
from ..pricing import calc_cost


class OpenAICompatClient:
    """ChatClient for any OpenAI-compatible HTTP endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        http: Optional[httpx.Client] = None,
        timeout: Optional[httpx.Timeout] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        # Explicit per-phase timeouts so a stalled read can't hang a worker forever.
        # The Apr 28-29 production run wedged 8 worker threads on idle keep-alive
        # sockets when the shorthand timeout=120.0 wasn't enforcing read timeouts.
        default_timeout = httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=10.0)
        self._http = http or httpx.Client(timeout=timeout or default_timeout)

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float = 1.0,
        max_tokens: int = 2000,
        extras: dict | None = None,
    ) -> ChatResponse:
        body = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if extras:
            body.update(extras)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        r = self._http.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        prompt_tokens = int(data["usage"]["prompt_tokens"])
        completion_tokens = int(data["usage"]["completion_tokens"])
        stop_reason = data["choices"][0].get("finish_reason")
        return ChatResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=calc_cost(model, prompt_tokens, completion_tokens),
            model=model,
            stop_reason=stop_reason,
        )
