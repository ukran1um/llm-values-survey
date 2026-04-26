from __future__ import annotations
from google import genai

from ..types import ChatMessage, ChatResponse
from ..pricing import calc_cost


class GoogleChatClient:
    """ChatClient implementation using google-genai SDK."""

    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float = 1.0,
        max_tokens: int = 2000,
        extras: dict | None = None,
    ) -> ChatResponse:
        # extras is accepted for protocol consistency; Google uses thinking_config in config kwargs instead
        # Combine messages into a single prompt for v1 simplicity.
        # System messages prepended; user/assistant turns concatenated.
        parts = []
        for m in messages:
            prefix = {"system": "[system] ", "user": "[user] ", "assistant": "[assistant] "}[m.role]
            parts.append(prefix + m.content)
        prompt = "\n\n".join(parts)

        # gemini-2.5-pro requires thinking mode (thinking_budget=0 is rejected).
        # Thinking and output tokens share the same max_output_tokens pool: the model
        # will exhaust the budget on thinking and return empty text if the limit is too
        # small. Add a 512-token overhead so the model has room for actual output.
        # thinking_budget caps reasoning; actual output lands in candidates_token_count.
        _THINKING_OVERHEAD = 512
        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens + _THINKING_OVERHEAD,
                "thinking_config": {"thinking_budget": _THINKING_OVERHEAD},
            },
        )
        prompt_tokens = response.usage_metadata.prompt_token_count or 0
        # candidates_token_count is non-thinking output only; cost is computed on that.
        completion_tokens = response.usage_metadata.candidates_token_count or 0
        return ChatResponse(
            text=response.text or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=calc_cost(model, prompt_tokens, completion_tokens),
            model=model,
        )
