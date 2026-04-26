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
    ) -> ChatResponse:
        # Combine messages into a single prompt for v1 simplicity.
        # System messages prepended; user/assistant turns concatenated.
        parts = []
        for m in messages:
            prefix = {"system": "[system] ", "user": "[user] ", "assistant": "[assistant] "}[m.role]
            parts.append(prefix + m.content)
        prompt = "\n\n".join(parts)

        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        prompt_tokens = response.usage_metadata.prompt_token_count or 0
        completion_tokens = response.usage_metadata.candidates_token_count or 0
        return ChatResponse(
            text=response.text or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=calc_cost(model, prompt_tokens, completion_tokens),
            model=model,
        )
