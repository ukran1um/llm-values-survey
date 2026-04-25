from __future__ import annotations
from anthropic import Anthropic

from ..types import ChatMessage, ChatResponse
from ..pricing import calc_cost


class AnthropicChatClient:
    """ChatClient implementation using the Anthropic SDK."""

    def __init__(self, api_key: str):
        self._client = Anthropic(api_key=api_key)

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float = 1.0,
        max_tokens: int = 2000,
    ) -> ChatResponse:
        sdk_messages = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        system = next((m.content for m in messages if m.role == "system"), None)
        kwargs = dict(
            model=model,
            messages=sdk_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if system:
            kwargs["system"] = system
        msg = self._client.messages.create(**kwargs)
        text = "".join(block.text for block in msg.content if hasattr(block, "text"))
        prompt_tokens = msg.usage.input_tokens
        completion_tokens = msg.usage.output_tokens
        return ChatResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=calc_cost(model, prompt_tokens, completion_tokens),
            model=model,
        )
