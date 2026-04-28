from __future__ import annotations
from dataclasses import dataclass, field

from ..types import ChatMessage, ChatResponse


@dataclass
class MockCall:
    model: str
    messages: list[ChatMessage]
    temperature: float
    max_tokens: int
    extras: dict | None = None


@dataclass
class MockChatClient:
    """Test fixture. Returns scripted responses; records every call.

    Usage:
        client = MockChatClient(scripted=["first response", "second response"])
        response = client.chat("any-model", [...])  # → "first response"
        client.calls  # → [MockCall(...)]
    """

    scripted: list[str] = field(default_factory=list)
    calls: list[MockCall] = field(default_factory=list)
    fixed_prompt_tokens: int = 100
    fixed_completion_tokens: int = 200
    fixed_cost_usd: float = 0.001
    fixed_stop_reason: str | None = "end_turn"
    fixed_thoughts_tokens: int = 0

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float = 1.0,
        max_tokens: int = 2000,
        extras: dict | None = None,
    ) -> ChatResponse:
        self.calls.append(
            MockCall(
                model=model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                extras=extras,
            )
        )
        if not self.scripted:
            raise RuntimeError(f"MockChatClient ran out of scripted responses (call #{len(self.calls)})")
        text = self.scripted.pop(0)
        return ChatResponse(
            text=text,
            prompt_tokens=self.fixed_prompt_tokens,
            completion_tokens=self.fixed_completion_tokens,
            thoughts_tokens=self.fixed_thoughts_tokens,
            cost_usd=self.fixed_cost_usd,
            model=model,
            stop_reason=self.fixed_stop_reason,
        )
