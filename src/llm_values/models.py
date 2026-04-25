from __future__ import annotations
from typing import Protocol

from .types import ChatMessage, ChatResponse


class ChatClient(Protocol):
    """Minimal provider surface used by the rest of the harness."""

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float = 1.0,
        max_tokens: int = 2000,
    ) -> ChatResponse: ...


# get_client is filled in once concrete clients exist (Task 11).
