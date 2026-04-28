from unittest.mock import MagicMock, patch

import pytest

from llm_values.clients.anthropic_client import AnthropicChatClient
from llm_values.types import ChatMessage


@patch("llm_values.clients.anthropic_client.Anthropic")
def test_anthropic_chat_constructs_request(MockAnthropic):
    fake_msg = MagicMock()
    fake_msg.content = [MagicMock(text="hello world")]
    fake_msg.usage = MagicMock(input_tokens=42, output_tokens=17)
    MockAnthropic.return_value.messages.create.return_value = fake_msg

    client = AnthropicChatClient(api_key="test-key")
    response = client.chat(
        model="claude-opus-4-7-20260416",
        messages=[ChatMessage(role="user", content="hi")],
        temperature=0.7,
        max_tokens=500,
    )

    assert response.text == "hello world"
    assert response.prompt_tokens == 42
    assert response.completion_tokens == 17
    assert response.model == "claude-opus-4-7-20260416"

    MockAnthropic.return_value.messages.create.assert_called_once()
    kwargs = MockAnthropic.return_value.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-4-7-20260416"
    assert "temperature" not in kwargs
    assert kwargs["max_tokens"] == 500
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


@patch("llm_values.clients.anthropic_client.Anthropic")
def test_anthropic_rejects_multiple_system_messages(MockAnthropic):
    client = AnthropicChatClient(api_key="test-key")
    with pytest.raises(ValueError, match="at most 1 system message"):
        client.chat(
            model="claude-opus-4-7-20260416",
            messages=[
                ChatMessage(role="system", content="first"),
                ChatMessage(role="system", content="second"),
                ChatMessage(role="user", content="hi"),
            ],
        )


@patch("llm_values.clients.anthropic_client.Anthropic")
def test_anthropic_extracts_system_message(MockAnthropic):
    fake_msg = MagicMock()
    fake_msg.content = [MagicMock(text="ok")]
    fake_msg.usage = MagicMock(input_tokens=10, output_tokens=5)
    MockAnthropic.return_value.messages.create.return_value = fake_msg

    client = AnthropicChatClient(api_key="test-key")
    client.chat(
        model="claude-opus-4-7-20260416",
        messages=[
            ChatMessage(role="system", content="be terse"),
            ChatMessage(role="user", content="hi"),
        ],
    )

    kwargs = MockAnthropic.return_value.messages.create.call_args.kwargs
    assert kwargs["system"] == "be terse"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


@patch("llm_values.clients.anthropic_client.Anthropic")
def test_anthropic_accepts_extras_param(MockAnthropic):
    fake_msg = MagicMock()
    fake_msg.content = [MagicMock(text="hi")]
    fake_msg.usage = MagicMock(input_tokens=5, output_tokens=3)
    MockAnthropic.return_value.messages.create.return_value = fake_msg

    client = AnthropicChatClient(api_key="test-key")
    response = client.chat(
        model="claude-opus-4-7-20260416",
        messages=[ChatMessage(role="user", content="hi")],
        extras={"some_flag": True},  # ignored, but must not crash
    )
    assert response.text == "hi"
