from unittest.mock import MagicMock, patch

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
        model="claude-opus-4-7",
        messages=[ChatMessage(role="user", content="hi")],
        temperature=0.7,
        max_tokens=500,
    )

    assert response.text == "hello world"
    assert response.prompt_tokens == 42
    assert response.completion_tokens == 17
    assert response.model == "claude-opus-4-7"

    MockAnthropic.return_value.messages.create.assert_called_once()
    kwargs = MockAnthropic.return_value.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-4-7"
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 500
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
