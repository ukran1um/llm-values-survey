from unittest.mock import MagicMock, patch

from llm_values.clients.google_client import GoogleChatClient
from llm_values.types import ChatMessage


@patch("llm_values.clients.google_client.genai")
def test_google_chat_constructs_request(mock_genai):
    fake_response = MagicMock()
    fake_response.text = "google answer"
    fake_response.usage_metadata = MagicMock(prompt_token_count=30, candidates_token_count=20)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = fake_response
    mock_genai.Client.return_value = mock_client

    client = GoogleChatClient(api_key="key")
    response = client.chat(
        model="gemini-2.5-pro",
        messages=[ChatMessage(role="user", content="hi")],
        temperature=0.5,
        max_tokens=100,
    )

    assert response.text == "google answer"
    assert response.prompt_tokens == 30
    assert response.completion_tokens == 20
    assert response.model == "gemini-2.5-pro"
    mock_client.models.generate_content.assert_called_once()
