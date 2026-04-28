from unittest.mock import MagicMock, patch

from llm_values.clients.google_client import GoogleChatClient
from llm_values.types import ChatMessage


@patch("llm_values.clients.google_client.genai")
def test_google_chat_constructs_request(mock_genai):
    fake_response = MagicMock()
    fake_response.text = "google answer"
    fake_response.usage_metadata = MagicMock(
        prompt_token_count=30,
        candidates_token_count=20,
        thoughts_token_count=50,
    )
    fake_candidate = MagicMock()
    fake_candidate.finish_reason = "STOP"
    fake_response.candidates = [fake_candidate]
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


@patch("llm_values.clients.google_client.genai")
def test_google_passes_thinking_budget_with_overhead(mock_genai):
    fake_response = MagicMock()
    fake_response.text = "ok"
    fake_response.usage_metadata = MagicMock(
        prompt_token_count=10,
        candidates_token_count=5,
        thoughts_token_count=0,
    )
    fake_candidate = MagicMock()
    fake_candidate.finish_reason = "STOP"
    fake_response.candidates = [fake_candidate]
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = fake_response
    mock_genai.Client.return_value = mock_client

    client = GoogleChatClient(api_key="key")
    client.chat(
        model="gemini-2.5-pro",
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=1000,
    )

    kwargs = mock_client.models.generate_content.call_args.kwargs
    # Gemini 2.5 Pro requires thinking_budget > 0; we use 512 as the minimum overhead.
    # max_output_tokens is bumped by the same overhead so the requested max_tokens still
    # represents the actual usable output budget.
    assert kwargs["config"]["thinking_config"] == {"thinking_budget": 512}
    assert kwargs["config"]["max_output_tokens"] == 1000 + 512
