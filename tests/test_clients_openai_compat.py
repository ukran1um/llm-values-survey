import httpx
import pytest

from llm_values.clients.openai_compat import OpenAICompatClient
from llm_values.types import ChatMessage


def fake_handler(request: httpx.Request) -> httpx.Response:
    assert request.method == "POST"
    assert "/chat/completions" in str(request.url)
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "mock answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def test_openai_compat_chat_returns_response():
    transport = httpx.MockTransport(fake_handler)
    http = httpx.Client(transport=transport)
    client = OpenAICompatClient(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        http=http,
    )

    response = client.chat(
        model="gpt-5",
        messages=[ChatMessage(role="user", content="hello")],
        temperature=0.5,
        max_tokens=100,
    )

    assert response.text == "mock answer"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 5
    assert response.model == "gpt-5"


def test_openai_compat_raises_on_http_error():
    def err(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    http = httpx.Client(transport=httpx.MockTransport(err))
    client = OpenAICompatClient(base_url="https://api.example.com/v1", api_key="k", http=http)
    with pytest.raises(httpx.HTTPStatusError):
        client.chat(model="x", messages=[ChatMessage(role="user", content="hi")])
