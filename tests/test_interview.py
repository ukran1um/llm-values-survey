import json

import pytest

from llm_values.clients.mock import MockChatClient
from llm_values.interview import generate_questions, conduct_interview, QUESTION_GEN_PROMPT
from llm_values.types import Axis


AXIS = Axis(
    id="beatles_vs_stones",
    battery="pilot",
    description="Music canon: Beatles vs Stones.",
    labels=["beatles", "stones"],
)


def test_generate_questions_parses_json_list():
    client = MockChatClient(scripted=[json.dumps(["q1?", "q2?", "q3?"])])
    questions, cost = generate_questions(client, "claude-opus-4-7", AXIS, n_questions=3)
    assert questions == ["q1?", "q2?", "q3?"]
    assert cost == 0.001
    # interviewer prompt includes axis labels and description
    call = client.calls[0]
    assert call.model == "claude-opus-4-7"
    assert call.temperature == 1.0  # gpt-5.5 only accepts default 1.0; locked across providers
    user_msg = call.messages[0].content
    assert "beatles" in user_msg
    assert "stones" in user_msg
    assert AXIS.description in user_msg


def test_conduct_interview_returns_response_text():
    client = MockChatClient(scripted=["a long answer"])
    text, cost = conduct_interview(client, "gpt-5", ["q1?", "q2?"])
    assert text == "a long answer"
    assert cost == 0.001  # MockChatClient.fixed_cost_usd
    assert "q1?" in client.calls[0].messages[0].content
    assert "q2?" in client.calls[0].messages[0].content


def test_generate_questions_strips_markdown_fence():
    fenced = "```json\n[\"q1\", \"q2\"]\n```"
    client = MockChatClient(scripted=[fenced])
    qs, _ = generate_questions(client, "x", AXIS, n_questions=2)
    assert qs == ["q1", "q2"]


def test_generate_questions_handles_prose_preamble():
    text = 'Sure, here are three questions: ["q1", "q2", "q3"]'
    client = MockChatClient(scripted=[text])
    qs, _ = generate_questions(client, "x", AXIS, n_questions=3)
    assert qs == ["q1", "q2", "q3"]


def test_generate_questions_rejects_non_string_list():
    client = MockChatClient(scripted=["[1, 2, 3]"])
    with pytest.raises(ValueError, match="non-list-of-strings"):
        generate_questions(client, "x", AXIS, n_questions=3)
