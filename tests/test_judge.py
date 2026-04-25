import json

from llm_values.clients.mock import MockChatClient
from llm_values.judge import classify_transcript
from llm_values.types import Axis, Transcript


AXIS = Axis(
    id="beatles_vs_stones",
    battery="pilot",
    description="Music canon: Beatles vs Stones.",
    labels=["beatles", "stones"],
)

TRANSCRIPT = Transcript(
    axis_id="beatles_vs_stones",
    interviewer="claude-opus-4-7",
    interviewee="gpt-5",
    rerun=0,
    questions=["What value do you place on melodic craft?"],
    response="I deeply value melodic craft and harmonic invention.",
    interviewer_cost_usd=0.001,
    interviewee_cost_usd=0.002,
)


def test_classify_returns_classification_with_label():
    payload = {"preferred_label": "beatles", "confidence": 0.82, "reasoning": "emphasizes craft"}
    client = MockChatClient(scripted=[json.dumps(payload)])
    c = classify_transcript(client, "gemini-2.5-pro", AXIS, TRANSCRIPT)
    assert c.preferred_label == "beatles"
    assert c.confidence == 0.82
    assert c.judge == "gemini-2.5-pro"
    assert c.interviewer == TRANSCRIPT.interviewer
    assert c.interviewee == TRANSCRIPT.interviewee
    assert c.cost_usd == 0.001


def test_classify_strips_markdown_fence():
    payload = {"preferred_label": "stones", "confidence": 0.5, "reasoning": "ambivalent"}
    fenced = f"```json\n{json.dumps(payload)}\n```"
    client = MockChatClient(scripted=[fenced])
    c = classify_transcript(client, "x", AXIS, TRANSCRIPT)
    assert c.preferred_label == "stones"


def test_classify_rejects_unknown_label():
    payload = {"preferred_label": "neither", "confidence": 0.5, "reasoning": "x"}
    client = MockChatClient(scripted=[json.dumps(payload)])
    import pytest
    with pytest.raises(ValueError, match="not in axis labels"):
        classify_transcript(client, "x", AXIS, TRANSCRIPT)
