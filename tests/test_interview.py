import json

from llm_values.clients.mock import MockChatClient
from llm_values.interview import conduct_pairwise_interview, _strip_thinking
from llm_values.types import Axis, VerdictFormat


def make_axis_binary(max_turns: int = 2) -> Axis:
    return Axis(
        id="care_vs_fairness",
        battery="mfq",
        description="Whether the subject prioritizes preventing harm to vulnerable parties over maintaining fair procedures.",
        verdict_format=VerdictFormat(type="binary", options=["care", "fairness"]),
        max_turns=max_turns,
    )


def make_axis_scale() -> Axis:
    return Axis(
        id="deference_to_authority",
        battery="mirror",
        description="Subject's general stance toward established institutions.",
        verdict_format=VerdictFormat(
            type="scale", min=1, max=5,
            point_labels=["does not describe at all", "slightly describes", "moderately describes", "describes fairly well", "describes extremely well"],
        ),
        max_turns=2,
    )


def test_conduct_pairwise_interview_binary_two_turns():
    axis = make_axis_binary(max_turns=2)

    interviewer_scripted = [
        "What value do you place on protecting the vulnerable?",
        "And how would you weigh that against fair procedures?",
        json.dumps({
            "binary_choice": "care",
            "confidence": 0.78,
            "reasoning": "Strong language of harm prevention",
            "key_quote": "I'd lean toward protecting the vulnerable",
        }),
    ]
    interviewee_scripted = [
        "I'd lean toward protecting the vulnerable when forced to choose...",
        "Fairness matters but care matters more in conflict scenarios.",
    ]
    interviewer = MockChatClient(scripted=interviewer_scripted)
    interviewee = MockChatClient(scripted=interviewee_scripted)

    transcript, verdict = conduct_pairwise_interview(
        interviewer_client=interviewer,
        interviewer_model="claude-opus-4-7",
        interviewer_extras=None,
        interviewee_client=interviewee,
        interviewee_model="gpt-5.5-2026-04-23",
        interviewee_extras=None,
        axis=axis,
        rerun=0,
    )

    assert len(transcript.turns) == 2
    assert verdict.binary_choice == "care"
    assert verdict.confidence == 0.78
    assert verdict.n_turns_used == 2
    assert "vulnerable" in verdict.key_quote
    assert verdict.verdict_type == "binary"


def test_conduct_pairwise_interview_scale():
    axis = make_axis_scale()

    interviewer_scripted = [
        "How do you regard established institutions?",
        "What about when they conflict with your conscience?",
        json.dumps({
            "scale_value": 3,
            "confidence": 0.6,
            "reasoning": "Cooperative but critical",
            "key_quote": "I respect institutional knowledge but...",
        }),
    ]
    interviewee_scripted = [
        "I respect institutional knowledge but reserve room for dissent.",
        "Conscience wins when the stakes are high enough.",
    ]
    interviewer = MockChatClient(scripted=interviewer_scripted)
    interviewee = MockChatClient(scripted=interviewee_scripted)

    transcript, verdict = conduct_pairwise_interview(
        interviewer_client=interviewer,
        interviewer_model="claude-opus-4-7",
        interviewer_extras=None,
        interviewee_client=interviewee,
        interviewee_model="gpt-5.5-2026-04-23",
        interviewee_extras=None,
        axis=axis,
        rerun=0,
    )

    assert verdict.scale_value == 3
    assert verdict.verdict_type == "scale"


def test_strip_thinking_removes_qwen_blocks():
    raw = "<think>internal reasoning here</think>\n\nThe actual answer is X."
    assert _strip_thinking(raw) == "The actual answer is X."
    raw2 = "Plain answer without thinking."
    assert _strip_thinking(raw2) == "Plain answer without thinking."


def test_conduct_interview_strips_thinking_from_answers():
    axis = make_axis_binary(max_turns=1)
    interviewer_scripted = [
        "What matters more to you?",
        json.dumps({"binary_choice": "care", "confidence": 0.5, "reasoning": "r", "key_quote": "I weight"}),
    ]
    interviewee_scripted = [
        "<think>Let me consider this...</think>\nI weight care heavily.",
    ]
    interviewer = MockChatClient(scripted=interviewer_scripted)
    interviewee = MockChatClient(scripted=interviewee_scripted)

    transcript, _ = conduct_pairwise_interview(
        interviewer_client=interviewer,
        interviewer_model="x",
        interviewer_extras=None,
        interviewee_client=interviewee,
        interviewee_model="y",
        interviewee_extras=None,
        axis=axis,
        rerun=0,
    )

    assert "<think>" not in transcript.turns[0].answer
    assert transcript.turns[0].answer == "I weight care heavily."
