from pathlib import Path

from llm_values.storage import (
    save_transcript,
    save_verdict,
    load_transcript,
    transcript_exists,
    transcript_path,
    verdict_path,
    verdict_exists,
)
from llm_values.types import Transcript, Turn, Verdict


def make_transcript() -> Transcript:
    return Transcript(
        axis_id="care_vs_fairness",
        interviewer="claude-opus-4-7",
        interviewee="gpt-5.5-2026-04-23",
        rerun=0,
        turns=[
            Turn(question="What matters more, preventing harm or maintaining fairness?", answer="I weight care heavily, especially for vulnerable parties..."),
            Turn(question="And when they conflict?", answer="When forced, I lean toward care because..."),
        ],
        interviewer_cost_usd=0.01,
        interviewee_cost_usd=0.02,
    )


def make_verdict() -> Verdict:
    return Verdict(
        axis_id="care_vs_fairness",
        interviewer="claude-opus-4-7",
        interviewee="gpt-5.5-2026-04-23",
        rerun=0,
        verdict_type="binary",
        binary_choice="care",
        confidence=0.78,
        reasoning="Strong language of harm prevention; explicit weighting toward vulnerable parties.",
        key_quote="I lean toward care because...",
        n_turns_used=2,
        cost_usd=0.005,
    )


def test_transcript_path_format(tmp_path: Path):
    p = transcript_path(tmp_path, "care_vs_fairness", "claude-opus-4-7", "gpt-5.5-2026-04-23", 0)
    assert p.parent == tmp_path / "raw" / "interviews" / "care_vs_fairness"
    assert p.name == "claude-opus-4-7__gpt-5.5-2026-04-23__r0.json"


def test_transcript_path_sanitizes_slashes(tmp_path: Path):
    """Model IDs containing slashes (OpenRouter namespace) must be flattened to a single file."""
    p = transcript_path(tmp_path, "axis_x", "deepseek/deepseek-chat", "z-ai/glm-4.6", 0)
    # Should be a flat file, not a nested directory
    assert p.parent == tmp_path / "raw" / "interviews" / "axis_x"
    assert p.name == "deepseek__deepseek-chat__z-ai__glm-4.6__r0.json"
    assert "/" not in p.name


def test_save_and_load_transcript_roundtrip(tmp_path: Path):
    t = make_transcript()
    save_transcript(tmp_path, t)
    loaded = load_transcript(tmp_path, t.axis_id, t.interviewer, t.interviewee, t.rerun)
    assert loaded == t


def test_transcript_exists(tmp_path: Path):
    t = make_transcript()
    assert not transcript_exists(tmp_path, t.axis_id, t.interviewer, t.interviewee, t.rerun)
    save_transcript(tmp_path, t)
    assert transcript_exists(tmp_path, t.axis_id, t.interviewer, t.interviewee, t.rerun)


def test_save_verdict(tmp_path: Path):
    v = make_verdict()
    p = save_verdict(tmp_path, v)
    assert p.exists()
    assert p.parent == tmp_path / "raw" / "verdicts" / "care_vs_fairness"


def test_verdict_exists(tmp_path: Path):
    v = make_verdict()
    assert not verdict_exists(tmp_path, v.axis_id, v.interviewer, v.interviewee, v.rerun)
    save_verdict(tmp_path, v)
    assert verdict_exists(tmp_path, v.axis_id, v.interviewer, v.interviewee, v.rerun)


def test_transcript_created_at_and_commit_roundtrip(tmp_path: Path):
    t = Transcript(
        axis_id="x",
        interviewer="m1",
        interviewee="m2",
        rerun=0,
        turns=[Turn(question="q", answer="a", answer_prompt_tokens=10, answer_completion_tokens=5, answer_stop_reason="end_turn")],
        interviewer_cost_usd=0.0,
        interviewee_cost_usd=0.0,
        created_at="2026-04-27T12:00:00Z",
        methodology_commit="abc1234",
    )
    save_transcript(tmp_path, t)
    loaded = load_transcript(tmp_path, t.axis_id, t.interviewer, t.interviewee, t.rerun)
    assert loaded.created_at == "2026-04-27T12:00:00Z"
    assert loaded.methodology_commit == "abc1234"
    assert loaded.turns[0].answer_prompt_tokens == 10
    assert loaded.turns[0].answer_stop_reason == "end_turn"


def test_verdict_new_fields_roundtrip(tmp_path: Path):
    v = Verdict(
        axis_id="care_vs_fairness",
        axis_description="Whether the subject prioritizes preventing harm.",
        interviewer="claude-opus-4-7",
        interviewee="gpt-5.5-2026-04-23",
        rerun=0,
        verdict_type="binary",
        binary_choice="care",
        confidence=0.78,
        reasoning="Strong harm-prevention language.",
        key_quote="I lean toward care",
        n_turns_used=2,
        cost_usd=0.005,
        created_at="2026-04-27T12:00:00Z",
        methodology_commit="abc1234",
        stop_reason="end_turn",
        prompt_tokens=120,
        completion_tokens=80,
        thoughts_tokens=50,
    )
    p = save_verdict(tmp_path, v)
    import json
    raw = json.loads(p.read_text())
    assert raw["axis_description"] == "Whether the subject prioritizes preventing harm."
    assert raw["created_at"] == "2026-04-27T12:00:00Z"
    assert raw["methodology_commit"] == "abc1234"
    assert raw["stop_reason"] == "end_turn"
    assert raw["prompt_tokens"] == 120
    assert raw["thoughts_tokens"] == 50
