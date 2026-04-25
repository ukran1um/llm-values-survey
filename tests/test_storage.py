from pathlib import Path

from llm_values.storage import (
    save_transcript,
    save_classification,
    load_transcript,
    transcript_exists,
    transcript_path,
)
from llm_values.types import Transcript, Classification


def make_transcript() -> Transcript:
    return Transcript(
        axis_id="beatles_vs_stones",
        interviewer="claude-opus-4-7",
        interviewee="gpt-5",
        rerun=0,
        questions=["q1", "q2"],
        response="answer text",
        interviewer_cost_usd=0.01,
        interviewee_cost_usd=0.02,
    )


def test_transcript_path_format(tmp_path: Path):
    t = make_transcript()
    p = transcript_path(tmp_path, t.axis_id, t.interviewer, t.interviewee, t.rerun)
    assert p.parent == tmp_path / "raw" / "interviews" / "beatles_vs_stones"
    assert p.name == "claude-opus-4-7__gpt-5__r0.json"


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


def test_save_classification(tmp_path: Path):
    t = make_transcript()
    c = Classification(
        axis_id=t.axis_id,
        judge="gemini-2.5-pro",
        interviewer=t.interviewer,
        interviewee=t.interviewee,
        rerun=t.rerun,
        preferred_label="beatles",
        confidence=0.7,
        reasoning="emphasis on melodic craft",
        cost_usd=0.005,
    )
    p = save_classification(tmp_path, c)
    assert p.exists()
    assert p.parent == tmp_path / "raw" / "judgments" / "beatles_vs_stones"
    assert "gemini-2.5-pro" in p.name
