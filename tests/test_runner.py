from pathlib import Path
from unittest.mock import patch

from llm_values.budget import Budget
from llm_values.runner import run_axis
from llm_values.types import Axis, VerdictFormat, Transcript, Turn, Verdict


AXIS = Axis(
    id="care_vs_fairness",
    battery="mfq",
    description="d",
    verdict_format=VerdictFormat(type="binary", options=["care", "fairness"]),
    max_turns=2,
)


def fake_interview(*, interviewer_model, interviewee_model, rerun, axis, **kwargs):
    t = Transcript(
        axis_id=axis.id,
        interviewer=interviewer_model,
        interviewee=interviewee_model,
        rerun=rerun,
        turns=[Turn(question="q", answer="a")],
        interviewer_cost_usd=0.001,
        interviewee_cost_usd=0.001,
    )
    v = Verdict(
        axis_id=axis.id,
        interviewer=interviewer_model,
        interviewee=interviewee_model,
        rerun=rerun,
        verdict_type="binary",
        binary_choice="care",
        confidence=0.7,
        reasoning="r",
        key_quote="q",
        n_turns_used=1,
        cost_usd=0.001,
    )
    return t, v


def test_run_axis_writes_transcripts_and_verdicts(tmp_path: Path):
    budget = Budget(state_path=tmp_path / "budget.json", cap_usd=10.0)
    with patch("llm_values.runner.conduct_pairwise_interview", side_effect=fake_interview):
        with patch("llm_values.runner.get_client", return_value=None):
            with patch("llm_values.runner._extras_for", return_value={}):
                run_axis(AXIS, models=["A", "B"], n_reruns=1, data_dir=tmp_path, budget=budget)

    assert (tmp_path / "raw" / "interviews" / AXIS.id / "A__B__r0.json").exists()
    assert (tmp_path / "raw" / "interviews" / AXIS.id / "B__A__r0.json").exists()
    assert (tmp_path / "raw" / "verdicts" / AXIS.id / "A__B__r0.json").exists()
    assert (tmp_path / "raw" / "verdicts" / AXIS.id / "B__A__r0.json").exists()


def test_run_axis_skips_when_both_exist(tmp_path: Path):
    budget = Budget(state_path=tmp_path / "budget.json", cap_usd=10.0)
    call_count = [0]

    def counting_fake(**kwargs):
        call_count[0] += 1
        return fake_interview(**kwargs)

    with patch("llm_values.runner.conduct_pairwise_interview", side_effect=counting_fake):
        with patch("llm_values.runner.get_client", return_value=None):
            with patch("llm_values.runner._extras_for", return_value={}):
                run_axis(AXIS, models=["A", "B"], n_reruns=1, data_dir=tmp_path, budget=budget)
                first_count = call_count[0]
                run_axis(AXIS, models=["A", "B"], n_reruns=1, data_dir=tmp_path, budget=budget)
                assert call_count[0] == first_count  # no new calls


def test_run_axis_parallel_writes_correct_files(tmp_path: Path):
    """Parallel execution should produce the same files as serial."""
    budget = Budget(state_path=tmp_path / "budget.json", cap_usd=10.0)
    with patch("llm_values.runner.conduct_pairwise_interview", side_effect=fake_interview):
        with patch("llm_values.runner.get_client", return_value=None):
            with patch("llm_values.runner._extras_for", return_value={}):
                run_axis(AXIS, models=["A", "B", "C"], n_reruns=1, data_dir=tmp_path, budget=budget, concurrency=4)

    # 3 models × 2 partners = 6 directed pairs
    assert len(list((tmp_path / "raw" / "interviews" / AXIS.id).glob("*.json"))) == 6
    assert len(list((tmp_path / "raw" / "verdicts" / AXIS.id).glob("*.json"))) == 6


def test_extras_for_uses_model_override_when_present():
    from llm_values.runner import _extras_for
    # grok-4.20 has an empty-dict override, suppressing the xai provider default
    assert _extras_for("grok-4.20") == {}
    # claude-opus-4-7 has no override, falls through to anthropic provider default (which is also empty {})
    assert _extras_for("claude-opus-4-7") == {}


def test_extras_for_uses_provider_default_when_no_override():
    from llm_values.runner import _extras_for
    # gpt-5.5 has no override, gets openai provider extras
    assert _extras_for("gpt-5.5-2026-04-23") != {}
    assert "reasoning_effort" in _extras_for("gpt-5.5-2026-04-23")
