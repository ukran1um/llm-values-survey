import json
from pathlib import Path

from llm_values.budget import Budget
from llm_values.clients.mock import MockChatClient
from llm_values.runner import run_axis
from llm_values.types import Axis


AXIS = Axis(
    id="beatles_vs_stones",
    battery="pilot",
    description="Music canon: Beatles vs Stones.",
    labels=["beatles", "stones"],
)


def make_judge_payload(label="beatles", conf=0.7):
    return json.dumps({"preferred_label": label, "confidence": conf, "reasoning": "r"})


def script_for(n_pairs, n_reruns, n_judges):
    """Each (pair, rerun) needs: 1 question_gen + 1 answer + n_judges judgments."""
    out = []
    for _ in range(n_pairs * n_reruns):
        out.append(json.dumps(["q1", "q2"]))   # question gen
        out.append("interview answer")          # answer
        for _ in range(n_judges):
            out.append(make_judge_payload())    # judge
    return out


def test_run_axis_writes_transcripts_and_judgments(tmp_path: Path, monkeypatch):
    models = ["A", "B"]   # 2 directed pairs (A→B, B→A)
    judges = ["C"]
    n_reruns = 1
    client = MockChatClient(scripted=script_for(n_pairs=2, n_reruns=1, n_judges=1))

    # patch get_client to always return our mock
    import llm_values.runner as r
    monkeypatch.setattr(r, "get_client", lambda model: client)

    budget = Budget(state_path=tmp_path / "budget.json", cap_usd=10.0)
    run_axis(AXIS, models=models, judges=judges, n_reruns=n_reruns, data_dir=tmp_path, budget=budget)

    # 2 transcripts written
    assert (tmp_path / "raw" / "interviews" / AXIS.id / "A__B__r0.json").exists()
    assert (tmp_path / "raw" / "interviews" / AXIS.id / "B__A__r0.json").exists()
    # 2 judgments written (1 per transcript × 1 judge)
    assert (tmp_path / "raw" / "judgments" / AXIS.id / "A__B__r0__jC.json").exists()
    assert (tmp_path / "raw" / "judgments" / AXIS.id / "B__A__r0__jC.json").exists()


def test_run_axis_skips_existing(tmp_path: Path, monkeypatch):
    models = ["A", "B"]
    judges = ["C"]
    client = MockChatClient(scripted=script_for(n_pairs=2, n_reruns=1, n_judges=1))
    import llm_values.runner as r
    monkeypatch.setattr(r, "get_client", lambda model: client)
    budget = Budget(state_path=tmp_path / "budget.json", cap_usd=10.0)

    # First run
    run_axis(AXIS, models=models, judges=judges, n_reruns=1, data_dir=tmp_path, budget=budget)
    calls_first = len(client.calls)

    # Second run — should skip everything
    run_axis(AXIS, models=models, judges=judges, n_reruns=1, data_dir=tmp_path, budget=budget)
    assert len(client.calls) == calls_first  # no new calls


def test_run_axis_skips_judge_equal_to_interviewer_or_interviewee(tmp_path: Path, monkeypatch):
    models = ["A", "B"]
    judges = ["A", "C"]   # A is also a model — must not judge any pair it's part of
    # For directed pairs (A→B, B→A), only judge C qualifies for both
    client = MockChatClient(scripted=script_for(n_pairs=2, n_reruns=1, n_judges=1))
    import llm_values.runner as r
    monkeypatch.setattr(r, "get_client", lambda model: client)
    budget = Budget(state_path=tmp_path / "budget.json", cap_usd=10.0)

    run_axis(AXIS, models=models, judges=judges, n_reruns=1, data_dir=tmp_path, budget=budget)

    # Only judge C produced files
    judg_dir = tmp_path / "raw" / "judgments" / AXIS.id
    files = sorted(p.name for p in judg_dir.iterdir())
    assert all("jC" in name for name in files)
    assert not any("jA" in name for name in files)


def test_run_axis_reuses_transcript_when_judgments_missing(tmp_path: Path, monkeypatch):
    """Existing transcript on disk should be loaded; only missing judgments are computed."""
    models = ["A", "B"]
    # First run: no judges, only transcripts get written
    client_first = MockChatClient(scripted=script_for(n_pairs=2, n_reruns=1, n_judges=0))
    import llm_values.runner as r
    monkeypatch.setattr(r, "get_client", lambda model: client_first)
    budget = Budget(state_path=tmp_path / "budget.json", cap_usd=10.0)
    run_axis(AXIS, models=models, judges=[], n_reruns=1, data_dir=tmp_path, budget=budget)

    # Sanity check: transcripts exist, no judgments yet
    assert (tmp_path / "raw" / "interviews" / AXIS.id / "A__B__r0.json").exists()
    assert not (tmp_path / "raw" / "judgments" / AXIS.id).exists()

    # Second run: same models, now with judge C — should NOT do question_gen or
    # answer calls again, only judge calls (one per existing transcript)
    client_second = MockChatClient(
        scripted=[make_judge_payload(), make_judge_payload()]  # 2 judge calls only
    )
    monkeypatch.setattr(r, "get_client", lambda model: client_second)
    run_axis(AXIS, models=models, judges=["C"], n_reruns=1, data_dir=tmp_path, budget=budget)

    # Only 2 calls happened on the second run (one judgment per existing transcript)
    assert len(client_second.calls) == 2
    assert (tmp_path / "raw" / "judgments" / AXIS.id / "A__B__r0__jC.json").exists()
    assert (tmp_path / "raw" / "judgments" / AXIS.id / "B__A__r0__jC.json").exists()
