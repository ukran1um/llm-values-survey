from __future__ import annotations
from itertools import permutations
from pathlib import Path

from .budget import Budget
from .interview import generate_questions, conduct_interview
from .judge import classify_transcript
from .models import get_client
from .storage import (
    save_transcript,
    save_classification,
    transcript_exists,
    load_transcript,
    judgment_path,
)
from .types import Axis, Transcript


def run_axis(
    axis: Axis,
    models: list[str],
    judges: list[str],
    n_reruns: int,
    data_dir: Path,
    budget: Budget,
    n_questions: int = 3,
) -> None:
    """Run all directed pairs × n_reruns for an axis. Skip pairs already on disk.

    For each (interviewer, interviewee, rerun):
      1. Interviewer generates questions about the axis (charged to budget).
      2. Interviewee answers (charged to budget).
      3. Transcript saved to disk.
      4. Each judge in `judges` (skipping any equal to interviewer or interviewee)
         classifies the transcript (charged to budget). Each judgment saved.

    Existing transcripts on disk are loaded and reused; missing judgments are
    still computed against the loaded transcript.
    """
    data_dir = Path(data_dir)
    for interviewer, interviewee in permutations(models, 2):
        for rerun in range(n_reruns):
            transcript = _ensure_transcript(
                axis, interviewer, interviewee, rerun, data_dir, budget, n_questions
            )
            for judge in judges:
                if judge == interviewer or judge == interviewee:
                    continue
                jpath = judgment_path(data_dir, axis.id, interviewer, interviewee, rerun, judge)
                if jpath.exists():
                    continue
                judge_client = get_client(judge)
                classification = classify_transcript(judge_client, judge, axis, transcript)
                budget.add(classification.cost_usd)
                save_classification(data_dir, classification)


def _ensure_transcript(
    axis: Axis,
    interviewer: str,
    interviewee: str,
    rerun: int,
    data_dir: Path,
    budget: Budget,
    n_questions: int,
) -> Transcript:
    if transcript_exists(data_dir, axis.id, interviewer, interviewee, rerun):
        return load_transcript(data_dir, axis.id, interviewer, interviewee, rerun)

    interviewer_client = get_client(interviewer)
    interviewee_client = get_client(interviewee)

    questions, interviewer_cost = generate_questions(interviewer_client, interviewer, axis, n_questions)
    budget.add(interviewer_cost)

    response_text, interviewee_cost = conduct_interview(interviewee_client, interviewee, questions)
    budget.add(interviewee_cost)

    transcript = Transcript(
        axis_id=axis.id,
        interviewer=interviewer,
        interviewee=interviewee,
        rerun=rerun,
        questions=questions,
        response=response_text,
        interviewer_cost_usd=interviewer_cost,
        interviewee_cost_usd=interviewee_cost,
    )
    save_transcript(data_dir, transcript)
    return transcript
