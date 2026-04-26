from __future__ import annotations
from itertools import permutations
from pathlib import Path

from .budget import Budget
from .interview import conduct_pairwise_interview
from .models import get_client, model_provider, PROVIDER_EXTRAS
from .storage import (
    save_transcript,
    save_verdict,
    transcript_exists,
    verdict_exists,
)
from .types import Axis


def _extras_for(model: str) -> dict:
    return PROVIDER_EXTRAS.get(model_provider(model), {})


def run_axis(
    axis: Axis,
    models: list[str],
    n_reruns: int,
    data_dir: Path,
    budget: Budget,
) -> None:
    """Run all directed pairs × n_reruns for an axis.

    For each (interviewer, interviewee, rerun):
      1. Skip if both transcript AND verdict already exist on disk.
      2. Otherwise: conduct multi-turn pairwise interview ending in verdict.
      3. Save transcript and verdict; charge total cost to budget after both calls return.
    """
    data_dir = Path(data_dir)
    for interviewer, interviewee in permutations(models, 2):
        for rerun in range(n_reruns):
            if (
                transcript_exists(data_dir, axis.id, interviewer, interviewee, rerun)
                and verdict_exists(data_dir, axis.id, interviewer, interviewee, rerun)
            ):
                continue
            interviewer_client = get_client(interviewer)
            interviewee_client = get_client(interviewee)
            transcript, verdict = conduct_pairwise_interview(
                interviewer_client=interviewer_client,
                interviewer_model=interviewer,
                interviewer_extras=_extras_for(interviewer),
                interviewee_client=interviewee_client,
                interviewee_model=interviewee,
                interviewee_extras=_extras_for(interviewee),
                axis=axis,
                rerun=rerun,
            )
            # Save first so a BudgetExceeded mid-run preserves the work whose API cost was already paid.
            save_transcript(data_dir, transcript)
            save_verdict(data_dir, verdict)
            # Total per pair: question-gen + interviewee answer + verdict-issuing call.
            total_cost = (
                transcript.interviewer_cost_usd
                + transcript.interviewee_cost_usd
                + verdict.cost_usd
            )
            budget.add(total_cost)
