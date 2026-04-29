from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import permutations
from pathlib import Path
from typing import Iterable

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .budget import Budget, BudgetExceeded
from .interview import conduct_pairwise_interview
from .models import get_client, model_provider, PROVIDER_EXTRAS, MODEL_EXTRAS_OVERRIDE
from .storage import (
    save_transcript,
    save_verdict,
    transcript_exists,
    verdict_exists,
)
from .types import Axis


log = logging.getLogger(__name__)


def _extras_for(model: str) -> dict:
    if model in MODEL_EXTRAS_OVERRIDE:
        return MODEL_EXTRAS_OVERRIDE[model]
    return PROVIDER_EXTRAS.get(model_provider(model), {})


# Retry transient errors (rate limits, timeouts, 5xx) with exponential backoff.
# Does NOT retry: BudgetExceeded, ValidationError, KeyError, ValueError (those are real bugs or genuine refusals).
@retry(
    retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def _run_one_cell(axis: Axis, interviewer: str, interviewee: str, rerun: int, data_dir: Path):
    """Run one (interviewer, interviewee, axis, rerun) cell. Returns (transcript, verdict, total_cost)
    or None if the cell is already on disk (checkpoint hit)."""
    if transcript_exists(data_dir, axis.id, interviewer, interviewee, rerun) and verdict_exists(
        data_dir, axis.id, interviewer, interviewee, rerun
    ):
        return None

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
    save_transcript(data_dir, transcript)
    save_verdict(data_dir, verdict)
    total_cost = (
        transcript.interviewer_cost_usd + transcript.interviewee_cost_usd + verdict.cost_usd
    )
    return (transcript, verdict, total_cost)


def run_axis(
    axis: Axis,
    models: list[str],
    n_reruns: int,
    data_dir: Path,
    budget: Budget,
    concurrency: int = 8,
) -> dict:
    """Run all directed pairs × n_reruns for an axis in parallel.

    Returns a summary dict: {ok, skipped, failed, total_cost}.
    """
    data_dir = Path(data_dir)
    cells = [
        (axis, interviewer, interviewee, rerun)
        for interviewer, interviewee in permutations(models, 2)
        for rerun in range(n_reruns)
    ]
    summary = {"ok": 0, "skipped": 0, "failed": 0, "total_cost": 0.0}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_run_one_cell, ax, i, e, r, data_dir): (ax, i, e, r)
            for ax, i, e, r in cells
        }
        for future in as_completed(futures):
            ax, i, e, r = futures[future]
            try:
                result = future.result()
            except BudgetExceeded:
                # Re-raise — this halts the whole run
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception as exc:
                log.exception(f"cell failed: {ax.id} {i}->{e} r{r}: {exc}")
                summary["failed"] += 1
                continue

            if result is None:
                summary["skipped"] += 1
                continue

            transcript, verdict, cost = result
            try:
                budget.add(cost)
            except BudgetExceeded:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            summary["ok"] += 1
            summary["total_cost"] += cost

    return summary


def run_battery(
    axes: list[Axis],
    models: list[str],
    n_reruns: int,
    data_dir: Path,
    budget: Budget,
    concurrency: int = 8,
) -> dict:
    """Run all axes in a battery, in sequence. Each axis is internally parallelized."""
    overall = {"ok": 0, "skipped": 0, "failed": 0, "total_cost": 0.0}
    for ax in axes:
        log.info(f"running axis {ax.id} ({ax.battery})")
        result = run_axis(ax, models, n_reruns, data_dir, budget, concurrency)
        for k in overall:
            overall[k] += result[k]
        log.info(f"axis {ax.id}: {result}")
    return overall
