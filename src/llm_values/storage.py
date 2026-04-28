from __future__ import annotations
from pathlib import Path

from .types import Transcript, Verdict


def _safe(model_id: str) -> str:
    """Sanitize a model ID for use as a filename component.
    Models from OpenRouter and similar providers use slash-namespacing
    (e.g. deepseek/deepseek-chat, z-ai/glm-4.6). Replace / with __ to
    keep everything in one directory level."""
    return model_id.replace("/", "__")


def transcript_path(
    data_dir: Path,
    axis_id: str,
    interviewer: str,
    interviewee: str,
    rerun: int,
) -> Path:
    return (
        Path(data_dir)
        / "raw" / "interviews" / axis_id
        / f"{_safe(interviewer)}__{_safe(interviewee)}__r{rerun}.json"
    )


def verdict_path(
    data_dir: Path,
    axis_id: str,
    interviewer: str,
    interviewee: str,
    rerun: int,
) -> Path:
    """Verdict path mirrors transcript path 1:1 — every transcript has exactly one verdict."""
    return (
        Path(data_dir)
        / "raw" / "verdicts" / axis_id
        / f"{_safe(interviewer)}__{_safe(interviewee)}__r{rerun}.json"
    )


def save_transcript(data_dir: Path, t: Transcript) -> Path:
    """Write transcript to disk. Overwrites silently."""
    path = transcript_path(data_dir, t.axis_id, t.interviewer, t.interviewee, t.rerun)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(t.model_dump_json(indent=2), encoding="utf-8")
    return path


def save_verdict(data_dir: Path, v: Verdict) -> Path:
    path = verdict_path(data_dir, v.axis_id, v.interviewer, v.interviewee, v.rerun)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(v.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_transcript(
    data_dir: Path,
    axis_id: str,
    interviewer: str,
    interviewee: str,
    rerun: int,
) -> Transcript:
    path = transcript_path(data_dir, axis_id, interviewer, interviewee, rerun)
    return Transcript.model_validate_json(path.read_text(encoding="utf-8"))


def transcript_exists(
    data_dir: Path,
    axis_id: str,
    interviewer: str,
    interviewee: str,
    rerun: int,
) -> bool:
    return transcript_path(data_dir, axis_id, interviewer, interviewee, rerun).exists()


def verdict_exists(
    data_dir: Path,
    axis_id: str,
    interviewer: str,
    interviewee: str,
    rerun: int,
) -> bool:
    return verdict_path(data_dir, axis_id, interviewer, interviewee, rerun).exists()
