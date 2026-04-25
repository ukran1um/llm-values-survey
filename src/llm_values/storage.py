from __future__ import annotations
from pathlib import Path

from .types import Transcript, Classification


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
        / f"{interviewer}__{interviewee}__r{rerun}.json"
    )


def judgment_path(
    data_dir: Path,
    axis_id: str,
    interviewer: str,
    interviewee: str,
    rerun: int,
    judge: str,
) -> Path:
    return (
        Path(data_dir)
        / "raw" / "judgments" / axis_id
        / f"{interviewer}__{interviewee}__r{rerun}__j{judge}.json"
    )


def save_transcript(data_dir: Path, t: Transcript) -> Path:
    """Write transcript to disk. Overwrites silently; callers should use
    `transcript_exists` for checkpointing."""
    path = transcript_path(data_dir, t.axis_id, t.interviewer, t.interviewee, t.rerun)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(t.model_dump_json(indent=2), encoding="utf-8")
    return path


def save_classification(data_dir: Path, c: Classification) -> Path:
    path = judgment_path(data_dir, c.axis_id, c.interviewer, c.interviewee, c.rerun, c.judge)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(c.model_dump_json(indent=2), encoding="utf-8")
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
