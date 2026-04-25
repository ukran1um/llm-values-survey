from __future__ import annotations
from pathlib import Path

from .types import Transcript, Classification


def transcript_path(data_dir: Path, t: Transcript) -> Path:
    return (
        Path(data_dir)
        / "raw" / "interviews" / t.axis_id
        / f"{t.interviewer}__{t.interviewee}__r{t.rerun}.json"
    )


def judgment_path(data_dir: Path, c: Classification) -> Path:
    return (
        Path(data_dir)
        / "raw" / "judgments" / c.axis_id
        / f"{c.interviewer}__{c.interviewee}__r{c.rerun}__j{c.judge}.json"
    )


def save_transcript(data_dir: Path, t: Transcript) -> Path:
    path = transcript_path(data_dir, t)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(t.model_dump_json(indent=2), encoding="utf-8")
    return path


def save_classification(data_dir: Path, c: Classification) -> Path:
    path = judgment_path(data_dir, c)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(c.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_transcript(data_dir: Path, axis_id: str, interviewer: str, interviewee: str, rerun: int) -> Transcript:
    path = (
        Path(data_dir)
        / "raw" / "interviews" / axis_id
        / f"{interviewer}__{interviewee}__r{rerun}.json"
    )
    return Transcript.model_validate_json(path.read_text(encoding="utf-8"))


def transcript_exists(data_dir: Path, axis_id: str, interviewer: str, interviewee: str, rerun: int) -> bool:
    path = (
        Path(data_dir)
        / "raw" / "interviews" / axis_id
        / f"{interviewer}__{interviewee}__r{rerun}.json"
    )
    return path.exists()
