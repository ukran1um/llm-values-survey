from __future__ import annotations
import datetime as _dt
import subprocess
from pathlib import Path


def now_iso() -> str:
    """Current UTC time as ISO 8601 with Z suffix."""
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_methodology_commit(repo_dir: Path | None = None) -> str:
    """Return the short SHA of the inner-repo HEAD. Returns 'unknown' if git unavailable."""
    cwd = repo_dir or Path(__file__).resolve().parent.parent.parent
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"
