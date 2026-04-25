"""Shared JSON-parsing helpers for LLM responses.

LLMs commonly wrap JSON in markdown fences or add prose preamble. These helpers
strip both before json.loads is called.
"""
from __future__ import annotations
import re


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def strip_fence(text: str) -> str:
    """Remove markdown code fences and surrounding whitespace."""
    return _FENCE_RE.sub("", text).strip()


def extract_json_list(text: str) -> str:
    """Strip fences, then slice from first '[' to last ']' to defend against
    prose preamble like 'Sure, here are: [...]'."""
    stripped = strip_fence(text)
    start, end = stripped.find("["), stripped.rfind("]")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def extract_json_object(text: str) -> str:
    """Strip fences, then slice from first '{' to last '}' to defend against
    prose preamble like 'Here is the classification: {...}'."""
    stripped = strip_fence(text)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    return stripped
