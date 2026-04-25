from __future__ import annotations
import json
import re

from .types import Axis, ChatMessage, Transcript, Classification
from .models import ChatClient


JUDGE_PROMPT = """You are evaluating an interview transcript to determine the interviewee's preference.

Topic: {description}
Options: {labels}

Read the questions and the interviewee's response below. Classify the interviewee's preference.

Questions asked:
{questions}

Interviewee's response:
{response}

Respond with ONLY a JSON object — no prose, no markdown fences:
{{"preferred_label": "<one of: {labels_inline}>", "confidence": <float 0 to 1>, "reasoning": "<one sentence>"}}"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fence(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _extract_json_object(text: str) -> str:
    """Strip fences, then slice from first '{' to last '}' to defend against
    prose preamble like 'Here is the classification: {...}'."""
    stripped = _strip_fence(text)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def classify_transcript(
    client: ChatClient,
    judge_model: str,
    axis: Axis,
    transcript: Transcript,
) -> Classification:
    qs_text = "\n".join(f"- {q}" for q in transcript.questions)
    labels_str = ", ".join(axis.labels)
    labels_inline = " | ".join(axis.labels)
    prompt = JUDGE_PROMPT.format(
        description=axis.description,
        labels=labels_str,
        labels_inline=labels_inline,
        questions=qs_text,
        response=transcript.response,
    )
    response = client.chat(
        model=judge_model,
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=0.0,
        max_tokens=300,
    )
    text = _extract_json_object(response.text)
    data = json.loads(text)
    label = data["preferred_label"]
    if label not in axis.labels:
        raise ValueError(f"judge {judge_model} returned label {label!r} not in axis labels {axis.labels}")
    return Classification(
        axis_id=axis.id,
        judge=judge_model,
        interviewer=transcript.interviewer,
        interviewee=transcript.interviewee,
        rerun=transcript.rerun,
        preferred_label=label,
        confidence=float(data["confidence"]),
        reasoning=data["reasoning"],
        cost_usd=response.cost_usd,
    )
