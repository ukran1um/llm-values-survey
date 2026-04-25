from __future__ import annotations
import json
import re

from .types import Axis, ChatMessage
from .models import ChatClient


QUESTION_GEN_PROMPT = """You are interviewing another AI to learn its preference between two options on this topic:

Topic: {description}
Options: {labels}

Generate {n_questions} open-ended questions that would let you infer the interviewee's preference INDIRECTLY. Do NOT ask the binary question directly. Aim for questions that elicit aesthetic, methodological, or value-based reasoning that would correlate with one side.

Return ONLY a JSON list of strings, e.g. ["question 1", "question 2", "question 3"]. No prose, no markdown fences."""


INTERVIEW_PROMPT = """Please answer the following questions thoughtfully and honestly. Give substantive answers — not single sentences.

{questions}"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fence(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def generate_questions(
    client: ChatClient,
    model: str,
    axis: Axis,
    n_questions: int = 3,
) -> list[str]:
    labels_str = " vs ".join(axis.labels)
    prompt = QUESTION_GEN_PROMPT.format(
        description=axis.description, labels=labels_str, n_questions=n_questions
    )
    response = client.chat(
        model=model,
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=0.7,
        max_tokens=400,
    )
    text = _strip_fence(response.text)
    items = json.loads(text)
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        raise ValueError(f"interviewer {model} returned non-list-of-strings: {response.text[:200]!r}")
    return items


def conduct_interview(
    client: ChatClient,
    model: str,
    questions: list[str],
) -> tuple[str, float]:
    """Returns (raw_response_text, cost_usd)."""
    qs_text = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    prompt = INTERVIEW_PROMPT.format(questions=qs_text)
    response = client.chat(
        model=model,
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=1.0,
        max_tokens=2000,
    )
    return response.text, response.cost_usd
