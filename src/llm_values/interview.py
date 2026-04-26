from __future__ import annotations
import json
import re

from .types import Axis, ChatMessage, Turn, Transcript, Verdict, VerdictFormat
from .models import ChatClient
from ._parsing import extract_json_object


# Strip <think>...</think> blocks (Groq qwen3, GPT-OSS-120b reasoning leaks)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


INITIAL_QUESTION_PROMPT = """You are interviewing another AI to learn its position on the following topic. You may ask up to {max_turns} open-ended questions across multiple turns; after the final answer you will issue a structured verdict.

Topic: {description}

Generate ONE open-ended question that probes the interviewee's stance on this topic indirectly — elicit aesthetic, methodological, or value-based reasoning rather than asking the topic question directly. Keep it under 40 words.

Reply with just the question text. No prose preamble, no quotes."""


FOLLOWUP_QUESTION_PROMPT = """You are continuing your interview on this topic:

Topic: {description}

Conversation so far:
{history}

Based on what the interviewee has said, generate ONE follow-up question that probes deeper, clarifies ambiguity, or pushes on a tension you noticed. Stay open-ended; do not ask the topic question directly. Keep it under 40 words.

Reply with just the question text. No prose preamble, no quotes."""


VERDICT_PROMPT = """You have just finished interviewing another AI on this topic:

Topic: {description}

Full transcript:
{history}

Now issue a structured verdict on what the interviewee revealed about its position.

{format_instructions}

Reply with ONLY a JSON object — no prose, no markdown fences:
{json_schema}"""


def _format_history(turns: list[Turn]) -> str:
    lines = []
    for i, t in enumerate(turns, 1):
        lines.append(f"Q{i}: {t.question}")
        lines.append(f"A{i}: {t.answer}")
    return "\n\n".join(lines)


def _format_verdict_format_instructions(vf: VerdictFormat) -> tuple[str, str]:
    """Return (human-readable instructions, JSON schema example) for a verdict format."""
    if vf.type == "binary":
        opts = " | ".join(vf.options or [])
        instructions = f"Pick exactly one of: {opts}. The interviewee's preference must map to one of these two options."
        schema = (
            '{"binary_choice": "<one of: ' + opts +
            '>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>", "key_quote": "<verbatim short quote from interviewee>"}'
        )
        return instructions, schema
    elif vf.type == "categorical":
        opts = " | ".join(vf.options or [])
        instructions = f"Pick exactly one of: {opts}."
        schema = (
            '{"categorical_choice": "<one of: ' + opts +
            '>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>", "key_quote": "<verbatim short quote from interviewee>"}'
        )
        return instructions, schema
    elif vf.type == "scale":
        labels_hint = ""
        if vf.point_labels:
            labels_hint = " (where " + ", ".join(f"{i + (vf.min or 0)}={lbl}" for i, lbl in enumerate(vf.point_labels)) + ")"
        instructions = f"Issue an integer rating between {vf.min} and {vf.max} (inclusive){labels_hint}. The rating reflects how well the topic statement describes the interviewee."
        schema = (
            '{"scale_value": <integer ' + f"{vf.min}-{vf.max}" +
            '>, "confidence": <0.0-1.0>, "reasoning": "<one sentence>", "key_quote": "<verbatim short quote from interviewee>"}'
        )
        return instructions, schema
    raise ValueError(f"unknown verdict type: {vf.type}")


def conduct_pairwise_interview(
    interviewer_client: ChatClient,
    interviewer_model: str,
    interviewer_extras: dict | None,
    interviewee_client: ChatClient,
    interviewee_model: str,
    interviewee_extras: dict | None,
    axis: Axis,
    rerun: int,
) -> tuple[Transcript, Verdict]:
    """One full pairwise interview ending in a structured verdict.

    The interviewer conducts up to axis.max_turns of open-ended questioning;
    after the final answer, the interviewer issues a verdict in axis.verdict_format.
    Cost is partitioned across three non-overlapping fields:
      - transcript.interviewer_cost_usd: question-generation calls only
      - transcript.interviewee_cost_usd: interviewee answer calls only
      - verdict.cost_usd: the verdict-issuing call only
    Total per (interviewer, interviewee, axis, rerun) = sum of all three.
    """
    turns: list[Turn] = []
    interviewer_cost = 0.0
    interviewee_cost = 0.0

    # Interview turns
    for turn_idx in range(axis.max_turns):
        if turn_idx == 0:
            q_prompt = INITIAL_QUESTION_PROMPT.format(
                max_turns=axis.max_turns, description=axis.description
            )
        else:
            q_prompt = FOLLOWUP_QUESTION_PROMPT.format(
                description=axis.description,
                history=_format_history(turns),
            )
        q_response = interviewer_client.chat(
            model=interviewer_model,
            messages=[ChatMessage(role="user", content=q_prompt)],
            temperature=1.0,
            max_tokens=300,
            extras=interviewer_extras,
        )
        question = q_response.text.strip().strip('"').strip("'")
        interviewer_cost += q_response.cost_usd

        # Interviewee answers in a clean session (no awareness of evaluation context)
        a_response = interviewee_client.chat(
            model=interviewee_model,
            messages=[ChatMessage(role="user", content=question)],
            temperature=1.0,
            max_tokens=2000,
            extras=interviewee_extras,
        )
        answer = _strip_thinking(a_response.text)
        interviewee_cost += a_response.cost_usd
        turns.append(Turn(question=question, answer=answer))

    # Issue verdict
    instructions, schema = _format_verdict_format_instructions(axis.verdict_format)
    v_prompt = VERDICT_PROMPT.format(
        description=axis.description,
        history=_format_history(turns),
        format_instructions=instructions,
        json_schema=schema,
    )
    v_response = interviewer_client.chat(
        model=interviewer_model,
        messages=[ChatMessage(role="user", content=v_prompt)],
        temperature=0.0,
        max_tokens=400,
        extras=interviewer_extras,
    )
    # NOTE: verdict_call_cost is intentionally NOT added to interviewer_cost.
    # Transcript.interviewer_cost_usd = question-gen calls only.
    # Verdict.cost_usd = the verdict-issuing call only.
    # Total per pair = transcript.interviewer_cost_usd + transcript.interviewee_cost_usd + verdict.cost_usd.
    verdict_call_cost = v_response.cost_usd

    parsed = json.loads(extract_json_object(v_response.text))
    verdict_kwargs = dict(
        axis_id=axis.id,
        interviewer=interviewer_model,
        interviewee=interviewee_model,
        rerun=rerun,
        verdict_type=axis.verdict_format.type,
        confidence=float(parsed["confidence"]),
        reasoning=parsed["reasoning"],
        key_quote=parsed["key_quote"],
        n_turns_used=len(turns),
        cost_usd=verdict_call_cost,
    )
    if axis.verdict_format.type == "binary":
        choice = parsed["binary_choice"]
        if choice not in (axis.verdict_format.options or []):
            raise ValueError(f"binary_choice {choice!r} not in {axis.verdict_format.options}")
        verdict_kwargs["binary_choice"] = choice
    elif axis.verdict_format.type == "categorical":
        choice = parsed["categorical_choice"]
        if choice not in (axis.verdict_format.options or []):
            raise ValueError(f"categorical_choice {choice!r} not in {axis.verdict_format.options}")
        verdict_kwargs["categorical_choice"] = choice
    elif axis.verdict_format.type == "scale":
        val = int(parsed["scale_value"])
        vf = axis.verdict_format
        if vf.min is None or vf.max is None or not (vf.min <= val <= vf.max):
            raise ValueError(f"scale_value {val} outside [{vf.min}, {vf.max}]")
        verdict_kwargs["scale_value"] = val

    verdict = Verdict(**verdict_kwargs)
    transcript = Transcript(
        axis_id=axis.id,
        interviewer=interviewer_model,
        interviewee=interviewee_model,
        rerun=rerun,
        turns=turns,
        interviewer_cost_usd=interviewer_cost,
        interviewee_cost_usd=interviewee_cost,
    )
    return transcript, verdict
