from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


Battery = Literal["mft", "anglophone", "mirror", "recency", "pilot"]


class Axis(BaseModel):
    """A single value-elicitation axis (binary in v1)."""
    id: str = Field(..., description="snake_case identifier, e.g. beatles_vs_stones")
    battery: Battery
    description: str = Field(..., description="One sentence framing the axis")
    labels: list[str] = Field(..., min_length=2, max_length=2)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    model: str


class Transcript(BaseModel):
    """One pairwise interview run."""
    axis_id: str
    interviewer: str
    interviewee: str
    rerun: int
    questions: list[str]
    response: str
    interviewer_cost_usd: float
    interviewee_cost_usd: float


class Classification(BaseModel):
    """One judge's classification of one transcript."""
    axis_id: str
    judge: str
    interviewer: str
    interviewee: str
    rerun: int
    preferred_label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    cost_usd: float
