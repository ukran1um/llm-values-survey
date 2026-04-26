from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator


Battery = Literal["mfq", "mirror", "extension", "pilot"]
VerdictType = Literal["binary", "scale", "categorical"]


class VerdictFormat(BaseModel):
    """How the interviewer should structure its verdict for an axis."""
    type: VerdictType
    # Required for binary and categorical: list of allowed values.
    options: list[str] | None = None
    # Required for scale: integer min/max bounds (inclusive).
    min: int | None = None
    max: int | None = None
    # Optional for scale: human-readable labels for each integer point.
    point_labels: list[str] | None = None

    @model_validator(mode="after")
    def _check_format(self):
        if self.type == "binary":
            if not self.options or len(self.options) != 2:
                raise ValueError("binary verdict_format requires exactly 2 options")
        elif self.type == "categorical":
            if not self.options or len(self.options) < 2:
                raise ValueError("categorical verdict_format requires >=2 options")
        elif self.type == "scale":
            if self.min is None or self.max is None or self.max <= self.min:
                raise ValueError("scale verdict_format requires min and max with max > min")
            if self.point_labels is not None and len(self.point_labels) != self.max - self.min + 1:
                raise ValueError(
                    f"scale verdict_format point_labels must have {self.max - self.min + 1} entries "
                    f"for range [{self.min}, {self.max}], got {len(self.point_labels)}"
                )
        return self


class Axis(BaseModel):
    """A single value-elicitation axis. Method: pairwise interpretive multi-turn interview."""
    id: str = Field(..., description="snake_case identifier")
    battery: Battery
    description: str = Field(..., description="One sentence framing the axis for the interviewer")
    verdict_format: VerdictFormat
    max_turns: int = Field(default=2, ge=1, le=4)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    model: str


class Turn(BaseModel):
    """One Q/A exchange in a pairwise interview."""
    question: str
    answer: str


class Transcript(BaseModel):
    """A pairwise interview between two models on one axis."""
    axis_id: str
    interviewer: str
    interviewee: str
    rerun: int
    turns: list[Turn]
    interviewer_cost_usd: float
    interviewee_cost_usd: float


class Verdict(BaseModel):
    """The interviewer's structured verdict on what the interviewee revealed."""
    axis_id: str
    interviewer: str
    interviewee: str
    rerun: int
    verdict_type: VerdictType
    # Exactly one of these is populated based on verdict_type:
    binary_choice: str | None = None
    scale_value: int | None = None
    categorical_choice: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    key_quote: str = Field(..., description="Verbatim short quote from interviewee anchoring the verdict")
    n_turns_used: int
    cost_usd: float
