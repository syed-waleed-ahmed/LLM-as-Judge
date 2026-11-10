"""Typed data models for evaluation inputs and results.

Using Pydantic gives us three things for free:

* **Validation** of the LLM's JSON output (scores are finite numbers, the winner
  is a valid enum value, required fields are present).
* **Serialisation** to and from JSON and plain dicts for the API, CLI, and UI.
* **Documentation**: the schema is the single source of truth for what a
  judgment looks like.

Every result carries :data:`RESULT_SCHEMA_VERSION`. Backwards-compatible
(additive) changes bump the minor number; breaking changes bump the major.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Version of the JSON result schema produced by this library.
RESULT_SCHEMA_VERSION = "1.1"

# "Output A" / "Response B" / "Candidate A" style references. The prefix is
# case-insensitive but the letter must be uppercase, so ordinary prose such as
# "a response a reader expects" is never rewritten.
_LABEL_RE = re.compile(r"\b((?i:output|response|candidate))(\s+)([AB])\b")


def swap_labels(text: str) -> str:
    """Exchange "Output A" and "Output B" references in free text.

    Used when a judgment produced with the outputs in swapped positions is
    mapped back to the canonical orientation: the judge's prose refers to the
    labels it saw, so those references must be swapped along with the scores.
    Applying the function twice returns the original text.
    """
    return _LABEL_RE.sub(
        lambda m: m.group(1) + m.group(2) + ("B" if m.group(3) == "A" else "A"), text
    )


def _coerce_winner(value: object) -> object:
    """Accept case and whitespace variants such as ``"Tie"`` or ``" a "``.

    Winners are always recomputed from the scores, so being lenient here costs
    nothing and avoids a needless repair retry.
    """
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.lower() == "tie":
            return "tie"
        if cleaned.upper() in ("A", "B"):
            return cleaned.upper()
    return value


class Winner(str, Enum):
    """Which output a judge preferred."""

    A = "A"
    B = "B"
    TIE = "tie"

    def swapped(self) -> Winner:
        """Return the winner as if outputs A and B had been swapped."""
        if self is Winner.A:
            return Winner.B
        if self is Winner.B:
            return Winner.A
        return Winner.TIE


class Criterion(BaseModel):
    """A single dimension to evaluate outputs against.

    ``weight`` controls the criterion's share of the overall score. Weights are
    relative: ``[2, 1]`` and ``[4, 2]`` give identical results.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field(..., min_length=1, max_length=1000)
    scale_min: int = Field(default=1, ge=0)
    scale_max: int = Field(default=10, le=100)
    weight: float = Field(default=1.0, gt=0, le=100)

    @model_validator(mode="after")
    def _check_scale(self) -> Criterion:
        if self.scale_max <= self.scale_min:
            raise ValueError(
                f"scale_max ({self.scale_max}) must be greater than "
                f"scale_min ({self.scale_min}) for criterion '{self.name}'."
            )
        return self

    def clamp(self, score: float) -> float:
        """Clamp ``score`` into ``[scale_min, scale_max]``."""
        return max(float(self.scale_min), min(float(self.scale_max), score))

    def normalise(self, score: float) -> float:
        """Map a score on this criterion's scale onto ``[0, 1]``.

        Normalising before aggregating means a criterion scored 0-100 doesn't
        outweigh one scored 1-10 just because its numbers are bigger.
        """
        return (self.clamp(score) - self.scale_min) / (self.scale_max - self.scale_min)


class CriterionScore(BaseModel):
    """The judge's assessment of both outputs on one criterion."""

    criterion: str
    output_a_score: float = Field(allow_inf_nan=False)
    output_b_score: float = Field(allow_inf_nan=False)
    output_a_explanation: str = ""
    output_b_explanation: str = ""
    winner: Winner

    @field_validator("output_a_explanation", "output_b_explanation", mode="before")
    @classmethod
    def _none_to_empty(cls, value: object) -> object:
        return "" if value is None else value

    @field_validator("winner", mode="before")
    @classmethod
    def _lenient_winner(cls, value: object) -> object:
        return _coerce_winner(value)

    def swapped(self) -> CriterionScore:
        """Return a copy with A/B roles exchanged.

        Used by position-bias mitigation. When the outputs are sent in reversed
        order, the raw response is expressed in terms of the reversed labels, so
        it is swapped back into canonical (A, B) orientation before merging. The
        label references inside the explanations are swapped too.
        """
        return CriterionScore(
            criterion=self.criterion,
            output_a_score=self.output_b_score,
            output_b_score=self.output_a_score,
            output_a_explanation=swap_labels(self.output_b_explanation),
            output_b_explanation=swap_labels(self.output_a_explanation),
            winner=self.winner.swapped(),
        )


class EvaluationWarning(BaseModel):
    """A non-fatal signal that a judgment may be unreliable.

    Warnings never change scores. They tell the caller where a human should
    look, for example a suspected prompt injection or verbosity bias.
    """

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    criterion: str | None = None
    candidate: str | None = None


class JudgeResult(BaseModel):
    """The complete result of judging a pair of outputs.

    The LLM is asked to produce ``criteria``, ``overall_winner`` and
    ``overall_comment``. Everything else is computed by the library: the
    winners are recomputed from the scores, ``overall_score_a`` and
    ``overall_score_b`` are the weighted means of the normalised scores (0-1),
    and ``warnings`` and ``metadata`` describe how the judgment was made.
    """

    schema_version: str = RESULT_SCHEMA_VERSION
    criteria: list[CriterionScore] = Field(..., min_length=1)
    overall_winner: Winner
    overall_comment: str = ""
    overall_score_a: float | None = Field(default=None, ge=0, le=1)
    overall_score_b: float | None = Field(default=None, ge=0, le=1)
    warnings: list[EvaluationWarning] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("overall_comment", mode="before")
    @classmethod
    def _none_to_empty(cls, value: object) -> object:
        return "" if value is None else value

    @field_validator("overall_winner", mode="before")
    @classmethod
    def _lenient_winner(cls, value: object) -> object:
        return _coerce_winner(value)

    def score_for(self, output: str) -> float:
        """Return the unweighted mean raw score across criteria for ``"A"`` or ``"B"``.

        Prefer ``overall_score_a`` and ``overall_score_b``, which are weighted
        and scale-normalised, when the criteria use different scales.
        """
        if output not in ("A", "B"):
            raise ValueError("output must be 'A' or 'B'")
        attr = "output_a_score" if output == "A" else "output_b_score"
        return sum(getattr(c, attr) for c in self.criteria) / len(self.criteria)

    def swapped(self) -> JudgeResult:
        """Return a copy with every A/B role (scores, text, winners) exchanged."""
        return self.model_copy(
            update={
                "criteria": [c.swapped() for c in self.criteria],
                "overall_winner": self.overall_winner.swapped(),
                "overall_comment": swap_labels(self.overall_comment),
                "overall_score_a": self.overall_score_b,
                "overall_score_b": self.overall_score_a,
            }
        )
