"""Typed data models for evaluation inputs and results.

Using Pydantic gives us three things for free:

* **Validation** of the LLM's JSON output (scores in range, winner is a valid
  enum value, required fields present).
* **Serialisation** to/from JSON and plain dicts for the API, CLI, and UI.
* **Documentation** — the schema is the single source of truth for what a
  judgment looks like.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    """A single dimension to evaluate outputs against."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field(..., min_length=1, max_length=1000)
    scale_min: int = Field(default=1, ge=0)
    scale_max: int = Field(default=10, le=100)

    @model_validator(mode="after")
    def _check_scale(self) -> Criterion:
        if self.scale_max <= self.scale_min:
            raise ValueError(
                f"scale_max ({self.scale_max}) must be greater than "
                f"scale_min ({self.scale_min}) for criterion '{self.name}'."
            )
        return self


class CriterionScore(BaseModel):
    """The judge's assessment of both outputs on one criterion."""

    criterion: str
    output_a_score: float
    output_b_score: float
    output_a_explanation: str = ""
    output_b_explanation: str = ""
    winner: Winner

    def swapped(self) -> CriterionScore:
        """Return a copy with A/B roles exchanged.

        Used by position-bias mitigation: when we send outputs in reversed
        order, the raw response is expressed in terms of the reversed labels,
        so we swap it back into canonical (A, B) orientation before merging.
        """
        return CriterionScore(
            criterion=self.criterion,
            output_a_score=self.output_b_score,
            output_b_score=self.output_a_score,
            output_a_explanation=self.output_b_explanation,
            output_b_explanation=self.output_a_explanation,
            winner=self.winner.swapped(),
        )


class JudgeResult(BaseModel):
    """The complete result of judging a pair of outputs.

    This is the model the LLM is asked to produce (minus ``metadata``, which the
    library attaches afterwards).
    """

    criteria: list[CriterionScore] = Field(..., min_length=1)
    overall_winner: Winner
    overall_comment: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)

    def score_for(self, output: str) -> float:
        """Return the mean score across criteria for ``"A"`` or ``"B"``."""
        if output not in ("A", "B"):
            raise ValueError("output must be 'A' or 'B'")
        attr = "output_a_score" if output == "A" else "output_b_score"
        return sum(getattr(c, attr) for c in self.criteria) / len(self.criteria)
