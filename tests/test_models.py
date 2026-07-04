from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from llm_judge.models import Criterion, CriterionScore, JudgeResult, Winner


def test_winner_swapped():
    assert Winner.A.swapped() is Winner.B
    assert Winner.B.swapped() is Winner.A
    assert Winner.TIE.swapped() is Winner.TIE


def test_criterion_rejects_invalid_scale():
    with pytest.raises(PydanticValidationError):
        Criterion(name="X", description="d", scale_min=5, scale_max=5)
    with pytest.raises(PydanticValidationError):
        Criterion(name="X", description="d", scale_min=8, scale_max=3)


def test_criterion_is_frozen():
    c = Criterion(name="X", description="d")
    with pytest.raises(PydanticValidationError):
        c.name = "Y"  # type: ignore[misc]


def test_criterion_score_swapped():
    score = CriterionScore(
        criterion="C",
        output_a_score=9,
        output_b_score=3,
        output_a_explanation="a",
        output_b_explanation="b",
        winner=Winner.A,
    )
    swapped = score.swapped()
    assert swapped.output_a_score == 3
    assert swapped.output_b_score == 9
    assert swapped.output_a_explanation == "b"
    assert swapped.winner is Winner.B


def test_judge_result_score_for_averages_criteria():
    result = JudgeResult(
        criteria=[
            CriterionScore(criterion="C1", output_a_score=8, output_b_score=2, winner=Winner.A),
            CriterionScore(criterion="C2", output_a_score=6, output_b_score=4, winner=Winner.A),
        ],
        overall_winner=Winner.A,
    )
    assert result.score_for("A") == 7.0
    assert result.score_for("B") == 3.0
    with pytest.raises(ValueError):
        result.score_for("C")


def test_judge_result_requires_at_least_one_criterion():
    with pytest.raises(PydanticValidationError):
        JudgeResult(criteria=[], overall_winner=Winner.TIE)
