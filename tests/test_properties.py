"""Property-based tests (Hypothesis) for the scoring invariants."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llm_judge.config import Settings
from llm_judge.criteria import resolve_criterion
from llm_judge.judge import Judge
from llm_judge.models import Criterion, CriterionScore, Winner, swap_labels
from llm_judge.scoring import decide_winner, match_criteria, weighted_mean

from .conftest import PositionBiasedProvider

finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
unit = st.floats(min_value=0, max_value=1)


@st.composite
def criteria_scales(draw: st.DrawFn) -> Criterion:
    low = draw(st.integers(min_value=0, max_value=99))
    high = draw(st.integers(min_value=low + 1, max_value=100))
    return Criterion(name="C", description="d", scale_min=low, scale_max=high)


@given(criteria_scales(), finite)
def test_normalised_scores_stay_in_unit_interval(criterion, score):
    assert 0.0 <= criterion.normalise(score) <= 1.0


@given(unit, unit, st.floats(min_value=0, max_value=0.5))
def test_winner_decision_is_antisymmetric(a, b, margin):
    assert decide_winner(a, b, margin).swapped() is decide_winner(b, a, margin)


@given(st.lists(st.tuples(unit, st.floats(min_value=0.01, max_value=100)), min_size=1))
def test_weighted_mean_of_unit_scores_is_in_unit_interval(pairs):
    values, weights = zip(*pairs, strict=True)
    assert -1e-9 <= weighted_mean(values, weights) <= 1 + 1e-9


@given(st.permutations(["Creativity", "Helpfulness", "Conciseness"]))
def test_match_criteria_returns_request_order(order):
    requested = [resolve_criterion(n) for n in ("Helpfulness", "Conciseness", "Creativity")]
    returned = [
        CriterionScore(criterion=n, output_a_score=1, output_b_score=1, winner=Winner.TIE)
        for n in order
    ]
    assert [s.criterion for s in match_criteria(requested, returned)] == [
        "Helpfulness",
        "Conciseness",
        "Creativity",
    ]


fragments = st.sampled_from(
    ["Output A", "Output B", "output A", "Response B", "candidate A", "A", "B", "a", " ", "x", "\n"]
)


@given(st.lists(fragments).map("".join))
def test_label_swapping_is_an_involution(text):
    assert swap_labels(swap_labels(text)) == text


@settings(max_examples=40, deadline=None)
@given(st.integers(min_value=1, max_value=10), st.integers(min_value=1, max_value=10))
def test_mitigation_cancels_any_pure_position_bias(slot_a, slot_b):
    """Whatever fixed scores a purely positional judge gives, mitigation yields a tie."""
    judge_settings = Settings(
        _env_file=None, groq_api_key="k", mitigate_position_bias=True, parallel_bias_runs=False
    )
    provider = PositionBiasedProvider(slot_a=slot_a, slot_b=slot_b)
    result = Judge(provider=provider, settings=judge_settings).evaluate(
        "t", "a", "b", "Helpfulness"
    )
    score = result.criteria[0]
    assert score.output_a_score == pytest.approx(score.output_b_score)
    assert result.overall_winner is Winner.TIE
