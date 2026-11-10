"""Multi-candidate ranking (Phase 2)."""

from __future__ import annotations

from collections import Counter

import pytest

from llm_judge.exceptions import LLMJudgeError, ProviderError, ValidationError
from llm_judge.judge import Judge

from .conftest import (
    FnProvider,
    SlotAwareProvider,
    make_entry,
    make_payload,
    make_pointwise_payload,
    single_output,
)

QUALITY = {"best": 9, "good": 7, "fine": 7, "weak": 3}


def by_quality(slot_a: str, slot_b: str) -> dict:
    if "broken" in (slot_a, slot_b):
        raise ProviderError("judge unavailable", retryable=False)
    return make_payload(make_entry("Helpfulness", QUALITY[slot_a], QUALITY[slot_b]))


def _judge(settings, fn=by_quality) -> Judge:
    return Judge(provider=SlotAwareProvider(fn), settings=settings)


def test_round_robin_orders_candidates_by_quality(hermetic_settings):
    result = _judge(hermetic_settings).rank("t", {"x": "weak", "y": "best", "z": "good"})
    assert result.strategy == "round_robin"
    assert [r.candidate for r in result.leaderboard] == ["y", "z", "x"]
    assert [r.rank for r in result.leaderboard] == [1, 2, 3]
    assert [r.score for r in result.leaderboard] == [1.0, 0.5, 0.0]
    best = result.leaderboard[0]
    assert (best.wins, best.losses, best.ties) == (2, 0, 0)
    assert len(result.matches) == 3


def test_round_robin_balances_presentation_order(hermetic_settings):
    result = _judge(hermetic_settings).rank("t", {"x": "weak", "y": "best", "z": "good"})
    shown_first = Counter(m.presented_first for m in result.matches)
    assert shown_first == {"x": 1, "y": 1, "z": 1}


def test_match_scores_are_mapped_back_to_candidate_order(hermetic_settings):
    result = _judge(hermetic_settings).rank("t", {"x": "weak", "y": "best"})
    (match,) = result.matches
    assert match.presented_first == "y"  # pair (0, 1) is presented swapped
    assert (match.candidate_a, match.candidate_b) == ("x", "y")
    assert match.winner == "y"
    assert match.overall_score_a == pytest.approx(round(2 / 9, 4))  # x scored 3 on 1-10
    assert match.overall_score_b == pytest.approx(round(8 / 9, 4))


def test_equal_candidates_share_a_rank(hermetic_settings):
    result = _judge(hermetic_settings).rank("t", {"a": "good", "b": "fine", "c": "weak"})
    ranks = {r.candidate: r.rank for r in result.leaderboard}
    assert ranks == {"a": 1, "b": 1, "c": 3}


def test_list_candidates_get_numeric_ids(hermetic_settings):
    result = _judge(hermetic_settings).rank("t", ["weak", "best", "good"])
    assert result.leaderboard[0].candidate == "2"


def test_failed_comparisons_are_isolated(hermetic_settings):
    result = _judge(hermetic_settings).rank("t", {"a": "best", "b": "weak", "c": "broken"})
    assert [r.candidate for r in result.leaderboard] == ["a", "b"]
    codes = Counter(w.code for w in result.warnings)
    assert codes == {"comparison_failed": 2, "candidate_unranked": 1}
    assert sum(1 for m in result.matches if m.error) == 2


def test_all_comparisons_failing_raises(hermetic_settings):
    with pytest.raises(LLMJudgeError, match="All 1 comparisons failed"):
        _judge(hermetic_settings).rank("t", {"a": "broken", "b": "broken"})


def test_round_robin_with_bias_mitigation_doubles_the_calls(hermetic_settings):
    settings = hermetic_settings.model_copy(update={"mitigate_position_bias": True})
    provider = SlotAwareProvider(by_quality)
    result = Judge(provider=provider, settings=settings).rank(
        "t", {"x": "weak", "y": "best", "z": "good"}
    )
    assert provider.calls == 6
    assert result.metadata["position_bias_mitigated"] is True
    assert [r.candidate for r in result.leaderboard] == ["y", "z", "x"]


def test_pointwise_strategy_grades_each_candidate_once(hermetic_settings):
    provider = FnProvider(
        lambda messages: make_pointwise_payload(("Helpfulness", QUALITY[single_output(messages)]))
    )
    result = Judge(provider=provider, settings=hermetic_settings).rank(
        "t", {"x": "weak", "y": "best", "z": "good"}, strategy="pointwise"
    )
    assert provider.calls == 3
    assert result.matches == []
    assert set(result.grades) == {"x", "y", "z"}
    assert [r.candidate for r in result.leaderboard] == ["y", "z", "x"]
    assert result.leaderboard[0].score == pytest.approx(round(8 / 9, 4))


def test_pointwise_strategy_isolates_failures(hermetic_settings):
    def fn(messages):
        text = single_output(messages)
        if text == "broken":
            raise ProviderError("down", retryable=False)
        return make_pointwise_payload(("Helpfulness", QUALITY[text]))

    result = Judge(provider=FnProvider(fn), settings=hermetic_settings).rank(
        "t", {"x": "weak", "y": "broken"}, strategy="pointwise"
    )
    assert [r.candidate for r in result.leaderboard] == ["x"]
    assert [w.code for w in result.warnings] == ["candidate_unranked"]

    with pytest.raises(LLMJudgeError, match="All 2 gradings failed"):
        Judge(provider=FnProvider(fn), settings=hermetic_settings).rank(
            "t", {"x": "broken", "y": "broken"}, strategy="pointwise"
        )


def test_candidate_models_are_recorded_and_validated(hermetic_settings):
    judge = _judge(hermetic_settings)
    result = judge.rank("t", {"x": "weak", "y": "best"}, candidate_models={"y": "gpt-4o"})
    assert result.metadata["candidates"] == {"x": {"model": None}, "y": {"model": "gpt-4o"}}
    assert result.leaderboard[0].model == "gpt-4o"
    with pytest.raises(ValidationError, match="unknown candidates"):
        judge.rank("t", {"x": "weak", "y": "best"}, candidate_models={"z": "m"})


@pytest.mark.parametrize(
    "candidates,message",
    [
        ({"only": "best"}, "between 2 and"),
        ({str(i): "good" for i in range(9)}, "between 2 and 8"),
        ({"a": "best", "b": "   "}, "non-empty"),
        ({"a": "best", " ": "weak"}, "non-empty"),
        ("not a list", "list of texts"),
    ],
)
def test_invalid_candidates_are_rejected(hermetic_settings, candidates, message):
    with pytest.raises(ValidationError, match=message):
        _judge(hermetic_settings).rank("t", candidates)


def test_unknown_strategy_is_rejected(hermetic_settings):
    with pytest.raises(ValidationError, match="Unknown ranking strategy"):
        _judge(hermetic_settings).rank("t", ["weak", "best"], strategy="elo")  # type: ignore[arg-type]
