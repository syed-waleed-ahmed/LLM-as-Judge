from __future__ import annotations

import pytest

from llm_judge.exceptions import ProviderError, ValidationError
from llm_judge.judge import Judge
from llm_judge.models import Winner

from .conftest import FakeProvider, FlakyProvider, PositionBiasedProvider


def _judge(provider, settings):
    return Judge(provider=provider, settings=settings)


def test_basic_evaluation(hermetic_settings, single_criterion_payload):
    judge = _judge(FakeProvider(single_criterion_payload), hermetic_settings)
    result = judge.evaluate("task", "output a", "output b", "Helpfulness")
    assert result.overall_winner is Winner.A
    assert result.criteria[0].output_a_score == 8
    assert result.metadata["model"] == "fake-model"
    assert "latency_seconds" in result.metadata


def test_scores_are_clamped_to_criterion_range(hermetic_settings):
    payload = {
        "criteria": [
            {
                "criterion": "Helpfulness",
                "output_a_score": 15,  # above the 1-10 range
                "output_b_score": -4,  # below the range
                "output_a_explanation": "a",
                "output_b_explanation": "b",
                "winner": "A",
            }
        ],
        "overall_winner": "A",
        "overall_comment": "",
    }
    judge = _judge(FakeProvider(payload), hermetic_settings)
    result = judge.evaluate("t", "a", "b", "Helpfulness")
    assert result.criteria[0].output_a_score == 10
    assert result.criteria[0].output_b_score == 1


def test_winner_is_recomputed_from_scores(hermetic_settings):
    # Judge claims B wins but scores clearly favour A; we trust the numbers.
    payload = {
        "criteria": [
            {
                "criterion": "Helpfulness",
                "output_a_score": 9,
                "output_b_score": 2,
                "output_a_explanation": "a",
                "output_b_explanation": "b",
                "winner": "B",
            }
        ],
        "overall_winner": "B",
        "overall_comment": "",
    }
    judge = _judge(FakeProvider(payload), hermetic_settings)
    result = judge.evaluate("t", "a", "b", "Helpfulness")
    assert result.criteria[0].winner is Winner.A
    assert result.overall_winner is Winner.A


def test_close_scores_are_a_tie(hermetic_settings):
    payload = {
        "criteria": [
            {
                "criterion": "Helpfulness",
                "output_a_score": 7,
                "output_b_score": 7,
                "output_a_explanation": "a",
                "output_b_explanation": "b",
                "winner": "A",
            }
        ],
        "overall_winner": "A",
        "overall_comment": "",
    }
    judge = _judge(FakeProvider(payload), hermetic_settings)
    result = judge.evaluate("t", "a", "b", "Helpfulness")
    assert result.overall_winner is Winner.TIE


def test_position_bias_mitigation_neutralises_a_biased_judge(hermetic_settings):
    settings = hermetic_settings.model_copy(update={"mitigate_position_bias": True})
    provider = PositionBiasedProvider()
    judge = _judge(provider, settings)
    result = judge.evaluate("t", "a", "b", "Helpfulness")

    # The biased provider always favours position A. With mitigation we run both
    # orders, so the averaged scores are equal and the result is a tie.
    assert provider.calls == 2
    assert result.overall_winner is Winner.TIE
    assert result.metadata["position_bias_mitigated"] is True


def test_retries_transient_errors_then_succeeds(hermetic_settings, single_criterion_payload):
    provider = FlakyProvider(single_criterion_payload, fail_times=2, retryable=True)
    settings = hermetic_settings.model_copy(update={"max_retries": 3})
    judge = _judge(provider, settings)
    result = judge.evaluate("t", "a", "b", "Helpfulness")
    assert result.overall_winner is Winner.A
    assert provider.calls == 3  # 2 failures + 1 success


def test_non_retryable_error_is_not_retried(hermetic_settings, single_criterion_payload):
    provider = FlakyProvider(single_criterion_payload, fail_times=1, retryable=False)
    judge = _judge(provider, hermetic_settings)
    with pytest.raises(ProviderError):
        judge.evaluate("t", "a", "b", "Helpfulness")
    assert provider.calls == 1  # gave up immediately


def test_gives_up_after_max_retries(hermetic_settings, single_criterion_payload):
    provider = FlakyProvider(single_criterion_payload, fail_times=99, retryable=True)
    settings = hermetic_settings.model_copy(update={"max_retries": 2})
    judge = _judge(provider, settings)
    with pytest.raises(ProviderError):
        judge.evaluate("t", "a", "b", "Helpfulness")
    assert provider.calls == 3  # initial attempt + 2 retries


def test_empty_input_is_rejected(hermetic_settings, single_criterion_payload):
    judge = _judge(FakeProvider(single_criterion_payload), hermetic_settings)
    with pytest.raises(ValidationError):
        judge.evaluate("", "a", "b", "Helpfulness")
    with pytest.raises(ValidationError):
        judge.evaluate("t", "   ", "b", "Helpfulness")


def test_oversized_input_is_truncated(single_criterion_payload):
    from llm_judge.config import Settings

    settings = Settings(_env_file=None, groq_api_key="k", max_input_chars=10)
    judge = _judge(FakeProvider(single_criterion_payload), settings)
    # Should not raise; the judge truncates rather than failing.
    result = judge.evaluate("t", "x" * 1000, "y" * 1000, "Helpfulness")
    assert result.overall_winner is Winner.A


def test_multiple_criteria(hermetic_settings):
    payload = {
        "criteria": [
            {
                "criterion": "Creativity",
                "output_a_score": 9,
                "output_b_score": 3,
                "output_a_explanation": "a",
                "output_b_explanation": "b",
                "winner": "A",
            },
            {
                "criterion": "Conciseness",
                "output_a_score": 2,
                "output_b_score": 8,
                "output_a_explanation": "a",
                "output_b_explanation": "b",
                "winner": "B",
            },
        ],
        "overall_winner": "A",
        "overall_comment": "",
    }
    judge = _judge(FakeProvider(payload), hermetic_settings)
    result = judge.evaluate("t", "a", "b", ["Creativity", "Conciseness"])
    assert len(result.criteria) == 2
    # Means are equal (9+2)/2 == (3+8)/2 == 5.5 -> tie.
    assert result.overall_winner is Winner.TIE
