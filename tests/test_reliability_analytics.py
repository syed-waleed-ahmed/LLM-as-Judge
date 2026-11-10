"""Judge consistency and inter-judge agreement (Phase 8)."""

from __future__ import annotations

import threading

import pytest

from llm_judge.config import Settings
from llm_judge.datasets import DatasetRunner, load_sample
from llm_judge.exceptions import ProviderError, ValidationError
from llm_judge.experiments import ExperimentError, ExperimentStore, record_dataset_run
from llm_judge.judge import Judge
from llm_judge.providers import MockProvider
from llm_judge.reliability import cohen_kappa, judge_agreement, measure_consistency, spearman

from .conftest import FnProvider, make_entry, make_payload


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


class _Cycle:
    """Thread-safe iterator over scripted (a, b) score pairs."""

    def __init__(self, pairs):
        self.pairs = pairs
        self.i = 0
        self.lock = threading.Lock()

    def __call__(self, messages):
        with self.lock:
            a, b = self.pairs[self.i % len(self.pairs)]
            self.i += 1
        return make_payload(make_entry("Helpfulness", a, b))


def test_pairwise_consistency_statistics():
    judge = Judge(
        provider=FnProvider(_Cycle([(9, 3), (8, 3), (7, 3), (3, 8), (2, 9)])),
        settings=_settings(temperature=0.7),
    )
    report = measure_consistency(judge, "t", output_a="a", output_b="b", samples=5, max_workers=1)
    assert report.mode == "pairwise" and report.succeeded == 5
    assert report.winner_counts == {"A": 3, "B": 2}
    assert report.modal_winner == "A" and report.agreement_rate == 0.6
    assert report.ci_low <= report.mean <= report.ci_high
    spread = report.criteria["Helpfulness"]
    assert spread.mean_a == pytest.approx(5.8) and spread.std_a > 0
    assert report.notes == []  # temperature > 0, so no note


def test_pointwise_consistency_with_a_deterministic_judge():
    judge = Judge(provider=MockProvider(), settings=_settings())
    report = measure_consistency(judge, "Explain DNS", output="DNS maps names to IPs.", samples=3)
    assert report.mode == "pointwise" and report.std == 0
    assert report.ci_low == report.ci_high == report.mean
    assert report.agreement_rate is None
    assert any("Temperature is 0" in n for n in report.notes)


def test_consistency_reports_failed_samples():
    calls = {"n": 0}
    lock = threading.Lock()

    def flaky(messages):
        with lock:
            calls["n"] += 1
            n = calls["n"]
        if n % 2 == 0:
            raise ProviderError("down", retryable=False)
        return make_payload(make_entry("Helpfulness", 8, 4))

    judge = Judge(provider=FnProvider(flaky), settings=_settings())
    report = measure_consistency(judge, "t", output_a="a", output_b="b", samples=4, max_workers=1)
    assert report.succeeded == 2 and len(report.errors) == 2

    always_down = Judge(
        provider=FnProvider(lambda m: (_ for _ in ()).throw(ProviderError("x"))),
        settings=_settings(),
    )
    empty = measure_consistency(always_down, "t", output="o", samples=2)
    assert empty.mean is None and any("Fewer than two" in n for n in empty.notes)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"output_a": "a"}, "both output_a and output_b"),
        ({"output_a": "a", "output_b": "b", "output": "o"}, "both output_a and output_b"),
        ({}, "Give output_a and output_b, or output"),
        ({"output": "o", "samples": 1}, "between 2 and 50"),
    ],
)
def test_consistency_input_validation(kwargs, message):
    judge = Judge(provider=MockProvider(), settings=_settings())
    with pytest.raises(ValidationError, match=message):
        measure_consistency(judge, "t", **kwargs)


def test_cohen_kappa():
    assert cohen_kappa(["A", "A", "B", "B"], ["A", "B", "B", "B"]) == pytest.approx(0.5)
    assert cohen_kappa(["A", "B"], ["A", "B"]) == 1.0
    assert cohen_kappa(["A", "A"], ["A", "A"]) == 1.0  # chance agreement is total
    assert cohen_kappa(["A", "B"], ["B", "A"]) == pytest.approx(-1.0)
    with pytest.raises(ValidationError):
        cohen_kappa(["A"], ["A", "B"])


def test_spearman():
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    # Ties share average ranks: x ranks [1, 2.5, 2.5, 4], y ranks [1, 3, 2, 4];
    # Pearson on those ranks = 4.5 / sqrt(4.5 * 5).
    assert spearman([1, 2, 2, 3], [1, 3, 2, 4]) == pytest.approx(4.5 / (4.5 * 5) ** 0.5)
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    with pytest.raises(ValidationError):
        spearman([1], [1])


def test_judge_agreement_between_stored_runs():
    store = ExperimentStore(":memory:")
    judge = Judge(provider=MockProvider(), settings=_settings())

    def record(name):
        run = DatasetRunner(judge).run(load_sample(name))
        return record_dataset_run(store, run, judge).id

    a, b = record("safety"), record("safety")
    pairwise = judge_agreement(store, a, b)
    assert pairwise.cohen_kappa == 1.0 and pairwise.raw_agreement == 1.0
    assert pairwise.matched_rows == 5
    assert any("same judge" in n for n in pairwise.notes)

    g1, g2 = record("general_qa"), record("general_qa")
    graded = judge_agreement(store, g1, g2)
    assert graded.spearman == pytest.approx(1.0) and graded.mean_absolute_difference == 0

    with pytest.raises(ExperimentError, match="Can't compare"):
        judge_agreement(store, a, g1)
