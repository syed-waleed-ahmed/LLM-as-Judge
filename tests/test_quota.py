"""Call budgets and the quota-enforcing provider wrapper."""

from __future__ import annotations

import pytest

from llm_judge.config import Settings
from llm_judge.judge import Judge
from llm_judge.providers import CallBudget, MockProvider, QuotaExceededError, QuotaProvider


def test_budget_counts_refunds_and_resets():
    now = [0.0]
    budget = CallBudget(2, name="daily budget", window_seconds=60, clock=lambda: now[0])
    assert budget.try_consume() and budget.try_consume()
    assert not budget.try_consume() and budget.remaining == 0 and budget.used == 2
    budget.refund()
    assert budget.remaining == 1
    now[0] = 61.0  # a new window
    assert budget.used == 0 and budget.remaining == 2


def test_quota_provider_enforces_every_budget_all_or_nothing():
    session, daily = CallBudget(5, name="session budget"), CallBudget(1, name="daily budget")
    provider = QuotaProvider(MockProvider(), [session, daily])
    assert provider.name == "mock" and provider.model == "mock-heuristic-v1"
    assert provider.health().ok  # optional extras are delegated

    judge = Judge(provider=provider, settings=Settings(_env_file=None))
    judge.evaluate_pointwise("Explain DNS", "DNS maps names to addresses.")
    with pytest.raises(QuotaExceededError, match="daily budget of 1") as exc:
        judge.evaluate_pointwise("Explain DNS", "DNS maps names to addresses.")
    assert exc.value.retryable is False
    assert session.used == 1  # the refused call was refunded to the session budget


def test_retries_and_bias_runs_are_charged():
    budget = CallBudget(10)
    settings = Settings(_env_file=None, mitigate_position_bias=True)
    Judge(provider=QuotaProvider(MockProvider(), [budget]), settings=settings).evaluate(
        "t", "a detailed answer", "short", "Helpfulness"
    )
    assert budget.used == 2
