"""Shared pytest fixtures and test doubles.

No test in this suite makes a network call: the judge is always driven by an
in-memory fake provider, so the whole suite runs offline and deterministically.
"""

from __future__ import annotations

import json

import pytest

from llm_judge.config import Settings
from llm_judge.exceptions import ProviderError


class FakeProvider:
    """A provider that returns a canned JSON payload for every call."""

    def __init__(self, payload: dict | str, model: str = "fake-model") -> None:
        self._payload = payload if isinstance(payload, str) else json.dumps(payload)
        self.model = model
        self.calls = 0

    def complete_json(self, messages):  # noqa: ANN001, ANN201
        self.calls += 1
        return self._payload


class PositionBiasedProvider:
    """Always scores whatever is in position A higher — a pure position bias."""

    model = "biased-model"

    def __init__(self, criterion: str = "Helpfulness") -> None:
        self.criterion = criterion
        self.calls = 0

    def complete_json(self, messages):  # noqa: ANN001, ANN201
        self.calls += 1
        return json.dumps(
            {
                "criteria": [
                    {
                        "criterion": self.criterion,
                        "output_a_score": 9,
                        "output_b_score": 5,
                        "output_a_explanation": "A sits in the first slot.",
                        "output_b_explanation": "B sits in the second slot.",
                        "winner": "A",
                    }
                ],
                "overall_winner": "A",
                "overall_comment": "Position A wins.",
            }
        )


class FlakyProvider:
    """Raises a retryable ProviderError ``fail_times`` times, then succeeds."""

    model = "flaky-model"

    def __init__(self, payload: dict, fail_times: int, retryable: bool = True) -> None:
        self._payload = json.dumps(payload)
        self.fail_times = fail_times
        self.retryable = retryable
        self.calls = 0

    def complete_json(self, messages):  # noqa: ANN001, ANN201
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ProviderError("transient boom", retryable=self.retryable)
        return self._payload


@pytest.fixture
def hermetic_settings() -> Settings:
    """Settings that ignore the developer's real ``.env`` file."""
    return Settings(_env_file=None, groq_api_key="test-key", max_retries=2)


@pytest.fixture
def single_criterion_payload() -> dict:
    return {
        "criteria": [
            {
                "criterion": "Helpfulness",
                "output_a_score": 8,
                "output_b_score": 4,
                "output_a_explanation": "A answers the task.",
                "output_b_explanation": "B is vague.",
                "winner": "A",
            }
        ],
        "overall_winner": "A",
        "overall_comment": "A is more helpful.",
    }
