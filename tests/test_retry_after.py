"""Retries honour the provider's own retry-after hint (free-tier rate limits)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest

import llm_judge.judge as judge_module
from llm_judge.config import Settings
from llm_judge.exceptions import ProviderError
from llm_judge.judge import MAX_RETRY_WAIT_SECONDS, Judge
from llm_judge.providers import FallbackProvider, OllamaProvider, OpenAICompatibleProvider
from llm_judge.providers.base import retry_after_seconds
from llm_judge.providers.groq_provider import GroqProvider

from .conftest import make_entry, make_payload

PAYLOAD = make_payload(make_entry("Helpfulness", 8, 4))


@pytest.mark.parametrize(
    ("headers", "message", "expected"),
    [
        ({"Retry-After": "7"}, "", 7.0),
        ({"retry-after": "1.5"}, "", 1.5),
        (None, "Rate limit reached. Please try again in 6.48s. Need more tokens?", 6.48),
        (None, "Please try again in 1m2.5s.", 62.5),
        (None, "please try again in 500ms", 0.5),
        (None, "Please try again in 2m.", 120.0),
        ({"Retry-After": "not-a-number"}, "try again in 3s", 3.0),
        ({}, "no hint here", None),
        (None, "", None),
    ],
)
def test_retry_after_parsing(headers, message, expected):
    assert retry_after_seconds(headers, message) == expected


def test_retry_after_http_date():
    when = datetime.now(timezone.utc) + timedelta(seconds=30)
    seconds = retry_after_seconds({"Retry-After": format_datetime(when, usegmt=True)})
    assert seconds is not None and 25 <= seconds <= 31


class _RateLimitedOnce:
    """Fails the first call with a retry-after hint, then answers."""

    model = "m"

    def __init__(self, retry_after: float | None) -> None:
        self.calls = 0
        self.retry_after = retry_after

    def complete_json(self, messages):
        self.calls += 1
        if self.calls == 1:
            raise ProviderError("429", retryable=True, retry_after=self.retry_after)
        return json.dumps(PAYLOAD)


@pytest.fixture
def waits(monkeypatch):
    """Record waits instead of sleeping; a fake clock advances by each wait."""
    recorded: list[float] = []
    clock = [1_000.0]

    def sleep(seconds: float) -> None:
        recorded.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(judge_module, "_sleep", sleep)
    monkeypatch.setattr(judge_module, "_now", lambda: clock[0])
    return recorded


def _judge(provider) -> Judge:
    return Judge(provider, Settings(_env_file=None, groq_api_key="k", max_retries=2))


def test_the_judge_waits_as_long_as_the_provider_asked(waits):
    provider = _RateLimitedOnce(retry_after=6.48)
    _judge(provider).evaluate("t", "a", "b", "Helpfulness")
    assert provider.calls == 2
    # The hint plus a small margin and at most 10% jitter; the cooldown has then passed.
    assert len(waits) == 1 and 6.48 <= waits[0] <= (6.48 + 0.25) * 1.1


def test_the_wait_is_capped(waits):
    _judge(_RateLimitedOnce(retry_after=3600)).evaluate("t", "a", "b", "Helpfulness")
    assert waits == [MAX_RETRY_WAIT_SECONDS]


def test_without_a_hint_the_judge_backs_off_exponentially(waits):
    _judge(_RateLimitedOnce(retry_after=None)).evaluate("t", "a", "b", "Helpfulness")
    assert waits and 0 < waits[0] <= 8


def test_groq_errors_carry_the_hint():
    class RateLimitError(Exception):
        status_code = 429

        def __init__(self) -> None:
            super().__init__("Rate limit reached. Please try again in 4.5s.")
            self.response = httpx.Response(429, headers={"retry-after": "5"})

    error = GroqProvider._to_provider_error(RateLimitError())
    assert error.retryable and error.retry_after == 5.0  # the header wins over the text


def test_non_retryable_errors_have_no_hint():
    class AuthenticationError(Exception):
        status_code = 401

    error = GroqProvider._to_provider_error(AuthenticationError("bad key"))
    assert not error.retryable and error.retry_after is None


def _http_429(request: httpx.Request) -> httpx.Response:
    return httpx.Response(429, headers={"Retry-After": "3"}, json={"error": "slow down"})


def test_http_providers_carry_the_hint():
    compatible = OpenAICompatibleProvider(
        "http://local.test/v1", "m", transport=httpx.MockTransport(_http_429)
    )
    with pytest.raises(ProviderError) as info:
        compatible.complete_json([{"role": "user", "content": "x"}])
    assert info.value.retry_after == 3.0

    ollama = OllamaProvider("llama3.2", transport=httpx.MockTransport(_http_429))
    with pytest.raises(ProviderError) as info:
        ollama.complete_json([{"role": "user", "content": "x"}])
    assert info.value.retry_after == 3.0


def test_a_failed_fallback_chain_keeps_the_last_hint():
    class _Down:
        name, model = "x", "y"

        def complete_json(self, messages):
            raise ProviderError("429", retryable=True, retry_after=9.0)

    with pytest.raises(ProviderError) as info:
        FallbackProvider(_Down(), [_Down()]).complete_json([{"role": "user", "content": "x"}])
    assert info.value.retry_after == 9.0


def test_a_rate_limit_pauses_every_worker_of_the_judge(waits):
    """After one call is told to wait, other calls through the same judge wait too."""
    provider = _RateLimitedOnce(retry_after=5.0)
    judge = _judge(provider)
    judge.evaluate("t", "a", "b", "Helpfulness")  # sets the cooldown, then retries
    waits.clear()
    judge._cooldown.extend(5.0)  # as if another worker was just rate-limited
    judge.evaluate("t", "c", "d", "Helpfulness")
    assert waits and 4.9 <= waits[0] <= 5.0  # the next call waited for the cooldown
