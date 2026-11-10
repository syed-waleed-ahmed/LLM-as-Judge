"""Opt-in provider fallback: explicit, recorded on every result, never for quotas or mocks."""

from __future__ import annotations

import json
import threading

import pytest

from llm_judge.config import Settings
from llm_judge.exceptions import ConfigurationError, ProviderError
from llm_judge.judge import Judge
from llm_judge.providers import FallbackProvider, ProviderHealth, track_fallbacks
from llm_judge.providers.fallback import fallback_warnings
from llm_judge.providers.quota import QuotaExceededError
from llm_judge.providers.registry import build_provider, fallback_names

from .conftest import make_entry, make_payload, make_pointwise_payload

PAIRWISE = make_payload(make_entry("Helpfulness", 8, 4))


class _Stub:
    """A named provider that answers, or raises ``error`` when ``fails(messages)``."""

    def __init__(self, name, model, reply=PAIRWISE, error=None, fails=lambda m: True):
        self.name, self.model = name, model
        self._reply, self._error, self._fails = reply, error, fails
        self.calls = 0
        self._lock = threading.Lock()

    def complete_json(self, messages):
        with self._lock:
            self.calls += 1
        if self._error is not None and self._fails(messages):
            raise self._error
        return json.dumps(self._reply(messages) if callable(self._reply) else self._reply)

    def health(self):
        return ProviderHealth(self._error is None, self.name, self.model, "")


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, groq_api_key="test-key", max_retries=0, **overrides)


def _down(name="groq", model="big-model"):
    return _Stub(name, model, error=ProviderError("503 Service Unavailable", retryable=True))


def test_no_fallback_when_the_primary_answers():
    primary, backup = _Stub("groq", "big-model"), _Stub("ollama", "small")
    result = Judge(FallbackProvider(primary, [backup]), _settings()).evaluate(
        "t", "a", "b", "Helpfulness"
    )
    assert backup.calls == 0
    assert "fallback" not in result.metadata
    assert not [w for w in result.warnings if w.code == "provider_fallback"]


def test_fallback_answers_and_is_recorded_loudly():
    primary, backup = _down(), _Stub("ollama", "small")
    result = Judge(FallbackProvider(primary, [backup]), _settings()).evaluate(
        "t", "a", "b", "Helpfulness"
    )
    # The configured judge stays the named judge; the fallback is reported separately.
    assert result.metadata["judge"] == {"provider": "groq", "model": "big-model"}
    assert result.metadata["fallback"] == [
        {"failed": "groq · big-model", "used": "ollama · small", "error": "503 Service Unavailable"}
    ]
    [warning] = [w for w in result.warnings if w.code == "provider_fallback"]
    assert "ollama · small" in warning.message and "not be directly comparable" in warning.message


def test_fallbacks_are_tried_in_order():
    first, second = _down("ollama", "local"), _Stub("openai_compatible", "remote")
    provider = FallbackProvider(_down(), [first, second])
    with track_fallbacks() as events:
        provider.complete_json([{"role": "user", "content": "x"}])
    assert first.calls == 1 and second.calls == 1
    assert events[0].used == "openai_compatible · remote"


def test_every_provider_failing_raises_one_combined_error():
    provider = FallbackProvider(_down(), [_down("ollama", "local")])
    with pytest.raises(ProviderError, match="Every configured provider failed") as info:
        provider.complete_json([{"role": "user", "content": "x"}])
    assert "groq · big-model" in str(info.value) and "ollama · local" in str(info.value)
    assert info.value.retryable  # the last error's retryability is kept


def test_an_exhausted_quota_never_falls_back():
    primary = _Stub("groq", "m", error=QuotaExceededError("Demo budget spent."))
    backup = _Stub("ollama", "small")
    with pytest.raises(QuotaExceededError):
        FallbackProvider(primary, [backup]).complete_json([{"role": "user", "content": "x"}])
    assert backup.calls == 0


def test_both_bias_runs_report_their_fallbacks():
    """The swapped-order run happens on worker threads; its events must still count."""
    result = Judge(
        FallbackProvider(_down(), [_Stub("ollama", "small")]),
        _settings(mitigate_position_bias=True),
    ).evaluate("t", "a", "b", "Helpfulness")
    assert len(result.metadata["fallback"]) == 2
    [warning] = [w for w in result.warnings if w.code == "provider_fallback"]
    assert warning.message.startswith("2 judge call(s)")


def test_concurrent_evaluations_only_see_their_own_fallbacks():
    primary = _Stub(
        "groq",
        "m",
        error=ProviderError("timeout", retryable=True),
        fails=lambda messages: "FAILME" in messages[1]["content"],
    )
    judge = Judge(FallbackProvider(primary, [_Stub("ollama", "small")]), _settings())
    results: dict[str, object] = {}
    barrier = threading.Barrier(2)

    def run(name: str, text: str) -> None:
        barrier.wait()
        results[name] = judge.evaluate("t", text, "b", "Helpfulness")

    threads = [
        threading.Thread(target=run, args=("bad", "FAILME")),
        threading.Thread(target=run, args=("good", "fine")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert "fallback" in results["bad"].metadata  # type: ignore[attr-defined]
    assert "fallback" not in results["good"].metadata  # type: ignore[attr-defined]


def test_pointwise_and_rag_record_fallbacks():
    backup = _Stub(
        "ollama",
        "small",
        reply=lambda messages: (
            {
                "claims": [{"claim": "c", "supported": True, "context_ids": [1]}],
                "answer_relevance": {"score": 8, "explanation": "ok"},
                "contexts": [{"context_id": 1, "relevant": True, "reason": "r"}],
            }
            if "QUESTION" in messages[0]["content"]
            else make_pointwise_payload(("Helpfulness", 7))
        ),
    )
    judge = Judge(FallbackProvider(_down(), [backup]), _settings())
    graded = judge.evaluate_pointwise("t", "o")
    assert graded.metadata["fallback"][0]["used"] == "ollama · small"
    rag = judge.evaluate_rag("q?", ["context one"], "answer")
    assert rag.metadata["fallback"][0]["used"] == "ollama · small"
    assert any(w.code == "provider_fallback" for w in rag.warnings)


def test_fallback_warnings_group_by_provider_pair():
    from llm_judge.providers import FallbackEvent

    events = [FallbackEvent("a", "b", "e1"), FallbackEvent("a", "b", "e2")]
    [warning] = fallback_warnings(events)
    assert warning.message.startswith("2 judge call(s)") and "(e1)" in warning.message


def test_health_reports_every_provider_in_the_chain():
    health = FallbackProvider(_down(), [_Stub("ollama", "small")]).health()
    assert health.ok  # a working fallback keeps the judge usable
    assert "primary groq · big-model: unavailable" in health.detail
    assert "fallback ollama · small: ok" in health.detail


def test_needs_at_least_one_fallback():
    with pytest.raises(ConfigurationError):
        FallbackProvider(_Stub("groq", "m"), [])


# --- configuration --------------------------------------------------------------------


def test_fallback_is_off_unless_configured():
    assert fallback_names(_settings()) == []
    assert not isinstance(build_provider(_settings()), FallbackProvider)


def test_configured_fallback_wraps_the_primary_provider():
    provider = build_provider(_settings(fallback_providers=" ollama "))
    assert isinstance(provider, FallbackProvider)
    assert [getattr(p, "name", None) for p in provider.chain] == ["groq", "ollama"]
    # Building a specific, non-primary provider by name is never wrapped.
    assert not isinstance(
        build_provider(_settings(fallback_providers="ollama"), "ollama"), FallbackProvider
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("mock", "mock judge can't be a fallback"),
        ("groq", "can't include the primary provider"),
        ("ollama,ollama", "lists a provider twice"),
        ("nonexistent", "Unknown provider"),
    ],
)
def test_invalid_fallback_configuration_is_rejected(value, message):
    with pytest.raises(ConfigurationError, match=message):
        build_provider(_settings(fallback_providers=value))
