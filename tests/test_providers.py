from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_judge.exceptions import ConfigurationError, ProviderError
from llm_judge.providers.groq_provider import GroqProvider


def _make_provider() -> GroqProvider:
    """Construct a provider (no network happens until a request is made)."""
    return GroqProvider(api_key="test-key", model="fake-model")


def test_requires_api_key():
    with pytest.raises(ConfigurationError):
        GroqProvider(api_key="", model="m")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("RateLimitError", True),
        ("APITimeoutError", True),
        ("APIConnectionError", True),
        ("InternalServerError", True),
        ("BadRequestError", False),
        ("AuthenticationError", False),
    ],
)
def test_error_classification_by_exception_name(name, expected):
    exc_type = type(name, (Exception,), {})
    result = GroqProvider._to_provider_error(exc_type("boom"))
    assert result.retryable is expected


def test_error_classification_retryable_status_code():
    exc = Exception("rate limited")
    exc.status_code = 429  # type: ignore[attr-defined]
    result = GroqProvider._to_provider_error(exc)
    assert result.retryable is True


def test_complete_json_returns_content():
    provider = _make_provider()
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
    )
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )
    assert provider.complete_json([{"role": "user", "content": "hi"}]) == '{"ok": true}'


def test_complete_json_rejects_empty_response():
    provider = _make_provider()
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="   "))]
    )
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )
    with pytest.raises(ProviderError) as exc:
        provider.complete_json([{"role": "user", "content": "hi"}])
    assert exc.value.retryable is True


def test_complete_json_wraps_client_errors():
    provider = _make_provider()

    def _boom(**kwargs):
        err = RuntimeError("network down")
        err.status_code = 500  # type: ignore[attr-defined]
        raise err

    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_boom))
    )
    with pytest.raises(ProviderError) as exc:
        provider.complete_json([{"role": "user", "content": "hi"}])
    assert exc.value.retryable is True
