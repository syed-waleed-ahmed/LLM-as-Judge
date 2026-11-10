"""Groq-backed provider using the OpenAI-compatible chat completions API."""

from __future__ import annotations

import logging

from ..exceptions import ConfigurationError, ProviderError, ResponseParsingError
from .base import (
    RETRYABLE_STATUS,
    ChatMessage,
    ProviderHealth,
    UsageTracker,
    retry_after_seconds,
)

logger = logging.getLogger(__name__)


def _failed_generation(exc: Exception) -> str | None:
    """The rejected text Groq returns with ``json_validate_failed`` errors, if present."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and isinstance(error.get("failed_generation"), str):
            return str(error["failed_generation"])
    return None


class GroqProvider:
    """Concrete :class:`~llm_judge.providers.base.LLMProvider` for Groq.

    The heavy ``groq`` import is deferred to ``__init__`` so the package remains
    importable (for docs/tests/CLI help) even if the SDK is not installed.
    The underlying HTTP client is thread-safe, so one instance can serve
    concurrent batch workers and parallel bias-mitigation runs.
    """

    name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ConfigurationError("A Groq API key is required to create GroqProvider.")
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ConfigurationError(
                "The 'groq' package is not installed. Run: pip install groq"
            ) from exc

        # max_retries=0: retries are owned by the Judge (and counted by quotas); the
        # SDK's own retries would multiply requests and stretch timeouts.
        self._client = Groq(api_key=api_key, timeout=timeout, max_retries=0)
        self.model = model
        self.usage = UsageTracker()
        self._temperature = temperature

    def complete_json(self, messages: list[ChatMessage]) -> str:
        """Call Groq and return the response content as a JSON string.

        Raises :class:`ProviderError` with ``retryable`` set appropriately so the
        caller's retry policy can decide whether to try again.
        """
        try:
            response = self._client.chat.completions.create(  # type: ignore[call-overload]
                model=self.model,
                messages=messages,
                temperature=self._temperature,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # normalised into ProviderError below
            if "json_validate_failed" in str(exc):
                # JSON mode rejected the model's output: a malformed reply, which the
                # Judge answers with a repair retry rather than failing outright.
                raise ResponseParsingError(
                    "The model's reply was not valid JSON (rejected by Groq JSON mode).",
                    raw_response=_failed_generation(exc),
                ) from exc
            raise self._to_provider_error(exc) from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise ProviderError(f"Malformed Groq response: {exc}") from exc

        if not content or not content.strip():
            raise ProviderError("Groq returned an empty response.", retryable=True)
        usage = getattr(response, "usage", None)
        self.usage.record(
            getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)
        )
        return content.strip()

    def health(self) -> ProviderHealth:
        """Check the API key and model by listing models (no inference, no token cost)."""
        try:
            listing = self._client.models.list()
            ids = [m.id for m in getattr(listing, "data", [])]
        except Exception as exc:  # any SDK failure means "not healthy"
            return ProviderHealth(False, self.name, self.model, f"API check failed: {exc}")
        if ids and self.model not in ids:
            return ProviderHealth(
                False, self.name, self.model, f"model '{self.model}' is not available on Groq"
            )
        return ProviderHealth(True, self.name, self.model, "API key valid; model available")

    @staticmethod
    def _to_provider_error(exc: Exception) -> ProviderError:
        """Translate a groq SDK exception into a :class:`ProviderError`."""
        status = getattr(exc, "status_code", None)
        name = type(exc).__name__

        retryable_names = {
            "APIConnectionError",
            "APITimeoutError",
            "RateLimitError",
            "InternalServerError",
        }
        retryable = (status in RETRYABLE_STATUS) or name in retryable_names
        logger.warning(
            "Groq API error (%s, status=%s, retryable=%s): %s", name, status, retryable, exc
        )
        if status == 404 and "model" in str(exc).lower():
            return ProviderError(
                f"Groq API error: {exc}. The model may have been retired: set GROQ_MODEL to a "
                "current model (run `llm-judge health`, or see "
                "https://console.groq.com/docs/models).",
                retryable=False,
            )
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        return ProviderError(
            f"Groq API error: {exc}",
            retryable=retryable,
            retry_after=retry_after_seconds(headers, str(exc)) if retryable else None,
        )
