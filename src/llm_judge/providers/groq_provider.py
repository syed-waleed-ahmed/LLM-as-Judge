"""Groq-backed provider using the OpenAI-compatible chat completions API."""

from __future__ import annotations

import logging

from ..exceptions import ConfigurationError, ProviderError
from .base import ChatMessage

logger = logging.getLogger(__name__)

# HTTP status codes that indicate a transient, retryable server-side condition.
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class GroqProvider:
    """Concrete :class:`~llm_judge.providers.base.LLMProvider` for Groq.

    The heavy ``groq`` import is deferred to ``__init__`` so the package remains
    importable (for docs/tests/CLI help) even if the SDK is not installed.
    """

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

        self._client = Groq(api_key=api_key, timeout=timeout)
        self.model = model
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
            raise self._to_provider_error(exc) from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise ProviderError(f"Malformed Groq response: {exc}") from exc

        if not content or not content.strip():
            raise ProviderError("Groq returned an empty response.", retryable=True)
        return content.strip()

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
        retryable = (status in _RETRYABLE_STATUS) or name in retryable_names
        logger.warning(
            "Groq API error (%s, status=%s, retryable=%s): %s", name, status, retryable, exc
        )
        return ProviderError(f"Groq API error: {exc}", retryable=retryable)
