"""Generic provider for any OpenAI-compatible ``/chat/completions`` endpoint.

Works with self-hosted servers (LM Studio, vLLM, llama.cpp ``server``, Ollama's
``/v1`` API) and with hosted services that expose the same API, including
those offering free models. It's configured explicitly through ``JUDGE_OPENAI_*``
settings, so an unrelated (possibly paid) ``OPENAI_API_KEY`` in the
environment is never used by accident.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ..exceptions import ConfigurationError, ProviderError
from .base import (
    RETRYABLE_STATUS,
    ChatMessage,
    ProviderHealth,
    UsageTracker,
    retry_after_seconds,
    shorten,
)

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider:
    """:class:`~llm_judge.providers.base.LLMProvider` for OpenAI-compatible APIs."""

    name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
        json_mode: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url:
            raise ConfigurationError("An OpenAI-compatible provider needs a base URL.")
        if not model:
            raise ConfigurationError("An OpenAI-compatible provider needs a model name.")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.usage = UsageTracker()
        self._temperature = temperature
        self._json_mode = json_mode
        # httpx.Client is thread-safe, so one instance serves concurrent workers.
        self._client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=timeout, transport=transport
        )

    def complete_json(self, messages: list[ChatMessage]) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if self._json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            response = self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"Request to {self.base_url} timed out.", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ProviderError(f"Could not reach {self.base_url}: {exc}", retryable=True) from exc

        if response.status_code >= 400:
            retryable = response.status_code in RETRYABLE_STATUS
            logger.warning(
                "OpenAI-compatible API error (status=%s, retryable=%s)",
                response.status_code,
                retryable,
            )
            raise ProviderError(
                f"{self.base_url} returned HTTP {response.status_code}: {shorten(response.text)}",
                retryable=retryable,
                retry_after=(
                    retry_after_seconds(response.headers, response.text) if retryable else None
                ),
            )

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Malformed response from {self.base_url}: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError(f"{self.base_url} returned an empty response.", retryable=True)

        usage = data.get("usage") or {}
        self.usage.record(usage.get("prompt_tokens"), usage.get("completion_tokens"))
        return content.strip()

    def health(self) -> ProviderHealth:
        """Check that the server answers ``GET /models`` (no inference)."""
        try:
            response = self._client.get("/models")
        except httpx.HTTPError as exc:
            return ProviderHealth(False, self.name, self.model, f"unreachable: {exc}")
        if response.status_code >= 400:
            return ProviderHealth(
                False, self.name, self.model, f"HTTP {response.status_code} from /models"
            )
        try:
            ids = [m.get("id") for m in response.json().get("data", [])]
        except (ValueError, AttributeError):
            ids = []
        if ids and self.model not in ids:
            return ProviderHealth(
                False, self.name, self.model, f"model '{self.model}' not listed by the server"
            )
        return ProviderHealth(True, self.name, self.model, f"reachable at {self.base_url}")
