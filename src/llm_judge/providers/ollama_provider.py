"""Local inference through Ollama's native REST API (https://ollama.com).

Models run on your own hardware, at no cost and with no API key. Start the
server with ``ollama serve`` and download a model with ``ollama pull <model>``.
Small local models are noticeably weaker judges than large hosted ones, so
expect more repair retries and noisier scores.
"""

from __future__ import annotations

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


def _same_model(requested: str, available: str) -> bool:
    """Ollama treats ``llama3.2`` and ``llama3.2:latest`` as the same model."""

    def norm(name: str) -> str:
        return name if ":" in name else f"{name}:latest"

    return norm(requested) == norm(available)


class OllamaProvider:
    """:class:`~llm_judge.providers.base.LLMProvider` for a local Ollama server."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not model:
            raise ConfigurationError("OllamaProvider needs a model name, e.g. 'llama3.2'.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.usage = UsageTracker()
        self._temperature = temperature
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def complete_json(self, messages: list[ChatMessage]) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": self._temperature},
        }
        try:
            response = self._client.post("/api/chat", json=body)
        except httpx.ConnectError as exc:
            raise ProviderError(
                f"Could not connect to Ollama at {self.base_url}. Is it running? "
                "Start it with `ollama serve`.",
                retryable=False,
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderError("Ollama request timed out.", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ProviderError(f"Ollama transport error: {exc}", retryable=True) from exc

        if response.status_code == 404:
            raise ProviderError(
                f"Model '{self.model}' is not available in Ollama. "
                f"Download it with `ollama pull {self.model}`.",
                retryable=False,
            )
        if response.status_code >= 400:
            retryable = response.status_code in RETRYABLE_STATUS
            raise ProviderError(
                f"Ollama returned HTTP {response.status_code}: {shorten(response.text)}",
                retryable=retryable,
                retry_after=(
                    retry_after_seconds(response.headers, response.text) if retryable else None
                ),
            )

        try:
            data = response.json()
            content = data["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError(f"Malformed Ollama response: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("Ollama returned an empty response.", retryable=True)

        self.usage.record(data.get("prompt_eval_count"), data.get("eval_count"))
        return content.strip()

    def health(self) -> ProviderHealth:
        """Check that Ollama is running and the model has been pulled."""
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
            names = [m.get("name", "") for m in response.json().get("models", [])]
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            return ProviderHealth(
                False, self.name, self.model, f"Ollama not reachable at {self.base_url}: {exc}"
            )
        if not any(_same_model(self.model, n) for n in names):
            return ProviderHealth(
                False,
                self.name,
                self.model,
                f"model not pulled; run `ollama pull {self.model}`",
            )
        return ProviderHealth(True, self.name, self.model, f"running at {self.base_url}")
