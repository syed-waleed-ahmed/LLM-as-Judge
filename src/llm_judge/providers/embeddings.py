"""Optional embedding backends for semantic similarity in reference grading.

Set ``JUDGE_EMBEDDINGS_PROVIDER`` to ``ollama`` (local, for example
``JUDGE_EMBEDDINGS_MODEL=nomic-embed-text``) or ``openai_compatible`` (any
``/v1/embeddings`` server, such as LM Studio or vLLM). Reference grading then adds
``reference_metrics.semantic_similarity``: the cosine similarity between the
output's and the reference's embeddings. It complements token F1, which only
measures word overlap.

Both backends are free to run. They're tested against simulated HTTP
responses; neither had a live server to check against when this was written.
An embedding failure never fails an evaluation: the metric is left empty and
the result carries a ``semantic_similarity_unavailable`` warning.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import httpx

from ..config import Settings
from ..exceptions import ConfigurationError, ProviderError
from .base import RETRYABLE_STATUS, retry_after_seconds, shorten

logger = logging.getLogger(__name__)

EMBEDDING_PROVIDERS = ("ollama", "openai_compatible")


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns texts into vectors. Must be safe to call from several threads."""

    model: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per input text, in order."""
        ...


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of the angle between two vectors (0.0 if either is all zeros)."""
    if len(a) != len(b):
        raise ValueError(f"Embedding sizes differ ({len(a)} vs {len(b)}).")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class _HttpEmbeddings:
    """Shared HTTP plumbing and error normalisation."""

    name = "embeddings"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 60.0,
        headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url or not model:
            raise ConfigurationError("An embedding provider needs a base URL and a model.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=timeout, headers=headers or {}, transport=transport)

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        try:
            response = self._client.post(f"{self.base_url}{path}", json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"Embedding request to {self.base_url} timed out.", retryable=True
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Could not reach {self.base_url}: {exc}", retryable=True) from exc
        if response.status_code >= 400:
            retryable = response.status_code in RETRYABLE_STATUS
            raise ProviderError(
                f"{self.base_url} returned HTTP {response.status_code} for embeddings: "
                f"{shorten(response.text)}",
                retryable=retryable,
                retry_after=retry_after_seconds(response.headers, response.text)
                if retryable
                else None,
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(f"Malformed embedding response from {self.base_url}.") from exc

    @staticmethod
    def _vectors(raw: Any, expected: int) -> list[list[float]]:
        if not isinstance(raw, list) or len(raw) != expected:
            raise ProviderError(f"Expected {expected} embedding(s) in the response.")
        try:
            return [[float(x) for x in vector] for vector in raw]
        except (TypeError, ValueError) as exc:
            raise ProviderError("Embedding vectors must contain numbers.") from exc


class OllamaEmbeddings(_HttpEmbeddings):
    """Embeddings from a local Ollama server (``/api/embed``)."""

    name = "ollama"

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        *,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(base_url, model, timeout=timeout, transport=transport)

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post("/api/embed", {"model": self.model, "input": texts})
        return self._vectors(data.get("embeddings") if isinstance(data, dict) else None, len(texts))


class OpenAICompatibleEmbeddings(_HttpEmbeddings):
    """Embeddings from any OpenAI-compatible ``/embeddings`` endpoint."""

    name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        super().__init__(base_url, model, timeout=timeout, headers=headers, transport=transport)

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post("/embeddings", {"model": self.model, "input": texts})
        items = data.get("data") if isinstance(data, dict) else None
        if isinstance(items, list):
            ordered = sorted(
                (item for item in items if isinstance(item, dict)),
                key=lambda item: item.get("index", 0),
            )
            return self._vectors([item.get("embedding") for item in ordered], len(texts))
        return self._vectors(None, len(texts))


def build_embedder(settings: Settings) -> EmbeddingProvider | None:
    """The embedding backend configured by ``JUDGE_EMBEDDINGS_*``, or None when off."""
    name = settings.embeddings_provider.strip()
    if not name:
        return None
    if name not in EMBEDDING_PROVIDERS:
        raise ConfigurationError(
            f"JUDGE_EMBEDDINGS_PROVIDER must be one of {', '.join(EMBEDDING_PROVIDERS)} "
            f"(got '{name}')."
        )
    if not settings.embeddings_model:
        raise ConfigurationError("Set JUDGE_EMBEDDINGS_MODEL, e.g. nomic-embed-text.")
    if name == "ollama":
        return OllamaEmbeddings(
            settings.embeddings_model,
            settings.embeddings_base_url or settings.ollama_base_url,
            timeout=settings.request_timeout,
        )
    base_url = settings.embeddings_base_url or settings.openai_base_url
    if not base_url:
        raise ConfigurationError(
            "Set JUDGE_EMBEDDINGS_BASE_URL (or JUDGE_OPENAI_BASE_URL) for openai_compatible "
            "embeddings."
        )
    return OpenAICompatibleEmbeddings(
        base_url,
        settings.embeddings_model,
        api_key=settings.embeddings_api_key or settings.openai_api_key,
        timeout=settings.request_timeout,
    )


__all__ = [
    "EMBEDDING_PROVIDERS",
    "EmbeddingProvider",
    "OllamaEmbeddings",
    "OpenAICompatibleEmbeddings",
    "build_embedder",
    "cosine_similarity",
]
