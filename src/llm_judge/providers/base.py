"""Provider protocol and shared helpers for all LLM backends."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol, TypedDict, runtime_checkable

#: HTTP status codes that indicate a transient, retryable server-side condition.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class ChatMessage(TypedDict):
    """A single chat message in the OpenAI/Groq chat-completions format."""

    role: str  # "system" | "user" | "assistant"
    content: str


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface the judge needs from an LLM backend.

    Implementations are responsible for transport, authentication, timeouts, and
    translating provider-specific errors into
    :class:`~llm_judge.exceptions.ProviderError`. Retries are handled one layer
    up so that retry policy is uniform across providers.

    Optional extras that bundled providers implement:

    * ``name``: a short provider identifier recorded in result metadata.
    * ``health()``: returns a :class:`ProviderHealth` without running inference.
    * ``usage``: a :class:`UsageTracker` with cumulative request/token counts.

    Implementations must be safe to call from several threads at once, because
    batch, ranking and parallel bias runs share one provider instance.
    """

    #: Human-readable identifier for the active model, used in result metadata.
    model: str

    def complete_json(self, messages: list[ChatMessage]) -> str:
        """Return the assistant's response content as a JSON string.

        Implementations should request the provider's JSON output mode where
        available. The returned string is parsed and validated by the caller.
        """
        ...


@dataclass(frozen=True)
class ProviderHealth:
    """Result of a provider health check (no inference is performed)."""

    ok: bool
    provider: str
    model: str
    detail: str = ""


class UsageTracker:
    """Thread-safe cumulative request and token counters for one provider.

    Token counts are only as complete as the backend reports them; providers
    that don't report usage leave the token totals at zero.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def record(self, prompt_tokens: object = None, completion_tokens: object = None) -> None:
        with self._lock:
            self._requests += 1
            if isinstance(prompt_tokens, int):
                self._prompt_tokens += prompt_tokens
            if isinstance(completion_tokens, int):
                self._completion_tokens += completion_tokens

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "requests": self._requests,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
            }


def shorten(text: str, limit: int = 300) -> str:
    """Trim provider error bodies before they go into exception messages."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
