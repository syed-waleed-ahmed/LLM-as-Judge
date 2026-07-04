"""Provider protocol shared by all LLM backends."""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable


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
    """

    #: Human-readable identifier for the active model, used in result metadata.
    model: str

    def complete_json(self, messages: list[ChatMessage]) -> str:
        """Return the assistant's response content as a JSON string.

        Implementations should request the provider's JSON output mode where
        available. The returned string is parsed and validated by the caller.
        """
        ...
