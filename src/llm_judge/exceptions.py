"""Exception hierarchy for the llm_judge package.

A single base exception (:class:`LLMJudgeError`) lets callers catch every
error the library raises, while the specific subclasses allow fine-grained
handling (e.g. distinguishing a configuration problem from a transient
provider outage).
"""

from __future__ import annotations


class LLMJudgeError(Exception):
    """Base class for all errors raised by llm_judge."""


class ConfigurationError(LLMJudgeError):
    """Raised when the library is misconfigured (e.g. missing API key)."""


class ProviderError(LLMJudgeError):
    """Raised when the underlying LLM provider fails or is unreachable.

    ``retryable`` signals whether the failure is transient (rate limits,
    timeouts, 5xx responses) and therefore worth retrying.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ResponseParsingError(LLMJudgeError):
    """Raised when the judge response cannot be parsed into a valid result."""

    def __init__(self, message: str, *, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class ValidationError(LLMJudgeError):
    """Raised when user-supplied input fails validation before a request."""
