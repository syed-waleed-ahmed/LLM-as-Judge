"""Opt-in provider fallback.

When ``JUDGE_FALLBACK_PROVIDERS`` is set (for example ``"ollama"``), a call that
fails on the primary provider is retried on each fallback in order. A fallback
never happens silently:

* every evaluation that used a fallback gets a ``provider_fallback`` warning
  and a ``metadata["fallback"]`` record naming the provider that failed, the
  one that answered, and the error;
* the result's ``metadata["judge"]`` keeps naming the configured (primary)
  judge, so a fallback can't pass as the requested model.

Fallback events are collected per evaluation through a context variable, so
concurrent evaluations (batch rows, ranking matches) each see only their own.
The judge's own worker threads copy the context explicitly.

Two kinds of failure never trigger a fallback: an exhausted demo budget
(:class:`~llm_judge.providers.quota.QuotaExceededError`), which is a policy
limit rather than an outage, and replies that fail parsing, which the judge's
repair retry handles. The mock judge can't be a fallback, because its scores
are not LLM judgments and must never be mixed into live results.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass

from ..exceptions import ConfigurationError, ProviderError
from ..models import EvaluationWarning
from .base import ChatMessage, LLMProvider, ProviderHealth
from .quota import QuotaExceededError

logger = logging.getLogger(__name__)

_EVENTS: ContextVar[list[FallbackEvent] | None] = ContextVar(
    "llm_judge_fallback_events", default=None
)


@dataclass(frozen=True)
class FallbackEvent:
    """One call that the primary provider failed and a fallback answered."""

    failed: str
    used: str
    error: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def label(provider: LLMProvider) -> str:
    """``"name · model"`` for messages and metadata."""
    name = getattr(provider, "name", None) or type(provider).__name__
    return f"{name} · {provider.model}"


@contextmanager
def track_fallbacks() -> Iterator[list[FallbackEvent]]:
    """Collect the fallback events of the evaluation running inside this block."""
    events: list[FallbackEvent] = []
    token = _EVENTS.set(events)
    try:
        yield events
    finally:
        _EVENTS.reset(token)


def fallback_warnings(events: Sequence[FallbackEvent]) -> list[EvaluationWarning]:
    """One ``provider_fallback`` warning per distinct (failed, used) pair."""
    warnings: list[EvaluationWarning] = []
    seen: dict[tuple[str, str], int] = {}
    for event in events:
        seen[(event.failed, event.used)] = seen.get((event.failed, event.used), 0) + 1
    for (failed, used), count in seen.items():
        first_error = next(e.error for e in events if (e.failed, e.used) == (failed, used))
        warnings.append(
            EvaluationWarning(
                code="provider_fallback",
                message=(
                    f"{count} judge call(s) failed on {failed} and were answered by the "
                    f"configured fallback {used} instead ({first_error}). Scores from "
                    "different judges may not be directly comparable."
                ),
            )
        )
    return warnings


class FallbackProvider:
    """Try ``primary`` first, then each of ``fallbacks`` in order, on provider errors."""

    def __init__(self, primary: LLMProvider, fallbacks: Sequence[LLMProvider]) -> None:
        if not fallbacks:
            raise ConfigurationError("FallbackProvider needs at least one fallback provider.")
        self.primary = primary
        self.fallbacks = list(fallbacks)
        # Metadata always names the configured judge; fallbacks are reported separately.
        self.name = getattr(primary, "name", None) or type(primary).__name__
        self.model = primary.model
        self.usage = getattr(primary, "usage", None)

    @property
    def chain(self) -> list[LLMProvider]:
        return [self.primary, *self.fallbacks]

    def complete_json(self, messages: list[ChatMessage]) -> str:
        errors: list[str] = []  # "label: error", for the combined message
        first_error = ""
        chain = self.chain
        for index, provider in enumerate(chain):
            try:
                text = provider.complete_json(messages)
            except QuotaExceededError:
                raise  # a spent budget is a policy limit, not an outage
            except ProviderError as exc:
                errors.append(f"{label(provider)}: {exc}")
                first_error = first_error or str(exc)
                if index == len(chain) - 1:
                    raise ProviderError(
                        "Every configured provider failed. " + " | ".join(errors),
                        retryable=exc.retryable,
                    ) from exc
                logger.warning(
                    "Provider %s failed (%s); falling back to %s.",
                    label(provider),
                    exc,
                    label(chain[index + 1]),
                )
                continue
            if index > 0:
                events = _EVENTS.get()
                if events is not None:
                    events.append(FallbackEvent(label(chain[0]), label(provider), first_error))
            return text
        raise AssertionError("unreachable")  # pragma: no cover

    def health(self) -> ProviderHealth:
        """The primary's health, with each fallback's status in the detail."""
        statuses: list[tuple[LLMProvider, ProviderHealth | None]] = []
        for provider in self.chain:
            check = getattr(provider, "health", None)
            if check is None:
                statuses.append((provider, None))
                continue
            try:
                statuses.append((provider, check()))
            except Exception as exc:  # health checks must never raise
                statuses.append(
                    (provider, ProviderHealth(False, label(provider), provider.model, str(exc)))
                )
        detail = "; ".join(
            f"{'primary' if i == 0 else 'fallback'} {label(p)}: "
            + ("unknown" if h is None else ("ok" if h.ok else f"unavailable ({h.detail})"))
            for i, (p, h) in enumerate(statuses)
        )
        return ProviderHealth(
            ok=any(h is not None and h.ok for _, h in statuses),
            provider=self.name,
            model=self.model,
            detail=detail,
        )


__all__ = [
    "FallbackEvent",
    "FallbackProvider",
    "fallback_warnings",
    "label",
    "track_fallbacks",
]
