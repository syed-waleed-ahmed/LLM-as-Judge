"""Call budgets: hard caps on how many judge requests a provider may make.

Free tiers are finite. :class:`QuotaProvider` wraps any provider and charges
every ``complete_json`` call (including retries and repair attempts) against
one or more :class:`CallBudget` objects, for example a per-visitor budget and
a shared daily budget. When any budget is exhausted the call is refused before
it reaches the network, with a non-retryable :class:`QuotaExceededError`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence

from ..exceptions import ProviderError
from .base import ChatMessage, LLMProvider


class QuotaExceededError(ProviderError):
    """A call budget is exhausted. Never retried."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


class CallBudget:
    """A thread-safe counter of allowed calls, optionally resetting every ``window`` seconds."""

    def __init__(
        self,
        limit: int,
        *,
        name: str = "budget",
        window_seconds: float | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.limit = limit
        self.name = name
        self.window_seconds = window_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._used = 0
        self._window_start = clock()

    def _roll(self) -> None:
        if self.window_seconds and self._clock() - self._window_start >= self.window_seconds:
            self._used = 0
            self._window_start = self._clock()

    @property
    def used(self) -> int:
        with self._lock:
            self._roll()
            return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            self._roll()
            return max(0, self.limit - self._used)

    def try_consume(self, n: int = 1) -> bool:
        with self._lock:
            self._roll()
            if self._used + n > self.limit:
                return False
            self._used += n
            return True

    def refund(self, n: int = 1) -> None:
        with self._lock:
            self._used = max(0, self._used - n)


class QuotaProvider:
    """Wrap ``inner`` so every call is charged to all ``budgets`` (all-or-nothing)."""

    def __init__(self, inner: LLMProvider, budgets: Sequence[CallBudget]) -> None:
        self.inner = inner
        self.budgets = list(budgets)
        self.model = inner.model
        self.name = getattr(inner, "name", type(inner).__name__)

    def complete_json(self, messages: list[ChatMessage]) -> str:
        charged: list[CallBudget] = []
        for budget in self.budgets:
            if not budget.try_consume():
                for b in charged:
                    b.refund()
                raise QuotaExceededError(
                    f"The {budget.name} of {budget.limit} judge request(s) is used up. "
                    "Switch to the offline mock judge, or try again later."
                )
            charged.append(budget)
        return self.inner.complete_json(messages)

    def __getattr__(self, item: str) -> object:
        # Delegate optional extras (health, usage, ...) to the wrapped provider.
        return getattr(self.inner, item)
