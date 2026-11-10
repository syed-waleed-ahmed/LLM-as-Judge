"""Shared pytest fixtures and test doubles.

No test in this suite makes a network call: the judge is always driven by an
in-memory fake provider, so the whole suite runs offline and deterministically.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable

import pytest

from llm_judge.config import Settings
from llm_judge.exceptions import ProviderError

_SLOT_RE = {
    label: re.compile(rf"<<<OUTPUT {label} (\w+)>>>\n(.*?)\n<<<END OUTPUT {label} \1>>>", re.S)
    for label in ("A", "B")
}


def make_entry(
    criterion: str,
    a: float,
    b: float,
    expl_a: str = "a",
    expl_b: str = "b",
    winner: str = "A",
) -> dict:
    return {
        "criterion": criterion,
        "output_a_score": a,
        "output_b_score": b,
        "output_a_explanation": expl_a,
        "output_b_explanation": expl_b,
        "winner": winner,
    }


def make_payload(*entries: dict, overall: str = "A", comment: str = "") -> dict:
    return {"criteria": list(entries), "overall_winner": overall, "overall_comment": comment}


def slots(messages: list[dict[str, str]]) -> tuple[str, str]:
    """Extract the (Output A, Output B) texts from the judge's user message."""
    user = messages[1]["content"]
    found = []
    for label in ("A", "B"):
        match = _SLOT_RE[label].search(user)
        assert match, f"Output {label} block not found in prompt"
        found.append(match.group(2))
    return found[0], found[1]


class _Recorder:
    """Thread-safe call counter that also keeps every message list it was sent."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.messages: list[list[dict[str, str]]] = []

    def _record(self, messages: list[dict[str, str]]) -> None:
        with self._lock:
            self.calls += 1
            self.messages.append([dict(m) for m in messages])


class FakeProvider(_Recorder):
    """A provider that returns a canned JSON payload for every call."""

    name = "fake"

    def __init__(self, payload: dict | str, model: str = "fake-model") -> None:
        super().__init__()
        self._payload = payload if isinstance(payload, str) else json.dumps(payload)
        self.model = model

    def complete_json(self, messages: list[dict[str, str]]) -> str:
        self._record(messages)
        return self._payload


class ScriptedProvider(_Recorder):
    """Returns the scripted replies in order, repeating the last one when exhausted."""

    model = "scripted-model"

    def __init__(self, replies: list[dict | str]) -> None:
        super().__init__()
        self._replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]

    def complete_json(self, messages: list[dict[str, str]]) -> str:
        self._record(messages)
        return self._replies[min(self.calls, len(self._replies)) - 1]


class SlotAwareProvider(_Recorder):
    """Builds its reply from whichever texts sit in slots A and B of the prompt."""

    model = "slot-aware-model"

    def __init__(self, judge_fn: Callable[[str, str], dict]) -> None:
        super().__init__()
        self._judge_fn = judge_fn

    def complete_json(self, messages: list[dict[str, str]]) -> str:
        self._record(messages)
        return json.dumps(self._judge_fn(*slots(messages)))


class PositionBiasedProvider(_Recorder):
    """Always scores whatever is in position A higher: a pure position bias."""

    model = "biased-model"

    def __init__(self, criterion: str = "Helpfulness", slot_a: int = 9, slot_b: int = 5) -> None:
        super().__init__()
        self.criterion = criterion
        self.slot_a = slot_a
        self.slot_b = slot_b

    def complete_json(self, messages: list[dict[str, str]]) -> str:
        self._record(messages)
        return json.dumps(
            make_payload(
                make_entry(
                    self.criterion,
                    self.slot_a,
                    self.slot_b,
                    "Output A sits in the first slot.",
                    "Output B sits in the second slot.",
                ),
                comment="Output A wins.",
            )
        )


class BarrierProvider(_Recorder):
    """Blocks until two calls are in flight at once; fails if calls are sequential."""

    model = "barrier-model"

    def __init__(self, payload: dict, parties: int = 2) -> None:
        super().__init__()
        self._payload = json.dumps(payload)
        self._barrier = threading.Barrier(parties, timeout=5)
        self.threads: set[int] = set()

    def complete_json(self, messages: list[dict[str, str]]) -> str:
        self._record(messages)
        self.threads.add(threading.get_ident())
        self._barrier.wait()
        return self._payload


class FlakyProvider(_Recorder):
    """Raises a retryable ProviderError ``fail_times`` times, then succeeds."""

    model = "flaky-model"

    def __init__(self, payload: dict, fail_times: int, retryable: bool = True) -> None:
        super().__init__()
        self._payload = json.dumps(payload)
        self.fail_times = fail_times
        self.retryable = retryable

    def complete_json(self, messages: list[dict[str, str]]) -> str:
        self._record(messages)
        if self.calls <= self.fail_times:
            raise ProviderError("transient boom", retryable=self.retryable)
        return self._payload


@pytest.fixture
def hermetic_settings() -> Settings:
    """Settings that ignore the developer's real ``.env`` file."""
    return Settings(_env_file=None, groq_api_key="test-key", max_retries=2)


@pytest.fixture
def single_criterion_payload() -> dict:
    return make_payload(
        make_entry("Helpfulness", 8, 4, "A answers the task.", "B is vague."),
        comment="A is more helpful.",
    )
