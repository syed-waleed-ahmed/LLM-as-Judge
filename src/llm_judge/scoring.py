"""Pure scoring helpers: criteria matching, normalisation, aggregation, winners.

Everything here is deterministic and free of I/O, which keeps the judge's
arithmetic easy to reason about and to property-test.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar

from .exceptions import ResponseParsingError
from .models import Criterion, Winner


class _Named(Protocol):
    @property
    def criterion(self) -> str: ...


_T = TypeVar("_T", bound=_Named)


def criterion_key(name: str) -> str:
    """Canonical form of a criterion name: whitespace-collapsed and case-folded."""
    return " ".join(name.split()).casefold()


def match_criteria(requested: Sequence[Criterion], returned: Sequence[_T]) -> list[_T]:
    """Align the judge's per-criterion entries with the requested criteria.

    Names are matched case-insensitively, with whitespace collapsed. The judge
    must return every requested criterion exactly once and nothing else. Any
    duplicate, unexpected or missing criterion raises
    :class:`ResponseParsingError`, so invalid output is never accepted
    silently. Returns the entries in the order of ``requested``.
    """
    expected = {criterion_key(c.name): c for c in requested}
    found: dict[str, _T] = {}
    problems: list[str] = []

    for entry in returned:
        key = criterion_key(entry.criterion)
        if key in found:
            problems.append(f"duplicate criterion '{entry.criterion}'")
        elif key not in expected:
            problems.append(f"unexpected criterion '{entry.criterion}'")
        else:
            found[key] = entry

    missing = [c.name for key, c in expected.items() if key not in found]
    if missing:
        problems.append(f"missing criteria {missing}")
    if problems:
        raise ResponseParsingError(
            "Judge response criteria do not match the request: " + "; ".join(problems)
        )
    return [found[criterion_key(c.name)] for c in requested]


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    """Weighted arithmetic mean. ``weights`` must be positive and the same length."""
    if len(values) != len(weights) or not values:
        raise ValueError("values and weights must be non-empty and the same length")
    total = sum(weights)
    return sum(v * w for v, w in zip(values, weights, strict=True)) / total


def decide_winner(score_a: float, score_b: float, tie_margin: float) -> Winner:
    """Pick a winner from two normalised (0-1) scores.

    Differences up to and including ``tie_margin`` count as a tie. The tiny
    epsilon absorbs floating-point noise, so a difference that is exactly the
    margin is treated consistently.
    """
    if abs(score_a - score_b) <= tie_margin + 1e-9:
        return Winner.TIE
    return Winner.A if score_a > score_b else Winner.B
