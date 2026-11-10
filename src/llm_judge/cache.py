"""Opt-in, in-memory cache of evaluation results.

Set ``JUDGE_CACHE_SIZE`` (for example ``256``) to reuse the result of an
evaluation whose inputs, criteria, instructions, judge and judging settings
are all identical to an earlier one. It's off by default.

It is deliberately conservative:

* **Temperature 0 only.** At a higher temperature the caller asked for
  sampling variation, so nothing is cached.
* **Exact keys.** The key is a SHA-256 of everything that can change the
  result: the evaluation kind, the texts, the full criterion definitions,
  the instructions, the provider and model, the prompt version, and the
  settings that affect scoring and warnings.
* **Labelled hits.** A cached result carries ``metadata["cache_hit"] = True``.
* **No fallback answers.** Results that a fallback provider produced are not
  cached, so a recovered primary is used again.
* **Not for consistency analysis**, which needs independent samples; it
  uses :meth:`~llm_judge.judge.Judge.without_cache`.

The cache lives in process memory only. Nothing is written to disk.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any, Generic, TypeVar

_V = TypeVar("_V")


class ResultCache(Generic[_V]):
    """A thread-safe least-recently-used cache with hit and miss counters."""

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self.maxsize = maxsize
        self._data: OrderedDict[str, _V] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> _V | None:
        with self._lock:
            value = self._data.get(key)
            if value is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, value: _V) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._data),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
            }


def cache_key(kind: str, payload: dict[str, Any]) -> str:
    """A stable SHA-256 key for an evaluation kind and its JSON-serialisable inputs."""
    blob = json.dumps({"kind": kind, **payload}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


_shared: dict[int, ResultCache[Any]] = {}
_shared_lock = threading.Lock()


def shared_cache(maxsize: int) -> ResultCache[Any]:
    """The process-wide cache of a given size, shared by every :class:`Judge` using it."""
    with _shared_lock:
        cache = _shared.get(maxsize)
        if cache is None:
            cache = _shared[maxsize] = ResultCache(maxsize)
        return cache


__all__ = ["ResultCache", "cache_key", "shared_cache"]
