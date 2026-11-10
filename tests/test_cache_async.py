"""Opt-in result cache and the async evaluation methods."""

from __future__ import annotations

import asyncio
import threading

import pytest

from llm_judge.cache import ResultCache, cache_key, shared_cache
from llm_judge.config import Settings
from llm_judge.judge import Judge
from llm_judge.providers import FallbackProvider
from llm_judge.reliability import measure_consistency

from .conftest import (
    BarrierProvider,
    FakeProvider,
    FnProvider,
    make_entry,
    make_payload,
    make_pointwise_payload,
)
from .test_fallback import _down, _Stub

PAYLOAD = make_payload(make_entry("Helpfulness", 8, 4))


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, groq_api_key="test-key", **overrides)


def _judge(provider=None, cache_size: int = 16, **overrides) -> Judge:
    return Judge(
        provider or FakeProvider(PAYLOAD),
        _settings(**overrides),
        cache=ResultCache(cache_size) if cache_size else None,
    )


# --- ResultCache ---------------------------------------------------------------------------


def test_lru_eviction_and_stats():
    cache: ResultCache[int] = ResultCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # "a" is now most recent
    cache.put("c", 3)  # evicts "b"
    assert cache.get("b") is None
    assert cache.stats() == {"size": 2, "maxsize": 2, "hits": 1, "misses": 1}
    cache.clear()
    assert len(cache) == 0 and cache.stats()["hits"] == 0


def test_cache_needs_a_positive_size():
    with pytest.raises(ValueError):
        ResultCache(0)


def test_keys_are_stable_and_sensitive():
    assert cache_key("pairwise", {"a": 1, "b": 2}) == cache_key("pairwise", {"b": 2, "a": 1})
    assert cache_key("pairwise", {"a": 1}) != cache_key("pointwise", {"a": 1})


def test_shared_cache_is_one_per_size():
    assert shared_cache(7) is shared_cache(7)
    assert shared_cache(7) is not shared_cache(8)


# --- Judge integration ---------------------------------------------------------------------


def test_cache_is_off_by_default():
    judge = Judge(FakeProvider(PAYLOAD), _settings())
    assert judge.cache is None


def test_cache_size_setting_uses_the_shared_cache():
    judge = Judge(FakeProvider(PAYLOAD), _settings(cache_size=99))
    assert judge.cache is shared_cache(99)


def test_identical_pairwise_evaluations_hit_the_cache():
    provider = FakeProvider(PAYLOAD)
    judge = _judge(provider)
    first = judge.evaluate("t", "a", "b", "Helpfulness")
    second = judge.evaluate("t", "a", "b", "Helpfulness")
    assert provider.calls == 1
    assert "cache_hit" not in first.metadata
    assert second.metadata["cache_hit"] is True
    assert second.criteria == first.criteria


@pytest.mark.parametrize(
    "change",
    [
        {"task": "other"},
        {"b": "different"},
        {"criteria": "Accuracy"},
        {"instructions": "Be strict."},
    ],
)
def test_any_input_change_misses(change):
    def reply(messages):
        name = "Accuracy" if '"Accuracy"' in messages[0]["content"] else "Helpfulness"
        return make_payload(make_entry(name, 8, 4))

    provider = FnProvider(reply)
    judge = _judge(provider)
    base = {"task": "t", "a": "a", "b": "b", "criteria": "Helpfulness", "instructions": None}
    judge.evaluate(base["task"], base["a"], base["b"], base["criteria"])
    args = {**base, **change}
    judge.evaluate(
        args["task"], args["a"], args["b"], args["criteria"], instructions=args["instructions"]
    )
    assert provider.calls == 2


def test_judging_settings_are_part_of_the_key():
    provider = FakeProvider(PAYLOAD)
    cache: ResultCache = ResultCache(8)
    Judge(provider, _settings(), cache=cache).evaluate("t", "a", "b", "Helpfulness")
    Judge(provider, _settings(tie_margin=0.2), cache=cache).evaluate("t", "a", "b", "Helpfulness")
    assert provider.calls == 2


def test_nothing_is_cached_above_temperature_zero():
    provider = FakeProvider(PAYLOAD)
    judge = _judge(provider, temperature=0.7)
    judge.evaluate("t", "a", "b", "Helpfulness")
    judge.evaluate("t", "a", "b", "Helpfulness")
    assert provider.calls == 2 and len(judge.cache) == 0  # type: ignore[arg-type]


def test_pointwise_and_reference_are_cached_separately():
    provider = FnProvider(lambda m: make_pointwise_payload(("Correctness", 7), ("Completeness", 6)))
    judge = _judge(provider)
    judge.evaluate_reference("t", "o", "ref")
    judge.evaluate_reference("t", "o", "ref")
    judge.evaluate_reference("t", "o", "another reference")
    assert provider.calls == 2


def test_answers_from_a_fallback_are_not_cached():
    provider = FallbackProvider(_down(), [_Stub("ollama", "small")])
    judge = _judge(provider)
    judge.evaluate("t", "a", "b", "Helpfulness")
    assert len(judge.cache) == 0  # type: ignore[arg-type]


def test_consistency_analysis_never_uses_the_cache():
    provider = FakeProvider(PAYLOAD)
    judge = _judge(provider)
    report = measure_consistency(
        judge, "t", output_a="a", output_b="b", criteria=["Helpfulness"], samples=4
    )
    assert provider.calls == 4
    assert report.samples == 4
    assert judge.cache is not None  # the original judge keeps its cache


def test_without_cache_returns_an_uncached_copy():
    judge = _judge()
    clone = judge.without_cache()
    assert clone.cache is None and judge.cache is not None
    assert clone.provider is judge.provider


# --- async ----------------------------------------------------------------------------------


def test_async_methods_match_their_sync_twins():
    judge = Judge(FakeProvider(PAYLOAD), _settings())
    sync = judge.evaluate("t", "a", "b", "Helpfulness")
    asynced = asyncio.run(judge.aevaluate("t", "a", "b", "Helpfulness"))
    assert asynced.overall_winner == sync.overall_winner
    assert asynced.criteria == sync.criteria


def test_async_evaluations_run_concurrently():
    """Three awaits must be in flight together: the provider only answers when all three are."""
    provider = BarrierProvider(PAYLOAD, parties=3)
    judge = Judge(provider, _settings())

    async def run_all():
        return await asyncio.gather(
            *(judge.aevaluate("t", f"a{i}", "b", "Helpfulness") for i in range(3))
        )

    results = asyncio.run(run_all())
    assert len(results) == 3 and provider.calls == 3


def test_every_evaluation_mode_has_an_async_twin():
    provider = FnProvider(
        lambda m: (
            make_pointwise_payload(("Helpfulness", 7))
            if "Output A" not in m[0]["content"]
            else make_payload(make_entry("Helpfulness", 8, 4))
        )
    )
    judge = Judge(provider, _settings())

    async def run():
        graded = await judge.aevaluate_pointwise("t", "o")
        ranked = await judge.arank("t", ["x", "y"], strategy="pointwise")
        return graded, ranked

    graded, ranked = asyncio.run(run())
    assert graded.overall_score is not None and len(ranked.leaderboard) == 2
    for name in (
        "aevaluate",
        "aevaluate_pointwise",
        "aevaluate_reference",
        "arank",
        "aevaluate_rag",
        "aevaluate_agent",
    ):
        assert asyncio.iscoroutinefunction(getattr(Judge, name)), name
        assert "Async version" in (getattr(Judge, name).__doc__ or "")


def test_cache_is_thread_safe_under_concurrent_hits():
    provider = FakeProvider(PAYLOAD)
    judge = _judge(provider, cache_size=4)
    judge.evaluate("t", "a", "b", "Helpfulness")
    errors: list[BaseException] = []

    def hammer():
        try:
            for _ in range(50):
                judge.evaluate("t", "a", "b", "Helpfulness")
        except BaseException as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and provider.calls == 1
    assert judge.cache.stats()["hits"] == 200  # type: ignore[union-attr]
