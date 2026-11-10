"""The core judging engine.

:class:`Judge` orchestrates a single pairwise evaluation:

1. Validate the inputs (truncating oversized text, with a warning).
2. Build injection-resistant prompts and call the provider, retrying transient
   provider failures with backoff.
3. Parse and validate the JSON reply. If it is malformed or the criteria don't
   match the request exactly, send a repair retry that tells the judge what was
   wrong.
4. Clamp scores into each criterion's range, normalise them to 0-1, and
   recompute every winner from the numbers, so the result is internally
   consistent. The overall winner uses the weighted mean of normalised scores.
5. Optionally judge again with the outputs swapped (concurrently by default),
   map that run back to canonical orientation, and merge the two runs. The
   merge keeps explanations consistent with the averaged verdict.
6. Attach reliability warnings (prompt injection, verbosity, self-preference,
   position inconsistency) and judge/candidate metadata.

Every evaluation method has an ``a``-prefixed async counterpart (``aevaluate``,
``arank`` and so on) that runs it in a worker thread, so it can be awaited
without blocking an event loop. An opt-in result cache (``JUDGE_CACHE_SIZE``)
reuses identical temperature-0 judgments; see :mod:`llm_judge.cache`.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import functools
import logging
import random
import threading
import time
from collections.abc import Callable, Coroutine, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Concatenate, Literal, ParamSpec, TypeVar

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .cache import ResultCache, cache_key, shared_cache
from .config import Settings, get_settings
from .criteria import DEFAULT_REFERENCE_CRITERIA, resolve_criterion
from .diagnostics import injection_warnings, self_preference_warnings, verbosity_warning
from .exceptions import ProviderError, ResponseParsingError, ValidationError
from .models import (
    Criterion,
    CriterionScore,
    EvaluationWarning,
    JudgeResult,
    PointwiseResult,
    RankingResult,
    Winner,
)
from .parsing import parse_judge_result
from .prompts import (
    PROMPT_VERSION,
    build_repair_prompt,
    build_system_prompt,
    build_user_prompt,
    new_boundary,
)
from .providers.base import ChatMessage, LLMProvider
from .providers.embeddings import EmbeddingProvider, build_embedder
from .providers.fallback import fallback_warnings, track_fallbacks
from .scoring import criterion_key, decide_winner, match_criteria, weighted_mean

if TYPE_CHECKING:
    from .agents import AgentResult, AgentTrace
    from .rag import RagResult

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_P = ParamSpec("_P")
_R = TypeVar("_R")

CriteriaInput = Sequence[Criterion | str] | Criterion | str

# How much of a rejected reply is echoed back to the judge in a repair retry.
_MAX_ECHO_CHARS = 4_000

#: Longest single wait between retries, even when a provider asks for more.
MAX_RETRY_WAIT_SECONDS = 60.0

_backoff = wait_exponential(multiplier=0.5, max=8)


def _retry_wait(state: Any) -> float:
    """Exponential backoff, stretched to the provider's own retry-after hint if longer.

    Free-tier rate limits (e.g. Groq's tokens per minute) often need several
    seconds; retrying sooner than the provider asked just fails again. Up to
    10% random jitter keeps concurrent workers from retrying in lockstep.
    """
    wait = float(_backoff(state))
    exc = state.outcome.exception() if state.outcome else None
    hint = getattr(exc, "retry_after", None)
    if isinstance(hint, (int, float)) and hint > 0:
        wait = max(wait, float(hint) + 0.25)
    wait *= 1 + random.uniform(0, 0.1)  # nosec B311 - jitter, not security
    return min(wait, MAX_RETRY_WAIT_SECONDS)


def _sleep(seconds: float) -> None:
    """Indirection so tests can observe retry waits without sleeping."""
    time.sleep(seconds)


def _now() -> float:
    """Monotonic clock, patchable in tests together with :func:`_sleep`."""
    return time.monotonic()


class _Cooldown:
    """A shared "don't call before" time for all threads using one judge."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._until = 0.0

    def extend(self, seconds: float) -> None:
        with self._lock:
            target = _now() + min(float(seconds), MAX_RETRY_WAIT_SECONDS)
            self._until = max(self._until, target)

    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self._until - _now())

    def wait(self) -> None:
        remaining = self.remaining()
        if remaining > 0:
            logger.info("Waiting %.1fs for the provider's rate limit to reset.", remaining)
            _sleep(remaining)


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, ProviderError) and exc.retryable


def provider_name(provider: LLMProvider) -> str:
    """Return a provider's short name (its ``name`` attribute, else its class name)."""
    name = getattr(provider, "name", None)
    return name if isinstance(name, str) and name else type(provider).__name__


def _async_method(
    method: Callable[Concatenate[Judge, _P], _R],
) -> Callable[Concatenate[Judge, _P], Coroutine[Any, Any, _R]]:
    """Make an awaitable twin of a :class:`Judge` method that runs it in a worker thread."""

    @functools.wraps(method)
    async def wrapper(self: Judge, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        return await asyncio.to_thread(method, self, *args, **kwargs)

    wrapper.__doc__ = (
        f"Async version of :meth:`Judge.{method.__name__}`: same arguments and result. "
        "It runs the evaluation in a worker thread, so awaiting it doesn't block the "
        "event loop, and several can run concurrently."
    )
    return wrapper


class Judge:
    """Evaluate and compare pairs of model outputs with an LLM judge."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        settings: Settings | None = None,
        *,
        cache: ResultCache[Any] | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or self._build_default_provider(self.settings)
        #: Optional embeddings for reference grading's semantic similarity.
        self.embedder: EmbeddingProvider | None = (
            embedder if embedder is not None else build_embedder(self.settings)
        )
        #: Optional result cache. Defaults to the shared cache when JUDGE_CACHE_SIZE > 0.
        self.cache: ResultCache[Any] | None = cache
        if cache is None and self.settings.cache_size > 0:
            self.cache = shared_cache(self.settings.cache_size)
        # When the provider asks for a pause (a rate-limit retry-after hint), every
        # worker using this judge waits it out, so they don't keep re-triggering it.
        self._cooldown = _Cooldown()

    def without_cache(self) -> Judge:
        """A copy of this judge that never reads or writes the result cache."""
        clone = copy.copy(self)
        clone.cache = None
        return clone

    @staticmethod
    def _build_default_provider(settings: Settings) -> LLMProvider:
        from .providers.registry import build_provider

        return build_provider(settings)

    # -- public API ---------------------------------------------------------

    def evaluate(
        self,
        task_description: str,
        output_a: str,
        output_b: str,
        criteria: CriteriaInput,
        *,
        candidate_a_model: str | None = None,
        candidate_b_model: str | None = None,
        instructions: str | None = None,
    ) -> JudgeResult:
        """Judge ``output_a`` against ``output_b`` for ``task_description``.

        ``criteria`` may be a single name, a :class:`Criterion`, or a list of
        either. ``candidate_a_model`` and ``candidate_b_model`` optionally name
        the models that produced each output. They're recorded in the metadata
        and used to detect self-preference risk. ``instructions`` is optional
        guidance for the judge, such as how strictly to score or what to
        penalise; it is recorded in the metadata.

        Returns a validated :class:`JudgeResult`.
        """
        key = self._cache_key(
            "pairwise",
            criteria,
            {
                "task": task_description,
                "a": output_a,
                "b": output_b,
                "models": [candidate_a_model, candidate_b_model],
                "instructions": instructions,
            },
        )
        return self._cached(
            key,
            lambda: self._evaluate_pairwise(
                task_description,
                output_a,
                output_b,
                criteria,
                candidate_a_model=candidate_a_model,
                candidate_b_model=candidate_b_model,
                instructions=instructions,
            ),
        )

    def _evaluate_pairwise(
        self,
        task_description: str,
        output_a: str,
        output_b: str,
        criteria: CriteriaInput,
        *,
        candidate_a_model: str | None,
        candidate_b_model: str | None,
        instructions: str | None,
    ) -> JudgeResult:
        resolved = self.resolve_criteria(criteria)
        warnings: list[EvaluationWarning] = []
        task_description = self.validate_text("task_description", task_description, warnings)
        output_a = self.validate_text("output_a", output_a, warnings)
        output_b = self.validate_text("output_b", output_b, warnings)
        warnings.extend(injection_warnings({"A": output_a, "B": output_b}))
        instructions = self.validate_instructions(instructions, warnings)

        started = time.perf_counter()
        runs: dict[str, object] = {}
        with track_fallbacks() as fallbacks:
            if self.settings.mitigate_position_bias:
                original, swapped_raw = self._run_both_orders(
                    task_description, output_a, output_b, resolved, instructions
                )
                swapped = swapped_raw.swapped()
                result = self._merge(resolved, original, swapped, warnings)
                runs = {"original": _run_summary(original), "swapped": _run_summary(swapped)}
            else:
                result = self._evaluate_once(
                    task_description, output_a, output_b, resolved, instructions
                )
        latency = round(time.perf_counter() - started, 3)
        warnings.extend(fallback_warnings(fallbacks))
        warnings.extend(result.warnings)

        verbosity = verbosity_warning(
            output_a, output_b, result.overall_winner, self.settings.verbosity_ratio_threshold
        )
        if verbosity:
            warnings.append(verbosity)
        judge_model = self.provider.model
        warnings.extend(
            self_preference_warnings(judge_model, {"A": candidate_a_model, "B": candidate_b_model})
        )

        metadata: dict[str, object] = {
            # "model" is kept for backwards compatibility with v1.0 consumers.
            "model": judge_model,
            "judge": {"provider": provider_name(self.provider), "model": judge_model},
            "candidates": {"A": {"model": candidate_a_model}, "B": {"model": candidate_b_model}},
            "prompt_version": PROMPT_VERSION,
            "position_bias_mitigated": self.settings.mitigate_position_bias,
            "latency_seconds": latency,
        }
        if runs:
            metadata["runs"] = runs
        if instructions:
            metadata["instructions"] = instructions
        if fallbacks:
            metadata["fallback"] = [event.as_dict() for event in fallbacks]
        return result.model_copy(update={"warnings": warnings, "metadata": metadata})

    def evaluate_pointwise(
        self,
        task_description: str,
        output: str,
        criteria: CriteriaInput = "Helpfulness",
        *,
        reference: str | None = None,
        candidate_model: str | None = None,
        instructions: str | None = None,
    ) -> PointwiseResult:
        """Grade a single ``output`` on its own merits.

        With ``reference``, the output is also judged against that expected
        answer. Prefer :meth:`evaluate_reference` for that, since it defaults
        to reference-oriented criteria. ``instructions`` is optional judge
        guidance, as in :meth:`evaluate`.
        """
        from .pointwise import run_pointwise

        key = self._cache_key(
            "pointwise",
            criteria,
            {
                "task": task_description,
                "output": output,
                "reference": reference,
                "model": candidate_model,
                "instructions": instructions,
            },
        )
        return self._cached(
            key,
            lambda: run_pointwise(
                self,
                task_description,
                output,
                criteria,
                reference=reference,
                candidate_model=candidate_model,
                instructions=instructions,
            ),
        )

    def evaluate_reference(
        self,
        task_description: str,
        output: str,
        reference: str,
        criteria: CriteriaInput | None = None,
        *,
        candidate_model: str | None = None,
        instructions: str | None = None,
    ) -> PointwiseResult:
        """Grade ``output`` against an expected ``reference`` answer.

        Defaults to the "Correctness" and "Completeness" criteria. The result
        lists ``missing_information`` and ``unsupported_claims``, and includes
        ``reference_metrics``: exact match and token F1, plus semantic similarity
        when an embedding backend is configured.
        """
        return self.evaluate_pointwise(
            task_description,
            output,
            list(DEFAULT_REFERENCE_CRITERIA) if criteria is None else criteria,
            reference=reference,
            candidate_model=candidate_model,
            instructions=instructions,
        )

    def rank(
        self,
        task_description: str,
        candidates: Mapping[str, str] | Sequence[str],
        criteria: CriteriaInput = "Helpfulness",
        *,
        strategy: Literal["round_robin", "pointwise"] = "round_robin",
        candidate_models: Mapping[str, str | None] | None = None,
        max_workers: int | None = None,
        instructions: str | None = None,
    ) -> RankingResult:
        """Rank two or more candidate outputs and return a leaderboard.

        ``candidates`` is a ``{id: text}`` mapping or a list of texts (ids
        "1", "2", ...). ``"round_robin"`` compares every pair head to head.
        ``"pointwise"`` grades each candidate independently, which is cheaper.
        """
        from .ranking import run_ranking

        return run_ranking(
            self,
            task_description,
            candidates,
            criteria,
            strategy=strategy,
            candidate_models=candidate_models,
            max_workers=max_workers,
            instructions=instructions,
        )

    def evaluate_rag(
        self,
        question: str,
        contexts: Sequence[str],
        answer: str,
        *,
        reference: str | None = None,
    ) -> RagResult:
        """Evaluate a retrieval-augmented answer: faithfulness, relevance, context quality.

        ``contexts`` are the retrieved passages in rank order. With ``reference``,
        context recall is computed too. See :mod:`llm_judge.rag`.
        """
        from .rag import run_rag

        return run_rag(self, question, contexts, answer, reference=reference)

    def evaluate_agent(
        self,
        trace: AgentTrace | dict[str, Any],
        criteria: CriteriaInput | None = None,
        *,
        candidate_model: str | None = None,
    ) -> AgentResult:
        """Evaluate an agent trajectory: deterministic checks plus a judged grade.

        See :mod:`llm_judge.agents` for the trace format.
        """
        from .agents import run_agent_evaluation

        items: list[Criterion | str] | None = None
        if criteria is not None:
            items = [criteria] if isinstance(criteria, (str, Criterion)) else list(criteria)
        return run_agent_evaluation(self, trace, items, candidate_model=candidate_model)

    # -- async counterparts (each runs the method above in a worker thread) ----

    aevaluate = _async_method(evaluate)
    aevaluate_pointwise = _async_method(evaluate_pointwise)
    aevaluate_reference = _async_method(evaluate_reference)
    arank = _async_method(rank)
    aevaluate_rag = _async_method(evaluate_rag)
    aevaluate_agent = _async_method(evaluate_agent)

    # -- result cache ---------------------------------------------------------

    def _cache_key(
        self, kind: str, criteria: CriteriaInput, inputs: dict[str, object]
    ) -> str | None:
        """The cache key for an evaluation, or None when caching doesn't apply."""
        if self.cache is None or self.settings.temperature != 0:
            return None
        from . import __version__  # lazy: the package imports this module first

        s = self.settings
        return cache_key(
            kind,
            {
                **inputs,
                "criteria": [c.model_dump() for c in self.resolve_criteria(criteria)],
                "judge": [provider_name(self.provider), self.provider.model],
                "embedder": getattr(self.embedder, "model", None),
                "prompt_version": PROMPT_VERSION,
                "version": __version__,
                "settings": [
                    s.mitigate_position_bias,
                    s.tie_margin,
                    s.verbosity_ratio_threshold,
                    s.max_input_chars,
                    s.max_instructions_chars,
                ],
            },
        )

    def _cached(self, key: str | None, compute: Callable[[], _T]) -> _T:
        """Return a cached result for ``key``, or compute (and cache) a fresh one."""
        if key is None or self.cache is None:
            return compute()
        hit = self.cache.get(key)
        if hit is not None:
            return hit.model_copy(update={"metadata": {**hit.metadata, "cache_hit": True}})
        result = compute()
        if not getattr(result, "metadata", {}).get("fallback"):
            self.cache.put(key, result)
        return result

    # -- shared helpers (also used by other evaluators) -----------------------

    def resolve_criteria(self, criteria: CriteriaInput) -> list[Criterion]:
        """Turn names and :class:`Criterion` objects into a validated list."""
        items: Sequence[Criterion | str]
        if isinstance(criteria, (str, Criterion)):
            items = [criteria]
        else:
            items = criteria
        if not items:
            raise ValidationError("At least one criterion is required.")
        if len(items) > self.settings.max_criteria:
            raise ValidationError(
                f"Too many criteria ({len(items)}); the limit is {self.settings.max_criteria}."
            )
        resolved = [c if isinstance(c, Criterion) else resolve_criterion(c) for c in items]
        keys = [criterion_key(c.name) for c in resolved]
        duplicates = sorted(
            {c.name for c, k in zip(resolved, keys, strict=True) if keys.count(k) > 1}
        )
        if duplicates:
            raise ValidationError(f"Duplicate criteria: {duplicates}")
        return resolved

    def validate_text(self, field: str, value: str, warnings: list[EvaluationWarning]) -> str:
        """Reject empty text and truncate oversized text (recording a warning)."""
        if value is None or not isinstance(value, str) or not value.strip():
            raise ValidationError(f"'{field}' must be a non-empty string.")
        limit = self.settings.max_input_chars
        if len(value) > limit:
            logger.warning("Truncating '%s' from %d to %d chars.", field, len(value), limit)
            warnings.append(
                EvaluationWarning(
                    code="input_truncated",
                    message=f"'{field}' was truncated from {len(value)} to {limit} characters.",
                )
            )
            return value[:limit]
        return value

    def validate_instructions(
        self, instructions: str | None, warnings: list[EvaluationWarning]
    ) -> str | None:
        """Normalise optional judge guidance: blank becomes None, oversized is truncated."""
        if instructions is None:
            return None
        if not isinstance(instructions, str):
            raise ValidationError("'instructions' must be a string.")
        instructions = instructions.strip()
        if not instructions:
            return None
        limit = self.settings.max_instructions_chars
        if len(instructions) > limit:
            warnings.append(
                EvaluationWarning(
                    code="input_truncated",
                    message=(
                        f"'instructions' was truncated from {len(instructions)} to "
                        f"{limit} characters."
                    ),
                )
            )
            instructions = instructions[:limit]
        return instructions

    def complete_and_parse(self, messages: list[ChatMessage], parse: Callable[[str], _T]) -> _T:
        """Call the provider and parse the reply, with repair retries.

        Provider errors are retried by :meth:`_call_with_retries`. Replies that
        ``parse`` rejects with :class:`ResponseParsingError` are retried up to
        ``max_parse_retries`` times. Each retry shows the judge its rejected
        reply and the reason it was rejected.
        """
        conversation: list[ChatMessage] = list(messages)
        attempts = self.settings.max_parse_retries + 1
        for attempt in range(1, attempts + 1):
            raw: str | None = None
            try:
                # A provider may itself reject invalid JSON (e.g. Groq's JSON mode);
                # that is handled exactly like a reply that fails our own parsing.
                raw = self._call_with_retries(conversation)
                return parse(raw)
            except ResponseParsingError as exc:
                rejected = raw if raw is not None else exc.raw_response
                if exc.raw_response is None:
                    exc.raw_response = rejected
                if attempt == attempts:
                    raise
                logger.warning("Judge reply rejected (attempt %d/%d): %s", attempt, attempts, exc)
                conversation = list(messages)
                if rejected:
                    conversation.append(
                        {"role": "assistant", "content": rejected[:_MAX_ECHO_CHARS]}
                    )
                conversation.append({"role": "user", "content": build_repair_prompt(str(exc))})
        raise AssertionError("unreachable")  # pragma: no cover

    # -- internals ----------------------------------------------------------

    def _run_both_orders(
        self,
        task: str,
        output_a: str,
        output_b: str,
        criteria: list[Criterion],
        instructions: str | None = None,
    ) -> tuple[JudgeResult, JudgeResult]:
        """Judge (A, B) and (B, A). Returns the raw results of both orderings."""
        if not self.settings.parallel_bias_runs:
            return (
                self._evaluate_once(task, output_a, output_b, criteria, instructions),
                self._evaluate_once(task, output_b, output_a, criteria, instructions),
            )
        # Each worker runs in a copy of this context, so per-evaluation state
        # (such as the fallback log) is shared with the calling thread.
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm-judge-swap") as pool:
            first = pool.submit(
                contextvars.copy_context().run,
                self._evaluate_once,
                task,
                output_a,
                output_b,
                criteria,
                instructions,
            )
            second = pool.submit(
                contextvars.copy_context().run,
                self._evaluate_once,
                task,
                output_b,
                output_a,
                criteria,
                instructions,
            )
            return first.result(), second.result()

    def _evaluate_once(
        self,
        task_description: str,
        output_a: str,
        output_b: str,
        criteria: list[Criterion],
        instructions: str | None = None,
    ) -> JudgeResult:
        boundary = new_boundary(task_description, output_a, output_b)
        system = build_system_prompt(criteria, boundary=boundary, instructions=instructions)
        messages: list[ChatMessage] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": build_user_prompt(
                    task_description, output_a, output_b, boundary=boundary
                ),
            },
        ]
        return self.complete_and_parse(
            messages, lambda raw: self._normalise(criteria, parse_judge_result(raw))
        )

    def _call_with_retries(self, messages: list[ChatMessage]) -> str:
        """Call the provider, retrying only on retryable :class:`ProviderError`s."""

        @retry(
            reraise=True,
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(self.settings.max_retries + 1),
            wait=_retry_wait,
            sleep=lambda seconds: _sleep(seconds),
            before_sleep=lambda state: logger.warning(
                "Retrying judge call (attempt %s) in %.1fs after error: %s",
                state.attempt_number,
                state.next_action.sleep if state.next_action else 0.0,
                state.outcome.exception() if state.outcome else "unknown",
            ),
        )
        def _do_call() -> str:
            self._cooldown.wait()
            try:
                return self.provider.complete_json(messages)
            except ProviderError as exc:
                if exc.retryable and exc.retry_after:
                    self._cooldown.extend(exc.retry_after)
                raise

        return _do_call()

    def _winner(self, norm_a: float, norm_b: float) -> Winner:
        return decide_winner(norm_a, norm_b, self.settings.tie_margin)

    def _overall(
        self, criteria: list[Criterion], scores: list[CriterionScore]
    ) -> tuple[float, float]:
        weights = [c.weight for c in criteria]
        norm_a = [c.normalise(s.output_a_score) for c, s in zip(criteria, scores, strict=True)]
        norm_b = [c.normalise(s.output_b_score) for c, s in zip(criteria, scores, strict=True)]
        return weighted_mean(norm_a, weights), weighted_mean(norm_b, weights)

    def _normalise(self, criteria: list[Criterion], parsed: JudgeResult) -> JudgeResult:
        """Validate criteria, clamp scores and recompute every winner.

        LLMs occasionally return out-of-range scores or a ``winner`` that
        contradicts the numbers. We trust the numbers: clamp them (recording a
        warning), then derive each winner from the clamped scores.
        """
        matched = match_criteria(criteria, parsed.criteria)
        warnings: list[EvaluationWarning] = []
        normalised: list[CriterionScore] = []
        for crit, score in zip(criteria, matched, strict=True):
            a, b = crit.clamp(score.output_a_score), crit.clamp(score.output_b_score)
            if (a, b) != (score.output_a_score, score.output_b_score):
                warnings.append(
                    EvaluationWarning(
                        code="score_out_of_range",
                        criterion=crit.name,
                        message=(
                            f"Judge returned scores outside {crit.scale_min}-{crit.scale_max} "
                            f"({score.output_a_score:g} and {score.output_b_score:g}); "
                            "they were clamped into range."
                        ),
                    )
                )
            normalised.append(
                score.model_copy(
                    update={
                        "criterion": crit.name,
                        "output_a_score": a,
                        "output_b_score": b,
                        "winner": self._winner(crit.normalise(a), crit.normalise(b)),
                    }
                )
            )

        overall_a, overall_b = self._overall(criteria, normalised)
        return JudgeResult(
            criteria=normalised,
            overall_winner=self._winner(overall_a, overall_b),
            overall_comment=parsed.overall_comment,
            overall_score_a=round(overall_a, 4),
            overall_score_b=round(overall_b, 4),
            warnings=warnings,
        )

    def _merge(
        self,
        criteria: list[Criterion],
        first: JudgeResult,
        second: JudgeResult,
        warnings: list[EvaluationWarning],
    ) -> JudgeResult:
        """Average two runs (original + swapped-back) into one result.

        Both runs are in canonical orientation and in ``criteria`` order. The
        explanations and comment come from the run whose verdict matches the
        averaged verdict, so the text never argues against the numbers. When
        neither run matches, both perspectives are kept and labelled.
        """
        warnings.extend(w for w in first.warnings if w.code == "score_out_of_range")
        warnings.extend(w for w in second.warnings if w.code == "score_out_of_range")

        merged: list[CriterionScore] = []
        for crit, s1, s2 in zip(criteria, first.criteria, second.criteria, strict=True):
            a = (s1.output_a_score + s2.output_a_score) / 2
            b = (s1.output_b_score + s2.output_b_score) / 2
            winner = self._winner(crit.normalise(a), crit.normalise(b))
            if s1.winner is not s2.winner:
                warnings.append(
                    EvaluationWarning(
                        code="position_inconsistent",
                        criterion=crit.name,
                        message=(
                            "The verdict changed when the outputs were swapped "
                            f"(original order: {s1.winner.value}, swapped order: "
                            f"{s2.winner.value})."
                        ),
                    )
                )
            expl_a, expl_b = _explanations_for(winner, s1, s2)
            merged.append(
                CriterionScore(
                    criterion=crit.name,
                    output_a_score=round(a, 2),
                    output_b_score=round(b, 2),
                    output_a_explanation=expl_a,
                    output_b_explanation=expl_b,
                    winner=winner,
                )
            )

        overall_a, overall_b = self._overall(criteria, merged)
        overall = self._winner(overall_a, overall_b)
        if first.overall_winner is not second.overall_winner:
            warnings.append(
                EvaluationWarning(
                    code="position_inconsistent",
                    message=(
                        "The overall verdict changed when the outputs were swapped "
                        f"(original order: {first.overall_winner.value}, swapped order: "
                        f"{second.overall_winner.value})."
                    ),
                )
            )
        return JudgeResult(
            criteria=merged,
            overall_winner=overall,
            overall_comment=_comment_for(overall, first, second),
            overall_score_a=round(overall_a, 4),
            overall_score_b=round(overall_b, 4),
        )


def _run_summary(result: JudgeResult) -> dict[str, object]:
    return {
        "overall_winner": result.overall_winner.value,
        "overall_score_a": result.overall_score_a,
        "overall_score_b": result.overall_score_b,
    }


def _labelled(original: str, swapped: str) -> str:
    parts = []
    if original:
        parts.append(f"[original order] {original}")
    if swapped:
        parts.append(f"[swapped order] {swapped}")
    return " ".join(parts)


def _explanations_for(
    winner: Winner, first: CriterionScore, second: CriterionScore
) -> tuple[str, str]:
    """Pick explanations that agree with ``winner``, preferring the original order."""
    for run in (first, second):
        if run.winner is winner:
            return run.output_a_explanation, run.output_b_explanation
    return (
        _labelled(first.output_a_explanation, second.output_a_explanation),
        _labelled(first.output_b_explanation, second.output_b_explanation),
    )


def _comment_for(overall: Winner, first: JudgeResult, second: JudgeResult) -> str:
    """Pick an overall comment that agrees with ``overall``, preferring the original."""
    agreeing = [run for run in (first, second) if run.overall_winner is overall]
    if agreeing:
        return next((run.overall_comment for run in agreeing if run.overall_comment), "")
    return (
        "The verdict changed when the outputs were swapped (original order: "
        f"{first.overall_winner.value}, swapped order: {second.overall_winner.value}). "
        f"Averaging both orderings gives: {overall.value}."
    )
