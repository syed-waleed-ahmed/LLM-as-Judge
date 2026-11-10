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
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, TypeVar

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

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
from .providers.groq_provider import GroqProvider
from .scoring import criterion_key, decide_winner, match_criteria, weighted_mean

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

CriteriaInput = Sequence[Criterion | str] | Criterion | str

# How much of a rejected reply is echoed back to the judge in a repair retry.
_MAX_ECHO_CHARS = 4_000


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, ProviderError) and exc.retryable


def provider_name(provider: LLMProvider) -> str:
    """Return a provider's short name (its ``name`` attribute, else its class name)."""
    name = getattr(provider, "name", None)
    return name if isinstance(name, str) and name else type(provider).__name__


class Judge:
    """Evaluate and compare pairs of model outputs with an LLM judge."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or self._build_default_provider(self.settings)

    @staticmethod
    def _build_default_provider(settings: Settings) -> LLMProvider:
        return GroqProvider(
            api_key=settings.require_api_key(),
            model=settings.groq_model,
            temperature=settings.temperature,
            timeout=settings.request_timeout,
        )

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
    ) -> JudgeResult:
        """Judge ``output_a`` against ``output_b`` for ``task_description``.

        ``criteria`` may be a single name, a :class:`Criterion`, or a list of
        either. ``candidate_a_model`` and ``candidate_b_model`` optionally name
        the models that produced each output. They're recorded in the metadata
        and used to detect self-preference risk.

        Returns a validated :class:`JudgeResult`.
        """
        resolved = self.resolve_criteria(criteria)
        warnings: list[EvaluationWarning] = []
        task_description = self.validate_text("task_description", task_description, warnings)
        output_a = self.validate_text("output_a", output_a, warnings)
        output_b = self.validate_text("output_b", output_b, warnings)
        warnings.extend(injection_warnings({"A": output_a, "B": output_b}))

        started = time.perf_counter()
        runs: dict[str, object] = {}
        if self.settings.mitigate_position_bias:
            original, swapped_raw = self._run_both_orders(
                task_description, output_a, output_b, resolved
            )
            swapped = swapped_raw.swapped()
            result = self._merge(resolved, original, swapped, warnings)
            runs = {"original": _run_summary(original), "swapped": _run_summary(swapped)}
        else:
            result = self._evaluate_once(task_description, output_a, output_b, resolved)
        latency = round(time.perf_counter() - started, 3)
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
        return result.model_copy(update={"warnings": warnings, "metadata": metadata})

    def evaluate_pointwise(
        self,
        task_description: str,
        output: str,
        criteria: CriteriaInput = "Helpfulness",
        *,
        reference: str | None = None,
        candidate_model: str | None = None,
    ) -> PointwiseResult:
        """Grade a single ``output`` on its own merits.

        With ``reference``, the output is also judged against that expected
        answer. Prefer :meth:`evaluate_reference` for that, since it defaults
        to reference-oriented criteria.
        """
        from .pointwise import run_pointwise

        return run_pointwise(
            self,
            task_description,
            output,
            criteria,
            reference=reference,
            candidate_model=candidate_model,
        )

    def evaluate_reference(
        self,
        task_description: str,
        output: str,
        reference: str,
        criteria: CriteriaInput | None = None,
        *,
        candidate_model: str | None = None,
    ) -> PointwiseResult:
        """Grade ``output`` against an expected ``reference`` answer.

        Defaults to the "Correctness" and "Completeness" criteria. The result
        lists ``missing_information`` and ``unsupported_claims``, and includes
        deterministic ``reference_metrics`` (exact match and token F1).
        """
        from .pointwise import run_pointwise

        return run_pointwise(
            self,
            task_description,
            output,
            list(DEFAULT_REFERENCE_CRITERIA) if criteria is None else criteria,
            reference=reference,
            candidate_model=candidate_model,
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
        )

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
            raw = self._call_with_retries(conversation)
            try:
                return parse(raw)
            except ResponseParsingError as exc:
                if exc.raw_response is None:
                    exc.raw_response = raw
                if attempt == attempts:
                    raise
                logger.warning("Judge reply rejected (attempt %d/%d): %s", attempt, attempts, exc)
                conversation = [
                    *messages,
                    {"role": "assistant", "content": raw[:_MAX_ECHO_CHARS]},
                    {"role": "user", "content": build_repair_prompt(str(exc))},
                ]
        raise AssertionError("unreachable")  # pragma: no cover

    # -- internals ----------------------------------------------------------

    def _run_both_orders(
        self, task: str, output_a: str, output_b: str, criteria: list[Criterion]
    ) -> tuple[JudgeResult, JudgeResult]:
        """Judge (A, B) and (B, A). Returns the raw results of both orderings."""
        if not self.settings.parallel_bias_runs:
            return (
                self._evaluate_once(task, output_a, output_b, criteria),
                self._evaluate_once(task, output_b, output_a, criteria),
            )
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm-judge-swap") as pool:
            first = pool.submit(self._evaluate_once, task, output_a, output_b, criteria)
            second = pool.submit(self._evaluate_once, task, output_b, output_a, criteria)
            return first.result(), second.result()

    def _evaluate_once(
        self,
        task_description: str,
        output_a: str,
        output_b: str,
        criteria: list[Criterion],
    ) -> JudgeResult:
        boundary = new_boundary(task_description, output_a, output_b)
        messages: list[ChatMessage] = [
            {"role": "system", "content": build_system_prompt(criteria, boundary=boundary)},
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
            wait=wait_exponential(multiplier=0.5, max=8),
            before_sleep=lambda state: logger.warning(
                "Retrying judge call (attempt %s) after error: %s",
                state.attempt_number,
                state.outcome.exception() if state.outcome else "unknown",
            ),
        )
        def _do_call() -> str:
            return self.provider.complete_json(messages)

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
