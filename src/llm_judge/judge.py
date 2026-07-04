"""The core judging engine.

:class:`Judge` orchestrates a single pairwise evaluation:

1. Validate and (if configured) truncate inputs.
2. Build the prompts and call the provider, retrying transient failures.
3. Parse and validate the JSON response.
4. Clamp scores into each criterion's range and recompute winners so the
   returned result is internally consistent.
5. Optionally repeat with the outputs swapped to mitigate position bias, then
   merge the two runs.
"""

from __future__ import annotations

import logging
import time

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import Settings, get_settings
from .criteria import resolve_criterion
from .exceptions import ProviderError, ValidationError
from .models import Criterion, CriterionScore, JudgeResult, Winner
from .parsing import parse_judge_result
from .prompts import build_system_prompt, build_user_prompt
from .providers.base import LLMProvider
from .providers.groq_provider import GroqProvider

logger = logging.getLogger(__name__)

# Scores within this absolute difference are treated as a tie when we recompute
# winners from the numeric scores.
_TIE_THRESHOLD = 0.5


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, ProviderError) and exc.retryable


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
        criteria: list[Criterion] | list[str] | str,
    ) -> JudgeResult:
        """Judge ``output_a`` against ``output_b`` for ``task_description``.

        ``criteria`` may be a single name, a list of names, or a list of
        :class:`Criterion` objects. Returns a validated :class:`JudgeResult`.
        """
        resolved = self._resolve_criteria(criteria)
        task_description = self._validate_text("task_description", task_description)
        output_a = self._validate_text("output_a", output_a)
        output_b = self._validate_text("output_b", output_b)

        started = time.perf_counter()
        result = self._evaluate_once(task_description, output_a, output_b, resolved)

        if self.settings.mitigate_position_bias:
            swapped_raw = self._evaluate_once(
                task_description, output_b, output_a, resolved
            )
            swapped_back = JudgeResult(
                criteria=[c.swapped() for c in swapped_raw.criteria],
                overall_winner=swapped_raw.overall_winner.swapped(),
                overall_comment=swapped_raw.overall_comment,
            )
            result = self._merge(resolved, result, swapped_back)

        result.metadata.update(
            {
                "model": self.provider.model,
                "position_bias_mitigated": self.settings.mitigate_position_bias,
                "latency_seconds": round(time.perf_counter() - started, 3),
            }
        )
        return result

    # -- internals ----------------------------------------------------------

    def _evaluate_once(
        self,
        task_description: str,
        output_a: str,
        output_b: str,
        criteria: list[Criterion],
    ) -> JudgeResult:
        system_prompt = build_system_prompt(criteria)
        user_prompt = build_user_prompt(task_description, output_a, output_b)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw = self._call_with_retries(messages)
        result = parse_judge_result(raw)
        return self._normalise(criteria, result)

    def _call_with_retries(self, messages: list[dict[str, str]]) -> str:
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
            return self.provider.complete_json(messages)  # type: ignore[arg-type]

        return _do_call()

    def _normalise(self, criteria: list[Criterion], result: JudgeResult) -> JudgeResult:
        """Clamp scores to each criterion's range and recompute winners.

        LLMs occasionally return out-of-range scores or a ``winner`` that
        contradicts the numbers. We trust the numbers: clamp them, then derive
        each per-criterion winner and the overall winner from the clamped scores.
        """
        by_name = {c.name: c for c in criteria}
        normalised: list[CriterionScore] = []
        for score in result.criteria:
            crit = by_name.get(score.criterion)
            lo, hi = (crit.scale_min, crit.scale_max) if crit else (0, 100)
            a = _clamp(score.output_a_score, lo, hi)
            b = _clamp(score.output_b_score, lo, hi)
            normalised.append(
                score.model_copy(
                    update={
                        "output_a_score": a,
                        "output_b_score": b,
                        "winner": _winner_from_scores(a, b),
                    }
                )
            )

        mean_a = sum(s.output_a_score for s in normalised) / len(normalised)
        mean_b = sum(s.output_b_score for s in normalised) / len(normalised)
        return result.model_copy(
            update={
                "criteria": normalised,
                "overall_winner": _winner_from_scores(mean_a, mean_b),
            }
        )

    def _merge(
        self, criteria: list[Criterion], first: JudgeResult, second: JudgeResult
    ) -> JudgeResult:
        """Average two runs (original + swapped) into one bias-mitigated result."""
        second_by_name = {c.criterion: c for c in second.criteria}
        merged: list[CriterionScore] = []
        for s1 in first.criteria:
            s2 = second_by_name.get(s1.criterion, s1)
            a = (s1.output_a_score + s2.output_a_score) / 2
            b = (s1.output_b_score + s2.output_b_score) / 2
            merged.append(
                CriterionScore(
                    criterion=s1.criterion,
                    output_a_score=round(a, 2),
                    output_b_score=round(b, 2),
                    output_a_explanation=s1.output_a_explanation,
                    output_b_explanation=s1.output_b_explanation,
                    winner=_winner_from_scores(a, b),
                )
            )

        mean_a = sum(s.output_a_score for s in merged) / len(merged)
        mean_b = sum(s.output_b_score for s in merged) / len(merged)
        return JudgeResult(
            criteria=merged,
            overall_winner=_winner_from_scores(mean_a, mean_b),
            overall_comment=first.overall_comment,
        )

    def _resolve_criteria(
        self, criteria: list[Criterion] | list[str] | str
    ) -> list[Criterion]:
        if isinstance(criteria, str):
            criteria = [criteria]
        if not criteria:
            raise ValidationError("At least one criterion is required.")
        resolved = [
            c if isinstance(c, Criterion) else resolve_criterion(c) for c in criteria
        ]
        return resolved

    def _validate_text(self, field: str, value: str) -> str:
        if value is None or not value.strip():
            raise ValidationError(f"'{field}' must be a non-empty string.")
        limit = self.settings.max_input_chars
        if len(value) > limit:
            logger.warning("Truncating '%s' from %d to %d chars.", field, len(value), limit)
            return value[:limit]
        return value


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _winner_from_scores(a: float, b: float) -> Winner:
    if abs(a - b) <= _TIE_THRESHOLD:
        return Winner.TIE
    return Winner.A if a > b else Winner.B
