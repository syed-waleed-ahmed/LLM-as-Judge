"""Multi-candidate ranking: compare two or more outputs and build a leaderboard.

Two strategies are available:

* ``"round_robin"`` judges every pair of candidates head to head with the
  pairwise judge: n(n-1)/2 comparisons, doubled when position-bias mitigation
  is on. Candidates are ranked by win rate (a tie counts as half a win), with
  ties broken by the mean normalised overall score. Presentation order
  alternates across pairs, so no candidate is always shown first.
* ``"pointwise"`` grades each candidate independently (n calls) and ranks by
  overall score. It's cheaper but less discriminative than head-to-head
  comparison.

Individual comparison failures are isolated. They're recorded, reported as
warnings, and excluded from the standings. Ranking fails only if every
comparison fails.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .exceptions import LLMJudgeError, ValidationError
from .models import (
    Criterion,
    EvaluationWarning,
    JudgeResult,
    PairwiseMatch,
    PointwiseResult,
    RankedCandidate,
    RankingResult,
    Winner,
)
from .prompts import PROMPT_VERSION

if TYPE_CHECKING:
    from .judge import Judge

Strategy = Literal["round_robin", "pointwise"]

# Scores are compared to this precision when deciding whether two candidates
# share a rank.
_RANK_PRECISION = 6


@dataclass
class _Standing:
    wins: int = 0
    losses: int = 0
    ties: int = 0
    overall_scores: list[float] = field(default_factory=list)

    @property
    def played(self) -> int:
        return self.wins + self.losses + self.ties


def normalise_candidates(candidates: Mapping[str, str] | Sequence[str]) -> dict[str, str]:
    """Accept ``{"id": text}`` or a list of texts (ids become "1", "2", ...)."""
    if isinstance(candidates, Mapping):
        items = {str(k).strip(): v for k, v in candidates.items()}
        if any(not k for k in items):
            raise ValidationError("Candidate ids must be non-empty.")
        if len(items) != len(candidates):
            raise ValidationError("Candidate ids must be unique after trimming whitespace.")
        return items
    if isinstance(candidates, str):
        raise ValidationError("Pass candidates as a list of texts or a {id: text} mapping.")
    return {str(i): text for i, text in enumerate(candidates, start=1)}


def _assign_ranks(rows: list[tuple[str, float, float | None]]) -> dict[str, int]:
    """Competition ranking ("1224") over (candidate, score, tie-break) rows."""

    def key(row: tuple[str, float, float | None]) -> tuple[float, float]:
        return (round(row[1], _RANK_PRECISION), round(row[2] or 0.0, _RANK_PRECISION))

    ordered = sorted(rows, key=key, reverse=True)
    ranks: dict[str, int] = {}
    for position, row in enumerate(ordered, start=1):
        if position > 1 and key(row) == key(ordered[position - 2]):
            ranks[row[0]] = ranks[ordered[position - 2][0]]
        else:
            ranks[row[0]] = position
    return ranks


def run_ranking(
    judge: Judge,
    task_description: str,
    candidates: Mapping[str, str] | Sequence[str],
    criteria: Sequence[Criterion | str] | Criterion | str,
    *,
    strategy: Strategy = "round_robin",
    candidate_models: Mapping[str, str | None] | None = None,
    max_workers: int | None = None,
    instructions: str | None = None,
) -> RankingResult:
    """Rank candidates for ``task_description``; see :meth:`Judge.rank`."""
    texts = normalise_candidates(candidates)
    limit = judge.settings.max_candidates
    if not 2 <= len(texts) <= limit:
        raise ValidationError(f"Ranking needs between 2 and {limit} candidates, got {len(texts)}.")
    models = dict(candidate_models or {})
    unknown = sorted(set(models) - set(texts))
    if unknown:
        raise ValidationError(f"candidate_models names unknown candidates: {unknown}")
    resolved = judge.resolve_criteria(criteria)
    # Validate every text up front so a bad candidate fails fast, not mid-tournament.
    judge.validate_text("task_description", task_description, [])
    for cid, text in texts.items():
        judge.validate_text(f"candidate '{cid}'", text, [])
    workers = max_workers or judge.settings.batch_max_workers
    guidance_warnings: list[EvaluationWarning] = []
    instructions = judge.validate_instructions(instructions, guidance_warnings)

    started = time.perf_counter()
    if strategy == "round_robin":
        result = _round_robin(
            judge, task_description, texts, resolved, models, workers, instructions
        )
    elif strategy == "pointwise":
        result = _pointwise(judge, task_description, texts, resolved, models, workers, instructions)
    else:
        raise ValidationError(f"Unknown ranking strategy '{strategy}'.")

    from .judge import provider_name

    result.metadata = {
        "judge": {"provider": provider_name(judge.provider), "model": judge.provider.model},
        "candidates": {cid: {"model": models.get(cid)} for cid in texts},
        "prompt_version": PROMPT_VERSION,
        "position_bias_mitigated": judge.settings.mitigate_position_bias,
        "latency_seconds": round(time.perf_counter() - started, 3),
    }
    if instructions:
        result.metadata["instructions"] = instructions
    result.warnings = guidance_warnings + result.warnings
    return result


def _round_robin(
    judge: Judge,
    task: str,
    texts: dict[str, str],
    criteria: list[Criterion],
    models: dict[str, str | None],
    workers: int,
    instructions: str | None = None,
) -> RankingResult:
    ids = list(texts)
    pairs = list(itertools.combinations(range(len(ids)), 2))

    def play(pair: tuple[int, int]) -> PairwiseMatch:
        i, j = pair
        a, b = ids[i], ids[j]
        # Alternate who is shown first so position bias doesn't systematically
        # favour candidates that appear early in the list.
        first, second = (b, a) if (i + j) % 2 else (a, b)
        try:
            judged: JudgeResult = judge.evaluate(
                task,
                texts[first],
                texts[second],
                criteria,
                candidate_a_model=models.get(first),
                candidate_b_model=models.get(second),
                instructions=instructions,
            )
        except LLMJudgeError as exc:
            return PairwiseMatch(
                candidate_a=a, candidate_b=b, presented_first=first, error=str(exc)
            )
        if first != a:  # map the judgment back to (a, b) orientation
            judged = judged.swapped()
        winner = {Winner.A: a, Winner.B: b, Winner.TIE: "tie"}[judged.overall_winner]
        return PairwiseMatch(
            candidate_a=a,
            candidate_b=b,
            presented_first=first,
            winner=winner,
            overall_score_a=judged.overall_score_a,
            overall_score_b=judged.overall_score_b,
            warnings=sorted({w.code for w in judged.warnings}),
        )

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="llm-judge-rank") as pool:
        matches = list(pool.map(play, pairs))

    failed = [m for m in matches if m.error]
    if len(failed) == len(matches):
        raise LLMJudgeError(
            f"All {len(matches)} comparisons failed; first error: {failed[0].error}"
        )

    standings = {cid: _Standing() for cid in ids}
    for m in matches:
        if m.error:
            continue
        sa, sb = standings[m.candidate_a], standings[m.candidate_b]
        if m.winner == "tie":
            sa.ties += 1
            sb.ties += 1
        elif m.winner == m.candidate_a:
            sa.wins += 1
            sb.losses += 1
        else:
            sb.wins += 1
            sa.losses += 1
        if m.overall_score_a is not None:
            sa.overall_scores.append(m.overall_score_a)
        if m.overall_score_b is not None:
            sb.overall_scores.append(m.overall_score_b)

    warnings = [
        EvaluationWarning(
            code="comparison_failed",
            message=f"{m.candidate_a} vs {m.candidate_b} failed and was excluded: {m.error}",
        )
        for m in failed
    ]
    rows: list[tuple[str, float, float | None]] = []
    for cid, st in standings.items():
        if st.played == 0:
            warnings.append(
                EvaluationWarning(
                    code="candidate_unranked",
                    candidate=cid,
                    message=f"Candidate '{cid}' has no successful comparisons and is unranked.",
                )
            )
            continue
        mean = sum(st.overall_scores) / len(st.overall_scores) if st.overall_scores else None
        rows.append((cid, (st.wins + 0.5 * st.ties) / st.played, mean))

    ranks = _assign_ranks(rows)
    leaderboard = sorted(
        (
            RankedCandidate(
                candidate=cid,
                rank=ranks[cid],
                score=round(score, 4),
                mean_overall_score=None if mean is None else round(mean, 4),
                wins=standings[cid].wins,
                losses=standings[cid].losses,
                ties=standings[cid].ties,
                model=models.get(cid),
            )
            for cid, score, mean in rows
        ),
        key=lambda r: (r.rank, r.candidate),
    )
    return RankingResult(
        strategy="round_robin",
        criteria=[c.name for c in criteria],
        leaderboard=leaderboard,
        matches=matches,
        warnings=warnings,
    )


def _pointwise(
    judge: Judge,
    task: str,
    texts: dict[str, str],
    criteria: list[Criterion],
    models: dict[str, str | None],
    workers: int,
    instructions: str | None = None,
) -> RankingResult:
    def grade(cid: str) -> tuple[str, PointwiseResult | None, str | None]:
        try:
            return (
                cid,
                judge.evaluate_pointwise(
                    task,
                    texts[cid],
                    criteria,
                    candidate_model=models.get(cid),
                    instructions=instructions,
                ),
                None,
            )
        except LLMJudgeError as exc:
            return cid, None, str(exc)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="llm-judge-rank") as pool:
        outcomes = list(pool.map(grade, texts))

    grades = {cid: res for cid, res, _ in outcomes if res is not None}
    if not grades:
        raise LLMJudgeError(f"All {len(outcomes)} gradings failed; first error: {outcomes[0][2]}")
    warnings = [
        EvaluationWarning(
            code="candidate_unranked",
            candidate=cid,
            message=f"Grading candidate '{cid}' failed and it is unranked: {err}",
        )
        for cid, res, err in outcomes
        if res is None
    ]
    rows = [(cid, res.overall_score or 0.0, res.overall_score) for cid, res in grades.items()]
    ranks = _assign_ranks(rows)
    leaderboard = sorted(
        (
            RankedCandidate(
                candidate=cid,
                rank=ranks[cid],
                score=score,
                mean_overall_score=mean,
                model=models.get(cid),
            )
            for cid, score, mean in rows
        ),
        key=lambda r: (r.rank, r.candidate),
    )
    return RankingResult(
        strategy="pointwise",
        criteria=[c.name for c in criteria],
        leaderboard=leaderboard,
        grades=grades,
        warnings=warnings,
    )
