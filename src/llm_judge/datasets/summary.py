"""Aggregate a dataset run into benchmark-style statistics.

These are plain descriptive statistics (counts, means, win rates) over the
rows that succeeded. They carry no confidence intervals; repeated-sampling
intervals are part of the analytics roadmap. Small datasets give noisy
numbers, so read differences between models with care.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import fmean

from pydantic import BaseModel, Field

from ..models import JudgeResult, PointwiseResult, Winner
from .records import DatasetKind, PairwiseRecord, PointwiseRecord
from .runner import RowOutcome, RunReport


class ModelStats(BaseModel):
    """Per-model aggregate. Pairwise fills the win/loss fields; graded runs the mean score."""

    model: str
    rows: int
    wins: int = 0
    losses: int = 0
    ties: int = 0
    win_rate: float | None = None
    mean_overall_score: float | None = None


class DatasetSummary(BaseModel):
    """Descriptive statistics for one dataset run."""

    kind: DatasetKind
    total: int
    succeeded: int
    failed: int
    resumed: int = 0
    interrupted: bool = False
    criteria: list[str] = Field(default_factory=list)
    # Pairwise only.
    wins_a: int | None = None
    wins_b: int | None = None
    ties: int | None = None
    mean_overall_score_a: float | None = None
    mean_overall_score_b: float | None = None
    # Pointwise and reference only.
    mean_overall_score: float | None = None
    mean_token_f1: float | None = None
    mean_semantic_similarity: float | None = None
    rows_with_unsupported_claims: int | None = None
    #: Mean raw score per criterion: {"A": .., "B": ..} (pairwise) or {"score": ..}.
    criteria_means: dict[str, dict[str, float]] = Field(default_factory=dict)
    models: list[ModelStats] = Field(default_factory=list)
    warning_counts: dict[str, int] = Field(default_factory=dict)


def _mean(values: list[float]) -> float | None:
    return round(fmean(values), 4) if values else None


def summarize(report: RunReport) -> DatasetSummary:
    """Summarise a :class:`RunReport`."""
    ok = [o for o in report.outcomes if o.ok and o.result is not None]
    warnings: Counter[str] = Counter()
    for outcome in ok:
        assert outcome.result is not None
        warnings.update({w.code for w in outcome.result.warnings})

    summary = DatasetSummary(
        kind=report.kind,
        total=report.total,
        succeeded=report.succeeded,
        failed=report.failed,
        resumed=report.resumed,
        interrupted=report.interrupted,
        criteria=report.criteria,
        warning_counts=dict(sorted(warnings.items())),
    )
    if report.kind == "pairwise":
        _pairwise(summary, ok)
    else:
        _graded(summary, ok)
    return summary


def _pairwise(summary: DatasetSummary, ok: list[RowOutcome]) -> None:
    winners = Counter[Winner]()
    scores_a: list[float] = []
    scores_b: list[float] = []
    per_criterion: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"A": [], "B": []})
    stats: dict[str, ModelStats] = {}

    for outcome in ok:
        result, record = outcome.result, outcome.record
        assert isinstance(result, JudgeResult) and isinstance(record, PairwiseRecord)
        winners[result.overall_winner] += 1
        if result.overall_score_a is not None:
            scores_a.append(result.overall_score_a)
        if result.overall_score_b is not None:
            scores_b.append(result.overall_score_b)
        for c in result.criteria:
            per_criterion[c.criterion]["A"].append(c.output_a_score)
            per_criterion[c.criterion]["B"].append(c.output_b_score)
        for side, model in (("A", record.model_a), ("B", record.model_b)):
            if not model:
                continue
            entry = stats.setdefault(model, ModelStats(model=model, rows=0))
            entry.rows += 1
            if result.overall_winner is Winner.TIE:
                entry.ties += 1
            elif result.overall_winner.value == side:
                entry.wins += 1
            else:
                entry.losses += 1

    for entry in stats.values():
        entry.win_rate = round((entry.wins + 0.5 * entry.ties) / entry.rows, 4)
    summary.wins_a = winners[Winner.A]
    summary.wins_b = winners[Winner.B]
    summary.ties = winners[Winner.TIE]
    summary.mean_overall_score_a = _mean(scores_a)
    summary.mean_overall_score_b = _mean(scores_b)
    summary.criteria_means = {
        name: {side: round(fmean(v), 3) for side, v in sides.items() if v}
        for name, sides in per_criterion.items()
    }
    summary.models = sorted(stats.values(), key=lambda s: (-(s.win_rate or 0), s.model))


def _graded(summary: DatasetSummary, ok: list[RowOutcome]) -> None:
    overall: list[float] = []
    f1: list[float] = []
    semantic: list[float] = []
    unsupported = 0
    per_criterion: dict[str, list[float]] = defaultdict(list)
    per_model: dict[str, list[float]] = defaultdict(list)

    for outcome in ok:
        result, record = outcome.result, outcome.record
        assert isinstance(result, PointwiseResult) and isinstance(record, PointwiseRecord)
        if result.overall_score is not None:
            overall.append(result.overall_score)
            if record.model:
                per_model[record.model].append(result.overall_score)
        for c in result.criteria:
            per_criterion[c.criterion].append(c.score)
        if result.reference_metrics is not None:
            f1.append(result.reference_metrics.token_f1)
            if result.reference_metrics.semantic_similarity is not None:
                semantic.append(result.reference_metrics.semantic_similarity)
        if result.unsupported_claims:
            unsupported += 1

    summary.mean_overall_score = _mean(overall)
    summary.criteria_means = {
        name: {"score": round(fmean(v), 3)} for name, v in per_criterion.items() if v
    }
    if summary.kind == "reference":
        summary.mean_token_f1 = _mean(f1)
        summary.mean_semantic_similarity = _mean(semantic)
        summary.rows_with_unsupported_claims = unsupported
    summary.models = sorted(
        (
            ModelStats(model=m, rows=len(v), mean_overall_score=_mean(v))
            for m, v in per_model.items()
        ),
        key=lambda s: (-(s.mean_overall_score or 0), s.model),
    )
