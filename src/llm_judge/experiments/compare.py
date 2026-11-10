"""Compare two stored runs and detect regressions.

Rows are paired by id. The per-row metric is:

* pointwise/reference: the normalised overall score (0-1);
* pairwise: the margin ``overall_score_a - overall_score_b`` (-1 to 1), i.e.
  how strongly output A was preferred.

The headline is the mean paired difference (candidate minus baseline) with a
95% confidence interval: a percentile bootstrap (2,000 resamples, seeded and
therefore reproducible) for up to 5,000 pairs, or a normal approximation for
more. Assumptions: rows are independent, and the two runs judged the same
tasks the same way. Comparability warnings flag violations of the second
assumption (a different judge, prompt version or criteria).

Verdict: ``regression`` / ``improvement`` when the whole interval is below /
above zero *and* the mean change is at least ``min_effect``; otherwise
``no_significant_change``. Fewer than two matched rows gives
``insufficient_data``. These are judge-scored differences, so an LLM judge's
own noise and biases are part of what is measured.
"""

from __future__ import annotations

import math
import random
from statistics import fmean, stdev
from typing import Literal

from pydantic import BaseModel, Field

from .store import ExperimentError, ExperimentStore, RunInfo, StoredRow

Verdict = Literal["regression", "improvement", "no_significant_change", "insufficient_data"]

_BOOTSTRAP_RESAMPLES = 2000
_BOOTSTRAP_MAX_PAIRS = 5000
_TOP_ROWS = 20


class RowDelta(BaseModel):
    row_id: str
    baseline: float
    candidate: float
    delta: float
    baseline_winner: str | None = None
    candidate_winner: str | None = None


class RunComparison(BaseModel):
    baseline_id: str
    candidate_id: str
    kind: str
    metric: str
    matched_rows: int
    baseline_only: int
    candidate_only: int
    task_mismatches: int
    baseline_mean: float | None = None
    candidate_mean: float | None = None
    mean_delta: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    ci_method: Literal["bootstrap", "normal", "none"] = "none"
    confidence: float = 0.95
    min_effect: float
    verdict: Verdict
    regressed_rows: int = 0
    improved_rows: int = 0
    winner_flips: int | None = None
    baseline_failed_rows: int = 0
    candidate_failed_rows: int = 0
    regressions: list[RowDelta] = Field(default_factory=list)
    improvements: list[RowDelta] = Field(default_factory=list)
    criteria_deltas: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def _metric(row: StoredRow, kind: str) -> float | None:
    if not row.ok:
        return None
    if kind == "pairwise":
        if row.overall_score_a is None or row.overall_score_b is None:
            return None
        return row.overall_score_a - row.overall_score_b
    return row.overall_score


def confidence_interval(
    deltas: list[float], *, seed: int = 0
) -> tuple[float, float, Literal["bootstrap", "normal"]]:
    """95% CI for the mean of paired differences (see the module docstring)."""
    n = len(deltas)
    if n <= _BOOTSTRAP_MAX_PAIRS:
        rng = random.Random(seed)
        means = sorted(fmean(rng.choices(deltas, k=n)) for _ in range(_BOOTSTRAP_RESAMPLES))
        low = means[int(0.025 * _BOOTSTRAP_RESAMPLES)]
        high = means[int(0.975 * _BOOTSTRAP_RESAMPLES) - 1]
        return low, high, "bootstrap"
    mean = fmean(deltas)
    half = 1.96 * stdev(deltas) / math.sqrt(n)
    return mean - half, mean + half, "normal"


def _criteria_means(rows: list[StoredRow], names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in names:
        values = [r.flat[name] for r in rows if r.ok and isinstance(r.flat.get(name), (int, float))]
        if values:
            out[name] = fmean(values)
    return out


def _comparability(baseline: RunInfo, candidate: RunInfo) -> list[str]:
    warnings = []
    pairs = (
        ("judge model", baseline.judge_model, candidate.judge_model),
        ("judge provider", baseline.judge_provider, candidate.judge_provider),
        ("prompt version", baseline.prompt_version, candidate.prompt_version),
        ("criteria", baseline.criteria, candidate.criteria),
        (
            "criterion definitions (descriptions, scales or weights)",
            baseline.config.get("criteria"),
            candidate.config.get("criteria"),
        ),
        (
            "position-bias mitigation",
            baseline.config.get("settings", {}).get("mitigate_position_bias"),
            candidate.config.get("settings", {}).get("mitigate_position_bias"),
        ),
        (
            "judge instructions",
            baseline.config.get("instructions"),
            candidate.config.get("instructions"),
        ),
    )
    for label, a, b in pairs:
        if label.startswith("criterion definitions") and baseline.criteria != candidate.criteria:
            continue  # already reported as a change of criteria
        if a != b:
            warnings.append(
                f"The runs differ in {label} ({_short(a)} vs {_short(b)}); score differences "
                "may reflect the judging setup rather than the outputs."
            )
    return warnings


def _short(value: object, limit: int = 60) -> str:
    """A compact rendering of a config value for warning messages."""
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def compare_runs(
    store: ExperimentStore,
    baseline_id: str,
    candidate_id: str,
    *,
    min_effect: float = 0.02,
    row_threshold: float = 0.1,
    seed: int = 0,
) -> RunComparison:
    """Compare ``candidate`` against ``baseline`` (see the module docstring)."""
    baseline = store.require_run(baseline_id)
    candidate = store.require_run(candidate_id)
    if baseline.kind != candidate.kind:
        raise ExperimentError(f"Can't compare a {baseline.kind} run with a {candidate.kind} run.")
    kind = baseline.kind
    base_rows = {r.row_id: r for r in store.get_rows(baseline.id)}
    cand_rows = {r.row_id: r for r in store.get_rows(candidate.id)}
    shared = [rid for rid in base_rows if rid in cand_rows]
    mismatched = {rid for rid in shared if base_rows[rid].task_hash != cand_rows[rid].task_hash}

    warnings = _comparability(baseline, candidate)
    if mismatched:
        warnings.append(
            f"{len(mismatched)} row id(s) refer to different tasks in the two runs and were "
            "excluded."
        )

    deltas: list[RowDelta] = []
    flips = 0
    failed_base = failed_cand = 0
    for rid in shared:
        if rid in mismatched:
            continue
        b, c = base_rows[rid], cand_rows[rid]
        failed_base += not b.ok
        failed_cand += not c.ok
        mb, mc = _metric(b, kind), _metric(c, kind)
        if mb is None or mc is None:
            continue
        deltas.append(
            RowDelta(
                row_id=rid,
                baseline=round(mb, 4),
                candidate=round(mc, 4),
                delta=round(mc - mb, 4),
                baseline_winner=b.overall_winner,
                candidate_winner=c.overall_winner,
            )
        )
        if kind == "pairwise" and b.overall_winner != c.overall_winner:
            flips += 1

    comparison = RunComparison(
        baseline_id=baseline.id,
        candidate_id=candidate.id,
        kind=kind,
        metric="overall_score_a - overall_score_b" if kind == "pairwise" else "overall_score",
        matched_rows=len(deltas),
        baseline_only=len(set(base_rows) - set(cand_rows)),
        candidate_only=len(set(cand_rows) - set(base_rows)),
        task_mismatches=len(mismatched),
        min_effect=min_effect,
        verdict="insufficient_data",
        winner_flips=flips if kind == "pairwise" else None,
        baseline_failed_rows=failed_base,
        candidate_failed_rows=failed_cand,
        warnings=warnings,
    )
    # The model copied ``warnings`` on construction, so append to it directly.
    comparable = len(shared) - len(mismatched)
    if comparison.baseline_only:
        comparison.warnings.append(
            f"{comparison.baseline_only} baseline row(s) are missing from the candidate run "
            "(for example an interrupted or sampled run) and were not compared."
        )
    if failed_base or failed_cand:
        comparison.warnings.append(
            f"{failed_cand} row(s) failed in the candidate run and {failed_base} in the "
            "baseline; failed rows were excluded from the comparison."
        )
    if comparable and len(deltas) < 0.5 * comparable:
        comparison.warnings.append(
            f"Only {len(deltas)} of {comparable} matched rows succeeded in both runs, too few "
            "for a reliable verdict."
        )
        return comparison
    if kind != "pairwise":
        names = [n for n in baseline.criteria if n in candidate.criteria]
        matched = [rid for rid in shared if rid not in mismatched]
        before = _criteria_means([base_rows[r] for r in matched], names)
        after = _criteria_means([cand_rows[r] for r in matched], names)
        comparison.criteria_deltas = {
            n: round(after[n] - before[n], 4) for n in names if n in before and n in after
        }
    if len(deltas) < 2:
        return comparison

    values = [d.delta for d in deltas]
    low, high, method = confidence_interval(values, seed=seed)
    mean = fmean(values)
    comparison.baseline_mean = round(fmean(d.baseline for d in deltas), 4)
    comparison.candidate_mean = round(fmean(d.candidate for d in deltas), 4)
    comparison.mean_delta = round(mean, 4)
    comparison.ci_low, comparison.ci_high, comparison.ci_method = (
        round(low, 4),
        round(high, 4),
        method,
    )
    if high < 0 and abs(mean) >= min_effect:
        comparison.verdict = "regression"
    elif low > 0 and abs(mean) >= min_effect:
        comparison.verdict = "improvement"
    else:
        comparison.verdict = "no_significant_change"

    ordered = sorted(deltas, key=lambda d: d.delta)
    comparison.regressions = [d for d in ordered if d.delta <= -row_threshold][:_TOP_ROWS]
    comparison.improvements = [d for d in reversed(ordered) if d.delta >= row_threshold][:_TOP_ROWS]
    comparison.regressed_rows = sum(1 for d in deltas if d.delta <= -row_threshold)
    comparison.improved_rows = sum(1 for d in deltas if d.delta >= row_threshold)
    return comparison
