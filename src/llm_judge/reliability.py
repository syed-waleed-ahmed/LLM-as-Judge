"""Judge reliability analytics: repeated-judgment consistency and inter-judge agreement.

LLM judges are noisy and biased, so measure them before trusting them:

* :func:`measure_consistency` judges the same input ``samples`` times and
  reports how often the verdict agrees with the most common one, the spread
  of the scores, and a bootstrap 95% confidence interval for the mean score
  (or, pairwise, for the mean margin of A over B). At temperature 0 the spread
  only reflects provider nondeterminism. Sample at a higher temperature
  (``JUDGE_TEMPERATURE``) to probe the judge's sensitivity.
* :func:`judge_agreement` compares two stored runs of the same dataset by
  different judges (or settings): Cohen's kappa on the pairwise winners, or
  Spearman rank correlation and mean absolute difference on graded scores.

All intervals come from repeated samples under stated assumptions (independent
samples, a stable judge). None of these numbers is an LLM's self-reported
confidence, and none should be read as a calibrated probability.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from statistics import fmean, pstdev
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from .exceptions import LLMJudgeError, ValidationError
from .models import Criterion, JudgeResult, PointwiseResult

if TYPE_CHECKING:
    from .experiments import ExperimentStore
    from .judge import Judge


class CriterionSpread(BaseModel):
    mean_a: float | None = None
    std_a: float | None = None
    mean_b: float | None = None
    std_b: float | None = None
    mean: float | None = None
    std: float | None = None


class ConsistencyReport(BaseModel):
    mode: Literal["pairwise", "pointwise"]
    samples: int
    succeeded: int
    temperature: float
    #: Pairwise: verdict counts, the modal verdict, and how often samples agree with it.
    winner_counts: dict[str, int] = Field(default_factory=dict)
    modal_winner: str | None = None
    agreement_rate: float | None = None
    #: Pairwise: overall_score_a - overall_score_b. Pointwise: overall_score.
    metric: str
    mean: float | None = None
    std: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    criteria: dict[str, CriterionSpread] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def measure_consistency(
    judge: Judge,
    task_description: str,
    *,
    output_a: str | None = None,
    output_b: str | None = None,
    output: str | None = None,
    criteria: Sequence[Criterion | str] | Criterion | str = "Helpfulness",
    samples: int = 5,
    max_workers: int | None = None,
    seed: int = 0,
) -> ConsistencyReport:
    """Judge one input repeatedly and summarise how stable the judgment is.

    Give ``output_a`` and ``output_b`` for a pairwise check, or ``output`` for a
    pointwise one.
    """
    from .experiments import confidence_interval

    pairwise = output_a is not None or output_b is not None
    if pairwise and (output_a is None or output_b is None or output is not None):
        raise ValidationError("Give both output_a and output_b, or a single output.")
    if not pairwise and output is None:
        raise ValidationError("Give output_a and output_b, or output.")
    if not 2 <= samples <= 50:
        raise ValidationError("samples must be between 2 and 50.")

    # Independent samples are the point here, so never reuse cached results.
    if hasattr(judge, "without_cache"):
        judge = judge.without_cache()

    def once(_: int) -> JudgeResult | PointwiseResult | str:
        try:
            if pairwise:
                return judge.evaluate(task_description, output_a or "", output_b or "", criteria)
            return judge.evaluate_pointwise(task_description, output or "", criteria)
        except LLMJudgeError as exc:
            return str(exc)

    workers = max_workers or judge.settings.batch_max_workers
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="llm-judge-consistency"
    ) as pool:
        outcomes = list(pool.map(once, range(samples)))
    results = [o for o in outcomes if not isinstance(o, str)]
    report = ConsistencyReport(
        mode="pairwise" if pairwise else "pointwise",
        samples=samples,
        succeeded=len(results),
        temperature=judge.settings.temperature,
        metric="overall_score_a - overall_score_b" if pairwise else "overall_score",
        errors=[o for o in outcomes if isinstance(o, str)],
    )
    if judge.settings.temperature == 0:
        report.notes.append(
            "Temperature is 0, so variation only reflects provider nondeterminism; raise "
            "JUDGE_TEMPERATURE to probe the judge's sensitivity."
        )
    if len(results) < 2:
        report.notes.append("Fewer than two successful samples; no statistics computed.")
        return report

    values: list[float] = []
    per_criterion: dict[str, dict[str, list[float]]] = {}
    for result in results:
        if isinstance(result, JudgeResult):
            values.append((result.overall_score_a or 0.0) - (result.overall_score_b or 0.0))
            for c in result.criteria:
                bucket = per_criterion.setdefault(c.criterion, {"a": [], "b": []})
                bucket["a"].append(c.output_a_score)
                bucket["b"].append(c.output_b_score)
        else:
            values.append(result.overall_score or 0.0)
            for p in result.criteria:
                per_criterion.setdefault(p.criterion, {"s": []})["s"].append(p.score)

    if pairwise:
        counts = Counter(r.overall_winner.value for r in results if isinstance(r, JudgeResult))
        modal, hits = counts.most_common(1)[0]
        report.winner_counts = dict(counts)
        report.modal_winner = modal
        report.agreement_rate = round(hits / len(results), 4)
        report.criteria = {
            name: CriterionSpread(
                mean_a=round(fmean(v["a"]), 3),
                std_a=round(pstdev(v["a"]), 3),
                mean_b=round(fmean(v["b"]), 3),
                std_b=round(pstdev(v["b"]), 3),
            )
            for name, v in per_criterion.items()
        }
    else:
        report.criteria = {
            name: CriterionSpread(mean=round(fmean(v["s"]), 3), std=round(pstdev(v["s"]), 3))
            for name, v in per_criterion.items()
        }
    low, high, _method = confidence_interval(values, seed=seed)
    report.mean = round(fmean(values), 4)
    report.std = round(pstdev(values), 4)
    report.ci_low, report.ci_high = round(low, 4), round(high, 4)
    return report


# --- agreement statistics -------------------------------------------------------------------


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Cohen's kappa for two raters over the same items (1 = perfect, 0 = chance)."""
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValidationError("Kappa needs two equally long, non-empty label sequences.")
    n = len(labels_a)
    observed = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / n
    count_a, count_b = Counter(labels_a), Counter(labels_b)
    expected = sum(count_a[k] * count_b[k] for k in set(count_a) | set(count_b)) / (n * n)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1  # ties share the average rank
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation (None when either side has no variation)."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValidationError("Spearman needs two equally long sequences of at least 2 values.")
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = fmean(rx), fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    var_x = sum((a - mx) ** 2 for a in rx)
    var_y = sum((b - my) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x * var_y) ** 0.5


class JudgeAgreement(BaseModel):
    run_a: str
    run_b: str
    kind: str
    judge_a: str
    judge_b: str
    matched_rows: int
    #: Pairwise
    raw_agreement: float | None = None
    cohen_kappa: float | None = None
    #: Graded
    spearman: float | None = None
    mean_absolute_difference: float | None = None
    notes: list[str] = Field(default_factory=list)


def judge_agreement(store: ExperimentStore, run_a: str, run_b: str) -> JudgeAgreement:
    """How closely two runs over the same rows agree (e.g. two different judges)."""
    from .experiments import ExperimentError

    a, b = store.require_run(run_a), store.require_run(run_b)
    if a.kind != b.kind:
        raise ExperimentError(f"Can't compare a {a.kind} run with a {b.kind} run.")
    rows_a = {r.row_id: r for r in store.get_rows(a.id) if r.ok}
    rows_b = {r.row_id: r for r in store.get_rows(b.id) if r.ok}
    shared = [
        rid for rid in rows_a if rid in rows_b and rows_a[rid].task_hash == rows_b[rid].task_hash
    ]
    report = JudgeAgreement(
        run_a=a.id,
        run_b=b.id,
        kind=a.kind,
        judge_a=f"{a.judge_provider}/{a.judge_model}",
        judge_b=f"{b.judge_provider}/{b.judge_model}",
        matched_rows=len(shared),
    )
    if len(shared) < 2:
        report.notes.append("Fewer than two matched rows; no statistics computed.")
        return report
    if a.kind == "pairwise":
        la = [rows_a[r].overall_winner or "" for r in shared]
        lb = [rows_b[r].overall_winner or "" for r in shared]
        report.raw_agreement = round(sum(x == y for x, y in zip(la, lb, strict=True)) / len(la), 4)
        report.cohen_kappa = round(cohen_kappa(la, lb), 4)
    else:
        xs = [rows_a[r].overall_score or 0.0 for r in shared]
        ys = [rows_b[r].overall_score or 0.0 for r in shared]
        rho = spearman(xs, ys)
        report.spearman = None if rho is None else round(rho, 4)
        report.mean_absolute_difference = round(
            fmean(abs(x - y) for x, y in zip(xs, ys, strict=True)), 4
        )
        if rho is None:
            report.notes.append("One run has identical scores on every row; correlation undefined.")
    if a.judge_model == b.judge_model and a.judge_provider == b.judge_provider:
        report.notes.append("Both runs used the same judge; this measures run-to-run stability.")
    return report
