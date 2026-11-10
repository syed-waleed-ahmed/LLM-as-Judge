"""Experiment tracking: persist runs, compare them, and detect regressions.

Quick start::

    from llm_judge import Judge
    from llm_judge.datasets import DatasetRunner, load_dataset
    from llm_judge.experiments import ExperimentStore, compare_runs, record_dataset_run

    store = ExperimentStore()            # .llm_judge/experiments.db
    records, report = load_dataset("qa.jsonl")
    judge = Judge()
    run = DatasetRunner(judge).run(records)
    info = record_dataset_run(store, run, judge, experiment="qa", dataset_report=report)
    ...
    print(compare_runs(store, baseline_id, info.id).verdict)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .compare import RowDelta, RunComparison, compare_runs, confidence_interval
from .store import (
    Experiment,
    ExperimentError,
    ExperimentStore,
    RunInfo,
    StoredRow,
    reproducibility_config,
)

if TYPE_CHECKING:
    from ..datasets import DatasetReport, RunReport
    from ..judge import Judge


def record_dataset_run(
    store: ExperimentStore,
    report: RunReport,
    judge: Judge,
    *,
    experiment: str | None = None,
    name: str | None = None,
    dataset_report: DatasetReport | None = None,
    dataset_source: str | None = None,
    notes: str = "",
) -> RunInfo:
    """Store a dataset run with the judge's identity and reproducibility metadata."""
    from ..datasets import dataset_fingerprint
    from ..judge import provider_name

    records = [o.record for o in report.outcomes]
    config: dict[str, Any] = reproducibility_config(judge.settings, report.criteria)  # type: ignore[arg-type]
    return store.record_run(
        report,
        judge_provider=provider_name(judge.provider),
        judge_model=judge.provider.model,
        config=config,
        experiment=experiment,
        name=name,
        dataset_source=dataset_source or (dataset_report.source if dataset_report else ""),
        dataset_fingerprint=(
            dataset_report.fingerprint if dataset_report else dataset_fingerprint(records)
        ),
        notes=notes,
    )


__all__ = [
    "Experiment",
    "ExperimentError",
    "ExperimentStore",
    "RowDelta",
    "RunComparison",
    "RunInfo",
    "StoredRow",
    "compare_runs",
    "confidence_interval",
    "record_dataset_run",
    "reproducibility_config",
]
