"""Checkpointed, resumable, bounded-concurrency dataset evaluation.

Guarantees:

* **Bounded memory and concurrency.** At most ``2 * max_workers``
  evaluations are in flight; the rest wait their turn.
* **Durable progress.** Each finished row is appended to a JSONL checkpoint
  and flushed immediately, so a crash loses at most the rows in flight.
* **Graceful interruption.** Ctrl+C (or ``stop_event``) stops new
  submissions, lets in-flight rows finish, and keeps the checkpoint.
* **Safe resume.** Rows already in the checkpoint are skipped (failed rows are
  retried by default). Resume refuses to continue if a row's content changed
  since it was checkpointed, so results from two dataset versions are never
  silently mixed.
* **Failure isolation.** A failing row is recorded with its error; it never
  aborts the run.
"""

from __future__ import annotations

import csv
import json
import logging
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from ..criteria import DEFAULT_REFERENCE_CRITERIA
from ..csv_safety import safe_rows
from ..models import Criterion, JudgeResult, PointwiseResult
from .io import DatasetError
from .records import (
    DatasetKind,
    PairwiseRecord,
    PointwiseRecord,
    Record,
    ReferenceRecord,
    record_fingerprint,
)
from .validation import ensure_single_kind

if TYPE_CHECKING:
    from ..judge import Judge

logger = logging.getLogger(__name__)

EvaluationResult = JudgeResult | PointwiseResult
CHECKPOINT_VERSION = 1


@dataclass
class RowOutcome:
    """The outcome of evaluating one record."""

    id: str
    kind: DatasetKind
    ok: bool
    record: Record
    result: EvaluationResult | None = None
    error: str | None = None
    #: True when the outcome was restored from a checkpoint instead of re-run.
    resumed: bool = False


@dataclass
class RunReport:
    """Everything a dataset run produced, in input order."""

    kind: DatasetKind
    outcomes: list[RowOutcome]
    total: int
    interrupted: bool = False
    duration_seconds: float = 0.0
    checkpoint: str | None = None
    criteria: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> int:
        return sum(1 for o in self.outcomes if o.ok)

    @property
    def failed(self) -> int:
        return sum(1 for o in self.outcomes if not o.ok)

    @property
    def resumed(self) -> int:
        return sum(1 for o in self.outcomes if o.resumed)

    @property
    def complete(self) -> bool:
        """Every record has an outcome (successful or not)."""
        return len(self.outcomes) == self.total and not self.interrupted


@dataclass
class _Checkpointed:
    fingerprint: str
    ok: bool
    result: dict[str, Any] | None
    error: str | None


class _CheckpointWriter:
    """Thread-safe append-only JSONL writer."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, outcome: RowOutcome) -> None:
        line = json.dumps(
            {
                "v": CHECKPOINT_VERSION,
                "id": outcome.id,
                "kind": outcome.kind,
                "fingerprint": record_fingerprint(outcome.record),
                "ok": outcome.ok,
                "error": outcome.error,
                "result": outcome.result.model_dump(mode="json") if outcome.result else None,
            },
            ensure_ascii=False,
        )
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def read_checkpoint(path: Path, kind: DatasetKind) -> dict[str, _Checkpointed]:
    """Load a checkpoint. Later lines for the same id win (retries append)."""
    entries: dict[str, _Checkpointed] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except (json.JSONDecodeError, RecursionError):
            if number == len(lines):  # a write cut short by a crash; that row reruns
                logger.warning("Ignoring incomplete last line of checkpoint %s", path)
                continue
            raise DatasetError(f"Checkpoint {path} is corrupt at line {number}.") from None
        if data.get("kind") != kind:
            raise DatasetError(
                f"Checkpoint {path} is for a {data.get('kind')} dataset, not {kind}."
            )
        entries[str(data["id"])] = _Checkpointed(
            fingerprint=data["fingerprint"],
            ok=bool(data["ok"]),
            result=data.get("result"),
            error=data.get("error"),
        )
    return entries


class DatasetRunner:
    """Evaluate a list of records of one kind with a shared :class:`Judge`."""

    def __init__(
        self,
        judge: Judge,
        criteria: Sequence[Criterion | str] | None = None,
        *,
        max_workers: int | None = None,
        checkpoint: str | Path | None = None,
        resume: bool = False,
        retry_failed: bool = True,
        progress: Callable[[int, int], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.judge = judge
        self.criteria = list(criteria) if criteria else None
        self.max_workers = max_workers or judge.settings.batch_max_workers
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.resume = resume
        self.retry_failed = retry_failed
        self.progress = progress
        self.stop_event = stop_event or threading.Event()

    def _criteria_for(self, kind: DatasetKind) -> list[Criterion | str]:
        if self.criteria:
            return self.criteria
        return list(DEFAULT_REFERENCE_CRITERIA) if kind == "reference" else ["Helpfulness"]

    def _evaluate(self, record: Record, kind: DatasetKind) -> RowOutcome:
        criteria = self._criteria_for(kind)
        try:
            result: EvaluationResult
            if isinstance(record, PairwiseRecord):
                result = self.judge.evaluate(
                    task_description=record.task_description,
                    output_a=record.output_a,
                    output_b=record.output_b,
                    criteria=criteria,
                    candidate_a_model=record.model_a,
                    candidate_b_model=record.model_b,
                )
            elif isinstance(record, ReferenceRecord):
                result = self.judge.evaluate_reference(
                    record.task_description,
                    record.output,
                    record.reference,
                    criteria,
                    candidate_model=record.model,
                )
            else:
                assert isinstance(record, PointwiseRecord)
                result = self.judge.evaluate_pointwise(
                    record.task_description, record.output, criteria, candidate_model=record.model
                )
        except Exception as exc:  # isolate per-row failures so one bad row never aborts the run
            logger.error("Failed to evaluate record %s: %s", record.id, exc)
            return RowOutcome(record.id, kind, ok=False, record=record, error=str(exc))
        return RowOutcome(record.id, kind, ok=True, record=record, result=result)

    def _restore(self, record: Record, kind: DatasetKind, entry: _Checkpointed) -> RowOutcome:
        result: EvaluationResult | None = None
        if entry.result is not None:
            model = JudgeResult if kind == "pairwise" else PointwiseResult
            result = model.model_validate(entry.result)
        return RowOutcome(
            record.id, kind, entry.ok, record, result=result, error=entry.error, resumed=True
        )

    def run(self, records: Sequence[Record]) -> RunReport:
        """Evaluate ``records`` and return outcomes in input order."""
        kind = ensure_single_kind(records)
        ids = [r.id for r in records]
        if len(set(ids)) != len(ids):
            raise DatasetError("Record ids must be unique to run (and resume) a dataset.")

        previous: dict[str, _Checkpointed] = {}
        if self.checkpoint and self.checkpoint.exists():
            if not self.resume:
                raise DatasetError(
                    f"Checkpoint {self.checkpoint} already exists. Resume it (--resume) "
                    "or delete it to start over."
                )
            previous = read_checkpoint(self.checkpoint, kind)

        outcomes: dict[str, RowOutcome] = {}
        pending: list[Record] = []
        for record in records:
            entry = previous.get(record.id)
            if entry is None:
                pending.append(record)
                continue
            if entry.fingerprint != record_fingerprint(record):
                raise DatasetError(
                    f"Record '{record.id}' changed since it was checkpointed. Delete the "
                    "checkpoint to re-run the new dataset version."
                )
            if entry.ok or not self.retry_failed:
                outcomes[record.id] = self._restore(record, kind, entry)
            else:
                pending.append(record)

        started = time.perf_counter()
        total = len(records)
        self._report_progress(len(outcomes), total)
        interrupted = self._execute(pending, kind, outcomes, total)

        return RunReport(
            kind=kind,
            outcomes=[outcomes[i] for i in ids if i in outcomes],
            total=total,
            interrupted=interrupted or len(outcomes) < total,
            duration_seconds=round(time.perf_counter() - started, 3),
            checkpoint=str(self.checkpoint) if self.checkpoint else None,
            criteria=[c.name if isinstance(c, Criterion) else c for c in self._criteria_for(kind)],
        )

    def _report_progress(self, done: int, total: int) -> None:
        if self.progress:
            self.progress(done, total)

    def _execute(
        self,
        pending: list[Record],
        kind: DatasetKind,
        outcomes: dict[str, RowOutcome],
        total: int,
    ) -> bool:
        """Run ``pending`` with a bounded window. Returns True if interrupted."""
        writer = _CheckpointWriter(self.checkpoint) if self.checkpoint else None
        window = self.max_workers * 2
        queue = iter(pending)
        inflight: dict[Future[RowOutcome], Record] = {}
        interrupted = False
        pool = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="llm-judge-ds")

        def collect(done: set[Future[RowOutcome]]) -> None:
            for future in done:
                inflight.pop(future)
                outcome = future.result()
                outcomes[outcome.id] = outcome
                if writer:
                    writer.write(outcome)
                self._report_progress(len(outcomes), total)

        try:
            while True:
                while len(inflight) < window and not self.stop_event.is_set():
                    record = next(queue, None)
                    if record is None:
                        break
                    inflight[pool.submit(self._evaluate, record, kind)] = record
                if not inflight:
                    break
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                collect(done)
        except KeyboardInterrupt:
            interrupted = True
            self.stop_event.set()
            logger.warning("Interrupted; finishing %d in-flight row(s)...", len(inflight))
            if inflight:
                done, _ = wait(inflight)
                collect(done)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            if writer:
                writer.close()
        return interrupted or self.stop_event.is_set()


# --- result files -------------------------------------------------------------------------


def _join(items: Sequence[str]) -> str:
    return " | ".join(items)


def flat_row(outcome: RowOutcome) -> dict[str, object]:
    """One CSV-friendly row per outcome. Pairwise columns match the v1 batch output."""
    row: dict[str, object] = {"id": outcome.id}
    record = outcome.record
    if isinstance(record, PairwiseRecord):
        result = outcome.result if isinstance(outcome.result, JudgeResult) else None
        row.update(
            {
                "model_a": record.model_a or "",
                "model_b": record.model_b or "",
                "ok": outcome.ok,
                "overall_winner": result.overall_winner.value if result else "",
                "overall_comment": result.overall_comment if result else "",
                "overall_score_a": "" if not result else result.overall_score_a,
                "overall_score_b": "" if not result else result.overall_score_b,
                "warnings": ";".join(w.code for w in result.warnings) if result else "",
                "error": outcome.error or "",
            }
        )
        for score in result.criteria if result else []:
            row[f"{score.criterion} (A)"] = score.output_a_score
            row[f"{score.criterion} (B)"] = score.output_b_score
        return row

    assert isinstance(record, PointwiseRecord)
    graded = outcome.result if isinstance(outcome.result, PointwiseResult) else None
    row.update(
        {
            "model": record.model or "",
            "ok": outcome.ok,
            "overall_score": "" if not graded else graded.overall_score,
            "feedback": graded.feedback if graded else "",
            "warnings": ";".join(w.code for w in graded.warnings) if graded else "",
            "error": outcome.error or "",
        }
    )
    for item in graded.criteria if graded else []:
        row[item.criterion] = item.score
    if isinstance(record, ReferenceRecord):
        metrics = graded.reference_metrics if graded else None
        row.update(
            {
                "token_f1": "" if metrics is None else metrics.token_f1,
                "exact_match": "" if metrics is None else metrics.exact_match,
                "missing_information": _join(graded.missing_information) if graded else "",
                "unsupported_claims": _join(graded.unsupported_claims) if graded else "",
            }
        )
    return row


def json_row(outcome: RowOutcome) -> dict[str, object]:
    """Lossless JSON form: the flat summary plus the full validated result."""
    row = flat_row(outcome)
    row["result"] = outcome.result.model_dump(mode="json") if outcome.result else None
    return row


def write_outcomes(outcomes: Sequence[RowOutcome], path: str | Path) -> None:
    """Write outcomes to ``.csv``, ``.jsonl`` or ``.json`` based on the extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = [flat_row(o) for o in outcomes]
        fieldnames: list[str] = []
        for row in rows:
            fieldnames.extend(k for k in row if k not in fieldnames)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(safe_rows(rows))
    elif suffix in (".jsonl", ".ndjson"):
        with path.open("w", encoding="utf-8") as fh:
            for outcome in outcomes:
                fh.write(json.dumps(json_row(outcome), ensure_ascii=False) + "\n")
    elif suffix == ".json":
        path.write_text(
            json.dumps([json_row(o) for o in outcomes], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        raise DatasetError(f"Unsupported output format '{suffix}'. Use .csv, .jsonl or .json.")
