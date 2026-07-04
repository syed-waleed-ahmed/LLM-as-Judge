"""Concurrent batch evaluation for large datasets.

Reads many task/output-A/output-B rows from CSV or JSONL, evaluates them in
parallel with a bounded thread pool, and writes the results back out. Failures
are captured per row so a single bad row never aborts the whole run — important
when grading thousands of items.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .judge import Judge
from .models import Criterion

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = ("task_description", "output_a", "output_b")


@dataclass
class BatchRecord:
    """One input row to be judged."""

    id: str
    task_description: str
    output_a: str
    output_b: str


@dataclass
class BatchRowResult:
    """The outcome of judging a single :class:`BatchRecord`."""

    id: str
    ok: bool
    overall_winner: str | None = None
    overall_comment: str | None = None
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    error: str | None = None

    def to_flat_dict(self) -> dict[str, object]:
        """Flatten into a single CSV-friendly row."""
        row: dict[str, object] = {
            "id": self.id,
            "ok": self.ok,
            "overall_winner": self.overall_winner or "",
            "overall_comment": self.overall_comment or "",
            "error": self.error or "",
        }
        for criterion, pair in self.scores.items():
            row[f"{criterion} (A)"] = pair.get("A", "")
            row[f"{criterion} (B)"] = pair.get("B", "")
        return row


def read_records(path: str | Path) -> list[BatchRecord]:
    """Read :class:`BatchRecord`s from a ``.csv`` or ``.jsonl`` file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = _read_csv(path)
    elif suffix in (".jsonl", ".ndjson"):
        rows = _read_jsonl(path)
    else:
        raise ValueError(f"Unsupported input format '{suffix}'. Use .csv or .jsonl.")

    records: list[BatchRecord] = []
    for i, row in enumerate(rows):
        missing = [f for f in _REQUIRED_FIELDS if not str(row.get(f, "")).strip()]
        if missing:
            raise ValueError(f"Row {i} is missing required field(s): {missing}")
        records.append(
            BatchRecord(
                id=str(row.get("id") or i),
                task_description=str(row["task_description"]),
                output_a=str(row["output_a"]),
                output_b=str(row["output_b"]),
            )
        )
    return records


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class BatchEvaluator:
    """Evaluate many records concurrently against a shared :class:`Judge`."""

    def __init__(self, judge: Judge, max_workers: int | None = None) -> None:
        self.judge = judge
        self.max_workers = max_workers or judge.settings.batch_max_workers

    def run(
        self,
        records: Sequence[BatchRecord],
        criteria: list[Criterion] | list[str],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[BatchRowResult]:
        """Judge every record, preserving input order in the returned list.

        ``progress`` (if given) is called as ``progress(done, total)`` after each
        record completes, for use with a progress bar.
        """
        total = len(records)
        results: list[BatchRowResult | None] = [None] * total
        done = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_index = {
                pool.submit(self._evaluate_one, record, criteria): idx
                for idx, record in enumerate(records)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                results[idx] = future.result()
                done += 1
                if progress:
                    progress(done, total)

        return [r for r in results if r is not None]

    def _evaluate_one(
        self, record: BatchRecord, criteria: list[Criterion] | list[str]
    ) -> BatchRowResult:
        try:
            result = self.judge.evaluate(
                task_description=record.task_description,
                output_a=record.output_a,
                output_b=record.output_b,
                criteria=criteria,
            )
        except Exception as exc:  # isolate per-row failures so one bad row never aborts the batch
            logger.error("Failed to judge record %s: %s", record.id, exc)
            return BatchRowResult(id=record.id, ok=False, error=str(exc))

        scores = {
            c.criterion: {"A": c.output_a_score, "B": c.output_b_score}
            for c in result.criteria
        }
        return BatchRowResult(
            id=record.id,
            ok=True,
            overall_winner=result.overall_winner.value,
            overall_comment=result.overall_comment,
            scores=scores,
        )


def write_results(results: Iterable[BatchRowResult], path: str | Path) -> None:
    """Write results to ``.csv`` or ``.jsonl`` based on the file extension."""
    path = Path(path)
    results = list(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        rows = [r.to_flat_dict() for r in results]
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    elif suffix in (".jsonl", ".ndjson"):
        with path.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported output format '{suffix}'. Use .csv or .jsonl.")
