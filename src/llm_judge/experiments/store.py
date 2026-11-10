"""SQLite-backed experiment store: runs, per-row results and reproducibility metadata.

SQLite ships with Python, so persistence needs no server, account or extra
dependency. The database is a single file (default ``.llm_judge/experiments.db``,
see ``JUDGE_DB_PATH``).

Privacy: raw task text, outputs and references are **not** stored, only their
content hashes (which match rows across runs) plus the judge's results.
Explanations in those results can still quote the evaluated text, so treat the
database like the dataset it came from.
"""

from __future__ import annotations

import hashlib
import json
import platform
import secrets
import sqlite3
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..config import Settings
from ..exceptions import LLMJudgeError
from ..models import RESULT_SCHEMA_VERSION, Criterion, JudgeResult, PointwiseResult
from ..prompts import PROMPT_VERSION

SCHEMA_VERSION = 1

RunStatus = Literal["completed", "completed_with_errors", "interrupted"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    experiment_id INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    dataset_source TEXT NOT NULL DEFAULT '',
    dataset_fingerprint TEXT,
    judge_provider TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    criteria TEXT NOT NULL,
    total INTEGER NOT NULL,
    succeeded INTEGER NOT NULL,
    failed INTEGER NOT NULL,
    duration_seconds REAL NOT NULL,
    summary TEXT NOT NULL,
    config TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_dataset ON runs(dataset_fingerprint);
CREATE TABLE IF NOT EXISTS run_rows (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    row_id TEXT NOT NULL,
    ok INTEGER NOT NULL,
    error TEXT,
    task_hash TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL,
    models TEXT NOT NULL,
    overall_winner TEXT,
    overall_score REAL,
    overall_score_a REAL,
    overall_score_b REAL,
    flat TEXT NOT NULL,
    result TEXT,
    PRIMARY KEY (run_id, row_id)
);
"""


class ExperimentError(LLMJudgeError):
    """An experiment, run or comparison request is invalid."""


class Experiment(BaseModel):
    id: int
    name: str
    description: str = ""
    created_at: str
    runs: int = 0


class RunInfo(BaseModel):
    """A stored run: what was evaluated, how, and the headline results."""

    id: str
    experiment: str | None
    name: str
    created_at: str
    kind: str
    status: RunStatus
    dataset_source: str
    dataset_fingerprint: str | None
    judge_provider: str
    judge_model: str
    prompt_version: str
    criteria: list[str]
    total: int
    succeeded: int
    failed: int
    duration_seconds: float
    summary: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class StoredRow(BaseModel):
    """One row of a stored run."""

    row_id: str
    ok: bool
    error: str | None
    task_hash: str
    content_fingerprint: str
    models: dict[str, str | None]
    overall_winner: str | None
    overall_score: float | None
    overall_score_a: float | None
    overall_score_b: float | None
    flat: dict[str, Any]
    result: dict[str, Any] | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def reproducibility_config(
    settings: Settings, criteria: list[Criterion | str] | None
) -> dict[str, Any]:
    """Everything needed to understand (and repeat) how a run was judged."""
    from .. import __version__

    criteria_info: list[dict[str, Any]] = []
    for c in criteria or []:
        if isinstance(c, Criterion):
            criteria_info.append(c.model_dump())
        else:
            criteria_info.append({"name": c})
    return {
        "llm_judge_version": __version__,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "python": platform.python_version(),
        "platform": sys.platform,
        "criteria": criteria_info,
        "settings": settings.model_dump(
            include={
                "provider",
                "temperature",
                "mitigate_position_bias",
                "parallel_bias_runs",
                "tie_margin",
                "max_parse_retries",
                "max_retries",
                "max_input_chars",
            }
        ),
    }


class ExperimentStore:
    """A small, thread-safe repository of experiments and runs in one SQLite file."""

    def __init__(self, path: str | Path = ".llm_judge/experiments.db") -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    # -- lifecycle ---------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> ExperimentStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock, self._conn:
            yield self._conn

    def _migrate(self) -> None:
        with self._tx() as conn:
            conn.executescript(_SCHEMA)
            row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif int(row["value"]) > SCHEMA_VERSION:
                raise ExperimentError(
                    f"{self.path} was created by a newer llm-judge (schema {row['value']})."
                )

    # -- experiments ---------------------------------------------------------------

    def ensure_experiment(self, name: str, description: str = "") -> Experiment:
        """Return the experiment called ``name``, creating it if needed."""
        name = name.strip()
        if not name:
            raise ExperimentError("Experiment names must be non-empty.")
        with self._tx() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO experiments (name, description, created_at) "
                "VALUES (?, ?, ?)",
                (name, description, _now()),
            )
        experiment = self.get_experiment(name)
        assert experiment is not None
        return experiment

    def get_experiment(self, name: str) -> Experiment | None:
        with self._tx() as conn:
            row = conn.execute(
                "SELECT e.*, COUNT(r.id) AS runs FROM experiments e "
                "LEFT JOIN runs r ON r.experiment_id = e.id WHERE e.name = ? GROUP BY e.id",
                (name,),
            ).fetchone()
        return Experiment(**dict(row)) if row else None

    def list_experiments(self) -> list[Experiment]:
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT e.*, COUNT(r.id) AS runs FROM experiments e "
                "LEFT JOIN runs r ON r.experiment_id = e.id GROUP BY e.id ORDER BY e.created_at"
            ).fetchall()
        return [Experiment(**dict(r)) for r in rows]

    # -- runs --------------------------------------------------------------------------

    def record_run(
        self,
        report: Any,
        *,
        judge_provider: str,
        judge_model: str,
        config: dict[str, Any],
        experiment: str | None = None,
        name: str | None = None,
        dataset_source: str = "",
        dataset_fingerprint: str | None = None,
        notes: str = "",
    ) -> RunInfo:
        """Persist a :class:`~llm_judge.datasets.RunReport` and its rows."""
        from ..datasets import record_fingerprint, summarize
        from ..datasets.records import PairwiseRecord
        from ..datasets.runner import flat_row

        run_id = _new_run_id()
        experiment_id = self.ensure_experiment(experiment).id if experiment else None
        if report.complete and report.failed == 0:
            status: RunStatus = "completed"
        elif report.complete:
            status = "completed_with_errors"
        else:
            status = "interrupted"
        summary = summarize(report).model_dump(mode="json")

        rows = []
        for outcome in report.outcomes:
            record = outcome.record
            result = outcome.result
            if isinstance(record, PairwiseRecord):
                models: dict[str, str | None] = {"A": record.model_a, "B": record.model_b}
            else:
                models = {"output": record.model}
            rows.append(
                (
                    run_id,
                    outcome.id,
                    int(outcome.ok),
                    outcome.error,
                    _hash(record.task_description),
                    record_fingerprint(record),
                    json.dumps(models),
                    result.overall_winner.value if isinstance(result, JudgeResult) else None,
                    result.overall_score if isinstance(result, PointwiseResult) else None,
                    result.overall_score_a if isinstance(result, JudgeResult) else None,
                    result.overall_score_b if isinstance(result, JudgeResult) else None,
                    json.dumps(flat_row(outcome), ensure_ascii=False, default=str),
                    json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
                    if result
                    else None,
                )
            )

        with self._tx() as conn:
            conn.execute(
                "INSERT INTO runs (id, experiment_id, name, created_at, kind, status, "
                "dataset_source, dataset_fingerprint, judge_provider, judge_model, "
                "prompt_version, criteria, total, succeeded, failed, duration_seconds, "
                "summary, config, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?)",
                (
                    run_id,
                    experiment_id,
                    name or run_id,
                    _now(),
                    report.kind,
                    status,
                    dataset_source,
                    dataset_fingerprint,
                    judge_provider,
                    judge_model,
                    config.get("prompt_version", PROMPT_VERSION),
                    json.dumps(report.criteria),
                    report.total,
                    report.succeeded,
                    report.failed,
                    report.duration_seconds,
                    json.dumps(summary),
                    json.dumps(config, default=str),
                    notes,
                ),
            )
            conn.executemany(
                "INSERT INTO run_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
        info = self.get_run(run_id)
        assert info is not None
        return info

    def _run_from_row(self, row: sqlite3.Row) -> RunInfo:
        data = dict(row)
        for key in ("criteria", "summary", "config"):
            data[key] = json.loads(data[key])
        data.pop("experiment_id", None)
        return RunInfo(**data)

    _RUN_SELECT = (
        "SELECT r.*, e.name AS experiment FROM runs r LEFT JOIN experiments e "
        "ON e.id = r.experiment_id"
    )

    def get_run(self, run_id: str) -> RunInfo | None:
        """Look up a run by id, or by a unique id prefix."""
        with self._tx() as conn:
            rows = conn.execute(
                f"{self._RUN_SELECT} WHERE r.id = ? OR r.id LIKE ? ORDER BY r.id = ? DESC",
                (run_id, f"{run_id}%", run_id),
            ).fetchall()
        if not rows:
            return None
        if len(rows) > 1 and rows[0]["id"] != run_id:
            raise ExperimentError(f"Run id prefix '{run_id}' is ambiguous ({len(rows)} matches).")
        return self._run_from_row(rows[0])

    def require_run(self, run_id: str) -> RunInfo:
        run = self.get_run(run_id)
        if run is None:
            raise ExperimentError(f"No run with id '{run_id}'.")
        return run

    def list_runs(
        self, *, experiment: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[RunInfo]:
        """Most recent runs first, optionally filtered by experiment."""
        query = self._RUN_SELECT
        params: list[Any] = []
        if experiment is not None:
            query += " WHERE e.name = ?"
            params.append(experiment)
        # rowid follows insertion order, unlike created_at (1 s resolution).
        query += " ORDER BY r.rowid DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self._tx() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._run_from_row(r) for r in rows]

    def count_runs(self, *, experiment: str | None = None) -> int:
        query = "SELECT COUNT(*) FROM runs r LEFT JOIN experiments e ON e.id = r.experiment_id"
        params: list[Any] = []
        if experiment is not None:
            query += " WHERE e.name = ?"
            params.append(experiment)
        with self._tx() as conn:
            return int(conn.execute(query, params).fetchone()[0])

    def statistics(self) -> dict[str, Any]:
        """Aggregate counts across the whole store."""
        with self._tx() as conn:
            experiments = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            rows = conn.execute("SELECT COUNT(*) FROM run_rows").fetchone()[0]
            by_kind = conn.execute("SELECT kind, COUNT(*) FROM runs GROUP BY kind").fetchall()
            by_status = conn.execute("SELECT status, COUNT(*) FROM runs GROUP BY status").fetchall()
        return {
            "experiments": int(experiments),
            "runs": int(runs),
            "rows_evaluated": int(rows),
            "runs_by_kind": {k: int(n) for k, n in by_kind},
            "runs_by_status": {k: int(n) for k, n in by_status},
        }

    def get_rows(self, run_id: str) -> list[StoredRow]:
        run = self.require_run(run_id)
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT * FROM run_rows WHERE run_id = ? ORDER BY rowid", (run.id,)
            ).fetchall()
        out = []
        for row in rows:
            data = dict(row)
            data.pop("run_id")
            data["ok"] = bool(data["ok"])
            data["models"] = json.loads(data["models"])
            data["flat"] = json.loads(data["flat"])
            data["result"] = json.loads(data["result"]) if data["result"] else None
            out.append(StoredRow(**data))
        return out

    def delete_run(self, run_id: str) -> None:
        run = self.require_run(run_id)
        with self._tx() as conn:
            conn.execute("DELETE FROM runs WHERE id = ?", (run.id,))
