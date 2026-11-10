"""A lightweight in-process job queue for dataset evaluations.

Bounded (``max_queued`` waiting jobs, ``max_running`` at once), cancellable
(through the runner's ``stop_event``), and idempotent (an ``Idempotency-Key``
returns the original job, or a conflict if the payload differs). Jobs live in
memory: they don't survive a restart, but runs recorded to an experiment do.
The queue interface is small on purpose, so it can be swapped for an external
queue without changing the API.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..datasets import DatasetKind, Record, RunReport, summarize
from .schemas import JobInfo, JobStatus

logger = logging.getLogger(__name__)

#: Runs a job: (records, stop_event, progress_callback) -> RunReport
JobRunner = Callable[[list[Record], threading.Event, Callable[[int, int], None]], RunReport]
#: Called after a job finishes, e.g. to record it as an experiment run; returns a run id.
JobRecorder = Callable[[RunReport, "Job"], str | None]


class QueueFullError(Exception):
    pass


class IdempotencyConflictError(Exception):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def payload_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


@dataclass
class Job:
    id: str
    kind: DatasetKind
    total: int
    payload_hash: str
    experiment: str | None = None
    run_name: str | None = None
    idempotency_key: str | None = None
    status: JobStatus = "queued"
    done: int = 0
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    run_id: str | None = None
    report: RunReport | None = None
    stop: threading.Event = field(default_factory=threading.Event)

    def info(self) -> JobInfo:
        return JobInfo(
            id=self.id,
            status=self.status,
            kind=self.kind,
            total=self.total,
            done=self.done,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            error=self.error,
            experiment=self.experiment,
            run_id=self.run_id,
            summary=summarize(self.report) if self.report else None,
        )


class JobManager:
    def __init__(
        self, *, max_running: int = 1, max_queued: int = 10, max_retained: int = 100
    ) -> None:
        self.max_queued = max_queued
        self.max_retained = max_retained
        self._executor = ThreadPoolExecutor(
            max_workers=max_running, thread_name_prefix="llm-judge-job"
        )
        self._lock = threading.Lock()
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._by_key: dict[str, str] = {}

    def submit(
        self,
        records: list[Record],
        kind: DatasetKind,
        run: JobRunner,
        *,
        payload: Any,
        idempotency_key: str | None = None,
        experiment: str | None = None,
        run_name: str | None = None,
        record: JobRecorder | None = None,
    ) -> tuple[Job, bool]:
        """Queue a job. Returns ``(job, created)``; ``created`` is False on idempotent replay."""
        digest = payload_hash(payload)
        with self._lock:
            if idempotency_key and idempotency_key in self._by_key:
                existing = self._jobs.get(self._by_key[idempotency_key])
                if existing is not None:
                    if existing.payload_hash != digest:
                        raise IdempotencyConflictError(idempotency_key)
                    return existing, False
            queued = sum(1 for j in self._jobs.values() if j.status == "queued")
            if queued >= self.max_queued:
                raise QueueFullError
            job = Job(
                id=uuid.uuid4().hex,
                kind=kind,
                total=len(records),
                payload_hash=digest,
                experiment=experiment,
                run_name=run_name,
                idempotency_key=idempotency_key,
            )
            self._jobs[job.id] = job
            if idempotency_key:
                self._by_key[idempotency_key] = job.id
            self._evict()
        self._executor.submit(self._execute, job, records, run, record)
        return job, True

    def _execute(
        self, job: Job, records: list[Record], run: JobRunner, record: JobRecorder | None
    ) -> None:
        if job.stop.is_set():
            job.status, job.finished_at = "cancelled", _now()
            return
        job.status, job.started_at = "running", _now()

        def progress(done: int, total: int) -> None:
            job.done = done

        try:
            report = run(records, job.stop, progress)
            job.report = report
            if record is not None:
                job.run_id = record(report, job)
            job.status = "cancelled" if job.stop.is_set() and not report.complete else "completed"
        except Exception as exc:  # a job failure must never take down the worker
            logger.exception("Job %s failed", job.id)
            job.status, job.error = "failed", str(exc)
        finally:
            job.finished_at = _now()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return list(reversed(self._jobs.values()))

    def cancel(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job is None:
            return None
        job.stop.set()
        if job.status == "queued":
            job.status, job.finished_at = "cancelled", _now()
        return job

    def _evict(self) -> None:
        finished = [
            j for j in self._jobs.values() if j.status in ("completed", "failed", "cancelled")
        ]
        while len(self._jobs) > self.max_retained and finished:
            job = finished.pop(0)
            self._jobs.pop(job.id, None)
            if job.idempotency_key:
                self._by_key.pop(job.idempotency_key, None)

    def shutdown(self) -> None:
        """Stop accepting work, cancel queued jobs and stop running ones gracefully."""
        for job in self.list():
            job.stop.set()
            if job.status == "queued":
                job.status, job.finished_at = "cancelled", _now()
        self._executor.shutdown(wait=True, cancel_futures=True)
