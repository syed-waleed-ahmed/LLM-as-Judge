"""The experiment store behaves the same on SQLite and PostgreSQL (e.g. Supabase).

PostgreSQL tests run when ``JUDGE_TEST_POSTGRES_URL`` points at a server
(CI starts one as a service container); otherwise they're skipped. They drop and
recreate the store's tables (run_rows, runs, experiments, meta), so point them at
a database used only for testing.
"""

from __future__ import annotations

import os

import pytest

from llm_judge.config import Settings
from llm_judge.datasets import DatasetRunner, PairwiseRecord, ReferenceRecord
from llm_judge.exceptions import ProviderError
from llm_judge.experiments import (
    ExperimentError,
    ExperimentStore,
    compare_runs,
    record_dataset_run,
    run_report,
)
from llm_judge.experiments.store import SCHEMA_VERSION, is_postgres_url, redact_url
from llm_judge.judge import Judge
from llm_judge.providers import MockProvider

from .conftest import FnProvider, make_pointwise_payload

POSTGRES_URL = os.environ.get("JUDGE_TEST_POSTGRES_URL", "")


def _reset_postgres(url: str) -> None:
    import psycopg

    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS run_rows, runs, experiments, meta CASCADE")


@pytest.fixture(params=["sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "sqlite":
        s = ExperimentStore(tmp_path / "runs.db")
    else:
        if not POSTGRES_URL:
            pytest.skip("set JUDGE_TEST_POSTGRES_URL to test the PostgreSQL store")
        _reset_postgres(POSTGRES_URL)
        s = ExperimentStore(POSTGRES_URL)
    yield s
    s.close()
    if request.param == "postgres":
        # Leave the database empty, not in a test's state (for example a future
        # schema version), so the same server is usable after the tests.
        _reset_postgres(POSTGRES_URL)


def _judge() -> Judge:
    return Judge(MockProvider(), Settings(_env_file=None, provider="mock"))


def _pairs(n: int = 3, prefix: str = "r") -> list[PairwiseRecord]:
    return [
        PairwiseRecord(
            id=f"{prefix}{i}",
            task_description=f"Explain topic {i}",
            output_a=f"A thorough, accurate explanation of topic {i}.",
            output_b="Short.",
            model_a="alpha",
            model_b="beta",
        )
        for i in range(n)
    ]


def _record(store, name, records=None, experiment="exp"):
    judge = _judge()
    report = DatasetRunner(judge, ["Helpfulness"]).run(records or _pairs())
    return record_dataset_run(store, report, judge, experiment=experiment, name=name)


def test_backend_identity(store):
    assert store.backend in ("sqlite", "postgresql")
    assert str(store.location)


def test_experiments_are_created_once(store):
    first = store.ensure_experiment("exp", "desc")
    again = store.ensure_experiment("exp", "ignored")
    assert first.id == again.id and again.description == "desc"
    assert [e.name for e in store.list_experiments()] == ["exp"]
    with pytest.raises(ExperimentError):
        store.ensure_experiment("   ")


def test_runs_round_trip_with_rows_in_order(store):
    records = _pairs(4)
    run = _record(store, "baseline", records)
    stored = store.require_run(run.id)
    assert stored.name == "baseline" and stored.experiment == "exp"
    assert stored.total == 4 and stored.succeeded == 4 and stored.kind == "pairwise"
    assert stored.criteria == ["Helpfulness"]
    assert stored.config["prompt_version"] and stored.summary["succeeded"] == 4
    rows = store.get_rows(run.id)
    assert [r.row_id for r in rows] == [r.id for r in records]  # insertion order
    assert all(r.ok and isinstance(r.ok, bool) for r in rows)
    assert rows[0].models == {"A": "alpha", "B": "beta"}
    assert rows[0].result and rows[0].result["criteria"]
    assert isinstance(rows[0].overall_score_a, float)


def test_listing_counts_pagination_and_filters(store):
    ids = [_record(store, f"run {i}").id for i in range(3)]
    other = _record(store, "elsewhere", experiment="other").id
    assert [r.id for r in store.list_runs()] == [other, *reversed(ids)]  # newest first
    assert [r.id for r in store.list_runs(experiment="exp", limit=2)] == [ids[2], ids[1]]
    assert [r.id for r in store.list_runs(experiment="exp", limit=2, offset=2)] == [ids[0]]
    assert store.count_runs() == 4 and store.count_runs(experiment="exp") == 3
    stats = store.statistics()
    assert stats["experiments"] == 2 and stats["runs"] == 4 and stats["rows_evaluated"] == 12
    assert stats["runs_by_kind"] == {"pairwise": 4}
    assert stats["runs_by_status"] == {"completed": 4}
    detail = {e.name: e.runs for e in store.list_experiments()}
    assert detail == {"exp": 3, "other": 1}


def test_prefix_lookup_escapes_wildcards_and_detects_ambiguity(store):
    run = _record(store, "one")
    assert store.require_run(run.id[:15]).id == run.id
    assert store.get_run(run.id[:8] + "%") is None  # "%" is literal, not a wildcard
    assert store.get_run("_" * len(run.id)) is None
    _record(store, "two")
    with pytest.raises(ExperimentError, match="ambiguous"):
        store.get_run(run.id[:4])
    with pytest.raises(ExperimentError, match="No run"):
        store.require_run("no-such-run")


def test_failed_rows_and_graded_runs(store):
    def reply(messages):
        if "explode" in messages[1]["content"]:
            raise ProviderError("upstream exploded")
        return make_pointwise_payload(("Correctness", 7), ("Completeness", 6), missing=[])

    judge = Judge(FnProvider(reply), Settings(_env_file=None, groq_api_key="k", max_retries=0))
    records = [
        ReferenceRecord(id="ok", task_description="t", output="fine", reference="ref"),
        ReferenceRecord(id="bad", task_description="t", output="explode", reference="ref"),
    ]
    report = DatasetRunner(judge).run(records)
    run = record_dataset_run(store, report, judge, experiment="exp", name="graded")
    assert store.require_run(run.id).status == "completed_with_errors"
    rows = {r.row_id: r for r in store.get_rows(run.id)}
    assert rows["bad"].ok is False and "upstream exploded" in (rows["bad"].error or "")
    assert rows["ok"].overall_score is not None and rows["ok"].result is not None


def test_comparison_and_report_work_on_every_backend(store):
    base = _record(store, "base")
    cand = _record(store, "cand")
    comparison = compare_runs(store, base.id, cand.id)
    assert comparison.matched_rows == 3 and comparison.verdict == "no_significant_change"
    text = run_report(store, cand.id, baseline_id=base.id)
    assert "## Comparison with the baseline" in text


def test_delete_removes_the_run_and_its_rows(store):
    run = _record(store, "doomed")
    keep = _record(store, "kept")
    store.delete_run(run.id)
    assert store.get_run(run.id) is None
    assert store.statistics()["rows_evaluated"] == 3
    assert store.require_run(keep.id)


def test_reopening_keeps_the_data(store, tmp_path):
    run = _record(store, "persisted")
    location = POSTGRES_URL if store.backend == "postgresql" else tmp_path / "runs.db"
    store.close()
    with ExperimentStore(location) as reopened:
        assert reopened.require_run(run.id).name == "persisted"
        assert len(reopened.get_rows(run.id)) == 3


def test_a_newer_schema_is_refused(store, tmp_path):
    location = POSTGRES_URL if store.backend == "postgresql" else tmp_path / "runs.db"
    with store._tx() as db:
        db.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION + 1),)
        )
    store.close()
    with pytest.raises(ExperimentError, match="newer llm-judge"):
        ExperimentStore(location)


# --- connection URLs ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "postgresql://user:secret@db.example.org:5432/app",
            "postgresql://user:***@db.example.org:5432/app",
        ),
        ("postgres://user:p%40ss@host/db", "postgres://user:***@host/db"),
        ("postgresql://user@host/db", "postgresql://user@host/db"),
        ("postgresql://host/db", "postgresql://host/db"),
    ],
)
def test_passwords_are_redacted(url, expected):
    assert redact_url(url) == expected


def test_backend_detection():
    assert is_postgres_url("postgresql://h/db") and is_postgres_url("postgres://h/db")
    assert not is_postgres_url(".llm_judge/experiments.db") and not is_postgres_url(":memory:")


def test_an_unreachable_server_fails_without_leaking_the_password():
    pytest.importorskip("psycopg")
    url = "postgresql://user:topsecret@127.0.0.1:1/db?connect_timeout=2"
    with pytest.raises(ExperimentError) as info:
        ExperimentStore(url)
    message = str(info.value)
    assert "topsecret" not in message and "***" in message
