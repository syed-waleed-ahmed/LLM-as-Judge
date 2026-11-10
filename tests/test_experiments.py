"""Experiment tracking: store, reproducibility metadata, comparisons and CLI (Phase 5)."""

from __future__ import annotations

import json
import sqlite3

import pytest

import llm_judge.config as config_module
from llm_judge.cli import main
from llm_judge.config import Settings
from llm_judge.datasets import DatasetRunner, PairwiseRecord, load_sample
from llm_judge.exceptions import ProviderError
from llm_judge.experiments import (
    ExperimentError,
    ExperimentStore,
    compare_runs,
    confidence_interval,
    record_dataset_run,
    reproducibility_config,
)
from llm_judge.judge import Judge
from llm_judge.models import Criterion
from llm_judge.providers import MockProvider

from .conftest import FnProvider, slots


def _judge(**overrides) -> Judge:
    return Judge(provider=MockProvider(), settings=Settings(_env_file=None, **overrides))


def _record(store, records, *, name=None, experiment="exp", judge=None, criteria=None):
    judge = judge or _judge()
    report = DatasetRunner(judge, criteria).run(records)
    return record_dataset_run(store, report, judge, experiment=experiment, name=name)


def _degrade(records, ids):
    return [
        r.model_copy(update={"output": "I am not sure."}) if r.id in ids else r for r in records
    ]


@pytest.fixture
def store():
    with ExperimentStore(":memory:") as s:
        yield s


# --- store -----------------------------------------------------------------------------------


def test_experiments_are_created_once(store):
    first = store.ensure_experiment("qa", "description")
    assert store.ensure_experiment("qa").id == first.id
    assert [e.name for e in store.list_experiments()] == ["qa"]
    assert store.get_experiment("missing") is None
    with pytest.raises(ExperimentError, match="non-empty"):
        store.ensure_experiment("  ")


def test_record_and_read_a_run(store):
    info = _record(store, load_sample("general_qa"), name="v1")
    assert info.status == "completed" and info.experiment == "exp" and info.name == "v1"
    assert (info.total, info.succeeded, info.kind) == (6, 6, "reference")
    assert info.judge_provider == "mock" and info.judge_model == "mock-heuristic-v1"
    assert info.dataset_fingerprint and info.summary["mean_overall_score"] is not None
    assert info.config["llm_judge_version"] and info.config["prompt_version"]

    rows = store.get_rows(info.id)
    assert [r.row_id for r in rows] == [f"qa-{i}" for i in range(1, 7)]
    assert rows[0].overall_score is not None and rows[0].result["mode"] == "reference"
    assert store.list_experiments()[0].runs == 1


def test_raw_inputs_are_not_stored(store):
    records = load_sample("general_qa")
    info = _record(store, records)
    for row, record in zip(store.get_rows(info.id), records, strict=True):
        # No raw input fields are stored, only hashes. (Judge *outputs* such as
        # missing_information may quote the reference; the store docs say so.)
        assert not {"task_description", "output", "reference"} & set(row.flat)
        assert record.task_description not in json.dumps(row.flat)
        assert len(row.task_hash) == 16 and len(row.content_fingerprint) == 16


def test_run_statuses(store):
    def fn(messages):
        if slots(messages)[0] == "broken":
            raise ProviderError("down", retryable=False)
        return {
            "criteria": [
                {
                    "criterion": "Helpfulness",
                    "output_a_score": 7,
                    "output_b_score": 3,
                    "winner": "A",
                }
            ],
            "overall_winner": "A",
        }

    judge = Judge(provider=FnProvider(fn), settings=Settings(_env_file=None))
    records = [
        PairwiseRecord(id="1", task_description="t", output_a="fine", output_b="b"),
        PairwiseRecord(id="2", task_description="t", output_a="broken", output_b="b"),
    ]
    report = DatasetRunner(judge).run(records)
    info = record_dataset_run(store, report, judge)
    assert info.status == "completed_with_errors" and info.experiment is None
    assert [r.ok for r in store.get_rows(info.id)] == [True, False]

    report.interrupted = True
    assert record_dataset_run(store, report, judge).status == "interrupted"


def test_run_lookup_listing_and_deletion(store):
    a = _record(store, load_sample("rag"), name="a", experiment="one")
    b = _record(store, load_sample("rag"), name="b", experiment="two")
    assert store.get_run(a.id[:-2]).id == a.id  # unique prefix
    with pytest.raises(ExperimentError, match="ambiguous"):
        store.get_run(a.id[:4])  # both runs share the date prefix
    assert store.get_run("nope") is None
    with pytest.raises(ExperimentError, match="No run"):
        store.require_run("nope")

    assert [r.name for r in store.list_runs()] == ["b", "a"]
    assert [r.name for r in store.list_runs(experiment="one")] == ["a"]
    assert [r.name for r in store.list_runs(limit=1, offset=1)] == ["a"]

    store.delete_run(b.id)
    assert store.get_run(b.id) is None
    with pytest.raises(ExperimentError):
        store.get_rows(b.id)


def test_store_persists_and_rejects_newer_schemas(tmp_path):
    path = tmp_path / "sub" / "exp.db"
    with ExperimentStore(path) as s:
        run_id = _record(s, load_sample("rag")).id
    with ExperimentStore(path) as s:
        assert s.require_run(run_id).total == 5

    conn = sqlite3.connect(path)
    conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(ExperimentError, match="newer llm-judge"):
        ExperimentStore(path)


def test_reproducibility_config_has_no_secrets():
    settings = Settings(_env_file=None, groq_api_key="gsk_secret", tie_margin=0.1)
    config = reproducibility_config(settings, [Criterion(name="X", description="d", weight=2), "Y"])
    assert config["settings"]["tie_margin"] == 0.1
    assert config["criteria"][0]["weight"] == 2 and config["criteria"][1] == {"name": "Y"}
    assert "secret" not in json.dumps(config)


# --- comparisons -----------------------------------------------------------------------------


def test_detects_a_regression_and_the_rows_responsible(store):
    base = _record(store, load_sample("general_qa"))
    cand = _record(store, _degrade(load_sample("general_qa"), {"qa-1", "qa-3", "qa-5"}))
    c = compare_runs(store, base.id, cand.id)
    assert c.verdict == "regression"
    assert c.ci_high < 0 and c.ci_method == "bootstrap"
    assert {d.row_id for d in c.regressions} == {"qa-1", "qa-3", "qa-5"}
    assert c.regressed_rows == 3 and c.improved_rows == 0
    assert c.criteria_deltas["Correctness"] < 0
    assert c.warnings == []


def test_detects_an_improvement(store):
    worse = _record(store, _degrade(load_sample("general_qa"), {"qa-1", "qa-3", "qa-5"}))
    better = _record(store, load_sample("general_qa"))
    assert compare_runs(store, worse.id, better.id).verdict == "improvement"


def test_identical_runs_show_no_change(store):
    a = _record(store, load_sample("rag"))
    b = _record(store, load_sample("rag"))
    c = compare_runs(store, a.id, b.id)
    assert c.verdict == "no_significant_change" and c.mean_delta == 0


def test_comparison_edge_cases(store):
    single = _record(store, load_sample("rag")[:1])
    other = _record(store, load_sample("rag")[:3])
    c = compare_runs(store, single.id, other.id)
    assert c.verdict == "insufficient_data" and c.candidate_only == 2

    pairwise = _record(store, load_sample("safety"))
    with pytest.raises(ExperimentError, match="Can't compare a reference run"):
        compare_runs(store, single.id, pairwise.id)


def test_comparability_and_task_mismatch_warnings(store):
    records = load_sample("rag")
    base = _record(store, records, criteria=["Correctness"])
    retasked = [records[0].model_copy(update={"task_description": "something else"}), *records[1:]]
    cand = _record(store, retasked, criteria=["Correctness", "Completeness"])
    c = compare_runs(store, base.id, cand.id)
    assert c.task_mismatches == 1 and c.matched_rows == 4
    assert any("criteria" in w for w in c.warnings)
    assert any("different tasks" in w for w in c.warnings)


def test_pairwise_comparison_uses_the_margin_and_counts_flips(store):
    records = load_sample("safety")
    base = _record(store, records)
    swapped = [
        r.model_copy(update={"output_a": r.output_b, "output_b": r.output_a}) for r in records
    ]
    cand = _record(store, swapped)
    c = compare_runs(store, base.id, cand.id)
    assert c.metric == "overall_score_a - overall_score_b"
    assert c.winner_flips >= 4 and c.verdict in (
        "regression",
        "improvement",
        "no_significant_change",
    )


def test_confidence_interval_methods():
    low, high, method = confidence_interval([0.1, 0.2, 0.3, 0.4], seed=1)
    assert method == "bootstrap" and 0.1 <= low <= high <= 0.4
    assert confidence_interval([0.1, 0.2, 0.3, 0.4], seed=1) == (low, high, method)
    values = [((i % 7) - 3) / 10 for i in range(6000)]
    low, high, method = confidence_interval(values)
    assert method == "normal" and low < 0 < high


# --- CLI ------------------------------------------------------------------------------


@pytest.fixture
def mock_cli(monkeypatch, tmp_path):
    monkeypatch.setattr(
        config_module,
        "get_settings",
        lambda: Settings(_env_file=None, provider="mock", db_path=str(tmp_path / "exp.db")),
    )
    return tmp_path


def _run_ids(capsys) -> list[str]:
    main(["experiments", "runs"])
    return [line.split()[0] for line in capsys.readouterr().out.splitlines() if line.strip()]


def test_cli_records_compares_exports_and_deletes(mock_cli, capsys):
    data = mock_cli / "qa.jsonl"
    main(["datasets", "export", "general_qa", "--output", str(data)])
    main(
        ["batch", "--input", str(data), "--output", str(mock_cli / "v1.csv"), "--experiment", "qa"]
    )
    assert "Recorded run" in capsys.readouterr().out

    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["id"] in ("qa-1", "qa-3", "qa-5"):
            row["output"] = "I am not sure."
    v2 = mock_cli / "qa_v2.jsonl"
    v2.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    main(["benchmark", str(v2), "--experiment", "qa", "--run-name", "v2"])
    capsys.readouterr()

    candidate, baseline = _run_ids(capsys)
    assert main(["experiments", "list"]) == 0
    assert "qa" in capsys.readouterr().out

    assert main(["experiments", "compare", baseline, candidate]) == 0
    assert "verdict: REGRESSION" in capsys.readouterr().out
    assert main(["experiment", "compare", baseline, candidate, "--fail-on-regression"]) == 1
    capsys.readouterr()
    assert main(["experiments", "compare", baseline, candidate, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "regression"

    assert main(["experiments", "show", candidate]) == 0
    assert "judge mock/mock-heuristic-v1" in capsys.readouterr().out
    assert main(["experiments", "show", candidate, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "v2"

    for suffix in ("csv", "jsonl"):
        out = mock_cli / f"export.{suffix}"
        assert main(["experiments", "export", candidate, "--output", str(out)]) == 0
        assert out.read_text(encoding="utf-8").count("qa-") >= 6
    capsys.readouterr()

    assert main(["experiments", "delete", candidate]) == 0
    capsys.readouterr()
    assert _run_ids(capsys) == [baseline]


def test_cli_empty_database_and_unknown_run(mock_cli, capsys):
    assert main(["experiments", "list"]) == 0
    assert "No experiments" in capsys.readouterr().out
    assert main(["experiments", "runs"]) == 0
    assert "No runs" in capsys.readouterr().out
    assert main(["experiments", "show", "nope"]) == 2
    assert "No run with id" in capsys.readouterr().err
