from __future__ import annotations

import json

import pytest

from llm_judge.batch import (
    BatchEvaluator,
    BatchRecord,
    read_records,
    write_results,
)
from llm_judge.judge import Judge

from .conftest import FakeProvider


def _judge(settings, payload):
    return Judge(provider=FakeProvider(payload), settings=settings)


def test_read_records_from_csv(tmp_path):
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(
        "id,task_description,output_a,output_b\n1,do it,alpha,beta\n",
        encoding="utf-8",
    )
    records = read_records(csv_path)
    assert len(records) == 1
    assert records[0].id == "1"
    assert records[0].output_a == "alpha"


def test_read_records_from_jsonl(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text(
        json.dumps({"task_description": "t", "output_a": "a", "output_b": "b"}) + "\n",
        encoding="utf-8",
    )
    records = read_records(path)
    assert records[0].id == "0"  # falls back to row index


def test_read_records_missing_field_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("task_description,output_a\nt,a\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_records(csv_path)


def test_read_records_unknown_format(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError):
        read_records(path)


def test_batch_evaluator_runs_all_records(hermetic_settings, single_criterion_payload):
    judge = _judge(hermetic_settings, single_criterion_payload)
    evaluator = BatchEvaluator(judge, max_workers=4)
    records = [
        BatchRecord(id=str(i), task_description="t", output_a="a", output_b="b")
        for i in range(5)
    ]
    seen = []
    results = evaluator.run(records, ["Helpfulness"], progress=lambda d, t: seen.append((d, t)))

    assert len(results) == 5
    assert all(r.ok for r in results)
    assert [r.id for r in results] == ["0", "1", "2", "3", "4"]  # input order preserved
    assert seen[-1] == (5, 5)


def test_batch_isolates_per_row_failures(hermetic_settings):
    # Invalid payload (empty criteria) makes every judgment fail, but the batch
    # itself must complete and report the failures.
    judge = _judge(hermetic_settings, {"criteria": [], "overall_winner": "A"})
    evaluator = BatchEvaluator(judge, max_workers=2)
    records = [BatchRecord(id="x", task_description="t", output_a="a", output_b="b")]
    results = evaluator.run(records, ["Helpfulness"])
    assert results[0].ok is False
    assert results[0].error


def test_write_results_csv_roundtrip(tmp_path, hermetic_settings, single_criterion_payload):
    judge = _judge(hermetic_settings, single_criterion_payload)
    evaluator = BatchEvaluator(judge, max_workers=2)
    records = [BatchRecord(id="7", task_description="t", output_a="a", output_b="b")]
    results = evaluator.run(records, ["Helpfulness"])

    out = tmp_path / "out.csv"
    write_results(results, out)
    content = out.read_text(encoding="utf-8")
    assert "overall_winner" in content
    assert "Helpfulness (A)" in content
    assert "7" in content


def test_write_results_jsonl(tmp_path, hermetic_settings, single_criterion_payload):
    judge = _judge(hermetic_settings, single_criterion_payload)
    evaluator = BatchEvaluator(judge, max_workers=2)
    records = [BatchRecord(id="7", task_description="t", output_a="a", output_b="b")]
    results = evaluator.run(records, ["Helpfulness"])

    out = tmp_path / "out.jsonl"
    write_results(results, out)
    line = out.read_text(encoding="utf-8").strip()
    parsed = json.loads(line)
    assert parsed["id"] == "7"
    assert parsed["ok"] is True
