"""Datasets: readers, validation, records, runner (checkpoint/resume), summary, samples."""

from __future__ import annotations

import csv
import json
import threading

import pytest

from llm_judge.config import Settings
from llm_judge.datasets import (
    SAMPLE_DATASETS,
    DatasetError,
    DatasetRunner,
    PairwiseRecord,
    PointwiseRecord,
    ReferenceRecord,
    dataset_fingerprint,
    infer_kind,
    iter_rows,
    load_dataset,
    load_sample,
    read_checkpoint,
    record_fingerprint,
    sample_records,
    sample_text,
    split_records,
    summarize,
    validate_dataset,
    write_outcomes,
)
from llm_judge.datasets.records import canonical_columns
from llm_judge.datasets.samples import get_sample
from llm_judge.datasets.validation import ensure_single_kind
from llm_judge.exceptions import ProviderError
from llm_judge.judge import Judge
from llm_judge.providers import MockProvider

from .conftest import FakeProvider, FnProvider, make_pointwise_payload, single_output, slots


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def _write(path, text: str):
    path.write_text(text, encoding="utf-8")
    return path


def _jsonl(path, rows: list[dict]):
    return _write(path, "".join(json.dumps(r) + "\n" for r in rows))


PAIR = {"task_description": "t", "output_a": "a", "output_b": "b"}


# --- readers ------------------------------------------------------------------------------


def test_reads_csv_jsonl_json_and_yaml(tmp_path):
    csv_path = _write(tmp_path / "d.csv", "task_description,output_a,output_b\nt,a,b\n")
    jsonl = _jsonl(tmp_path / "d.jsonl", [PAIR])
    as_list = _write(tmp_path / "d.json", json.dumps([PAIR]))
    wrapped = _write(tmp_path / "w.json", json.dumps({"records": [PAIR]}))
    yaml_path = _write(tmp_path / "d.yaml", "- task_description: t\n  output_a: a\n  output_b: b\n")
    for path in (csv_path, jsonl, as_list, wrapped, yaml_path):
        assert list(iter_rows(path)) == [PAIR], path.name


@pytest.mark.parametrize(
    "name,content,message",
    [
        ("d.txt", "x", "Unsupported dataset format"),
        ("d.jsonl", '{"a": 1}\nnot json\n', "line 2: invalid JSON"),
        ("d.jsonl", "[1, 2]\n", "line 1: expected a JSON object"),
        ("d.json", '{"rows": []}', "expected a list of records"),
        ("d.json", "[1]", "record 1 is not an object"),
        ("d.json", "{broken", "invalid JSON"),
        ("d.yaml", "a: [unclosed", "invalid YAML"),
        ("d.csv", "a,b\n1,2,3\n", "too many values"),
    ],
)
def test_reader_errors(tmp_path, name, content, message):
    with pytest.raises(DatasetError, match=message):
        list(iter_rows(_write(tmp_path / name, content)))


def test_reader_limits(tmp_path):
    path = _jsonl(tmp_path / "d.jsonl", [PAIR] * 5)
    with pytest.raises(DatasetError, match="JUDGE_MAX_DATASET_BYTES"):
        list(iter_rows(path, max_bytes=10))
    with pytest.raises(DatasetError, match="more than 3 rows"):
        list(iter_rows(path, max_rows=3))
    with pytest.raises(DatasetError, match="not found"):
        list(iter_rows(tmp_path / "missing.csv"))


def test_empty_csv_yields_nothing(tmp_path):
    assert list(iter_rows(_write(tmp_path / "e.csv", ""))) == []


# --- records --------------------------------------------------------------------------------


def test_kind_inference_and_aliases():
    assert infer_kind({"output_a", "output_b", "task_description"}) == "pairwise"
    assert infer_kind({"output", "reference"}) == "reference"
    assert infer_kind({"output"}) == "pointwise"
    assert infer_kind({"text"}) is None
    assert canonical_columns(["Question", "response", "ground_truth"]) == {
        "Question": "task_description",
        "response": "output",
        "ground_truth": "reference",
    }
    # An alias never shadows a canonical column that is already present.
    assert canonical_columns(["task_description", "prompt"])["prompt"] == "prompt"


def test_fingerprints():
    r1 = PairwiseRecord(id="1", **PAIR)
    r2 = PairwiseRecord(id="2", **PAIR)
    assert record_fingerprint(r1) == record_fingerprint(r2)  # the id is excluded
    assert record_fingerprint(r1) != record_fingerprint(r1.model_copy(update={"output_a": "z"}))
    assert dataset_fingerprint([r1, r2]) != dataset_fingerprint([r2, r1])


def test_sampling_and_splitting_are_reproducible():
    items = list(range(100))
    sample = sample_records(items, 10, seed=3)
    assert sample == sample_records(items, 10, seed=3)
    assert sample == sorted(sample) and len(sample) == 10
    assert sample_records(items, 500) == items

    train, test = split_records(items, (0.8, 0.2), seed=1)
    assert (len(train), len(test)) == (80, 20)
    assert sorted(train + test) == items and not set(train) & set(test)
    assert split_records(items, (0.8, 0.2), seed=1) == [train, test]

    with pytest.raises(Exception, match="sum to 1"):
        split_records(items, (0.5, 0.2))
    with pytest.raises(Exception, match="non-negative"):
        sample_records(items, -1)


# --- validation ----------------------------------------------------------------------------


def test_valid_dataset_report(tmp_path):
    path = _write(
        tmp_path / "d.csv",
        "id,prompt,output_a,output_b,model_a,notes\n"
        "x1,Say hi,Hello!,Hey,alpha,first\n"
        "x2,Say bye,Goodbye,Bye,alpha,\n",
    )
    records, report = load_dataset(path, settings=_settings())
    assert report.ok
    assert (report.kind, report.rows, report.valid_rows) == ("pairwise", 2, 2)
    assert report.aliases == {"prompt": "task_description"}
    assert report.unknown_columns == ["notes"]
    assert report.fingerprint == dataset_fingerprint(records)
    assert records[0] == PairwiseRecord(
        id="x1", task_description="Say hi", output_a="Hello!", output_b="Hey", model_a="alpha"
    )
    assert report.preview[0]["id"] == "x1"


def test_invalid_rows_are_reported_not_fatal(tmp_path):
    rows = [
        {"id": "ok", **PAIR},
        {"id": "dup", **PAIR, "output_a": "other"},
        {"id": "dup", **PAIR, "output_a": "another"},
        {"id": "empty", "task_description": "t", "output_a": " ", "output_b": "b"},
        {"id": "same", **PAIR},  # identical content to "ok"
        {"id": "huge", **PAIR, "output_b": "x" * 50},
    ]
    records, report = load_dataset(
        _jsonl(tmp_path / "d.jsonl", rows), settings=_settings(max_input_chars=20)
    )
    assert not report.ok
    assert [r.id for r in records] == ["ok", "same", "huge"]
    assert report.duplicate_ids == ["dup"]
    assert report.duplicate_content == [["ok", "same"]]
    assert report.error_count == 2  # missing field + duplicate id
    assert report.warning_count == 2  # duplicate content + oversized field
    assert report.issues[0].severity == "error"


def test_unknown_kind_and_explicit_kind(tmp_path):
    report = validate_dataset(_jsonl(tmp_path / "d.jsonl", [{"text": "x"}]), settings=_settings())
    assert report.kind is None and not report.ok and report.error_count == 1

    rows = [{"task_description": "t", "output": "o", "reference": "r"}]
    records, report = load_dataset(rows, kind="pointwise", settings=_settings())
    assert isinstance(records[0], PointwiseRecord) and not isinstance(records[0], ReferenceRecord)
    assert report.unknown_columns == ["reference"]
    with pytest.raises(DatasetError, match="Unknown dataset kind"):
        load_dataset(rows, kind="weird", settings=_settings())  # type: ignore[arg-type]


def test_in_memory_sources_and_limits():
    record = ReferenceRecord(id="r", task_description="t", output="o", reference="ref")
    records, report = load_dataset([record], settings=_settings())
    assert records == [record] and report.format == "python"
    with pytest.raises(DatasetError, match="More than 1 rows"):
        load_dataset([PAIR, PAIR], settings=_settings(max_dataset_rows=1))


def test_issue_list_is_capped(tmp_path):
    rows = [{"task_description": "t", "output": ""} for _ in range(150)]
    report = validate_dataset(rows, kind="pointwise", settings=_settings())
    assert len(report.issues) == 100 and report.issues_truncated == 50
    assert report.error_count == 150


def test_mixed_kinds_cannot_run_together():
    with pytest.raises(DatasetError, match="mixed kinds"):
        ensure_single_kind(
            [
                PairwiseRecord(id="1", **PAIR),
                PointwiseRecord(id="2", task_description="t", output="o"),
            ]
        )
    with pytest.raises(DatasetError, match="no valid records"):
        ensure_single_kind([])


# --- runner -----------------------------------------------------------------------------------


def _pairs(n: int) -> list[PairwiseRecord]:
    return [
        PairwiseRecord(id=f"r{i}", task_description="t", output_a=f"a{i}", output_b="b")
        for i in range(n)
    ]


def _failing_provider(bad: set[str], payload: dict):
    def fn(messages):
        if slots(messages)[0] in bad:
            raise ProviderError("judge down", retryable=False)
        return payload

    return FnProvider(fn)


def test_runner_evaluates_pairwise_in_input_order(hermetic_settings, single_criterion_payload):
    judge = Judge(provider=FakeProvider(single_criterion_payload), settings=hermetic_settings)
    seen = []
    report = DatasetRunner(judge, max_workers=3, progress=lambda d, t: seen.append((d, t))).run(
        _pairs(7)
    )
    assert [o.id for o in report.outcomes] == [f"r{i}" for i in range(7)]
    assert report.succeeded == 7 and report.complete and seen[-1] == (7, 7)
    assert report.criteria == ["Helpfulness"]


def test_runner_routes_pointwise_and_reference(hermetic_settings):
    provider = FnProvider(lambda m: make_pointwise_payload(("Correctness", 8), ("Completeness", 6)))
    judge = Judge(provider=provider, settings=hermetic_settings)
    report = DatasetRunner(judge).run(
        [ReferenceRecord(id="1", task_description="t", output="o", reference="r", model="m")]
    )
    assert report.outcomes[0].result.mode == "reference"
    assert report.criteria == ["Correctness", "Completeness"]

    provider = FnProvider(lambda m: make_pointwise_payload(("Helpfulness", int(single_output(m)))))
    judge = Judge(provider=provider, settings=hermetic_settings)
    report = DatasetRunner(judge).run(
        [PointwiseRecord(id=str(i), task_description="t", output=str(i)) for i in (3, 9)]
    )
    assert [o.result.criteria[0].score for o in report.outcomes] == [3, 9]


def test_runner_isolates_failures(hermetic_settings, single_criterion_payload):
    judge = Judge(
        provider=_failing_provider({"a1"}, single_criterion_payload), settings=hermetic_settings
    )
    report = DatasetRunner(judge).run(_pairs(3))
    assert [o.ok for o in report.outcomes] == [True, False, True]
    assert "judge down" in report.outcomes[1].error


def test_checkpoint_resume_skips_done_and_retries_failed(
    tmp_path, hermetic_settings, single_criterion_payload
):
    checkpoint = tmp_path / "run.ckpt.jsonl"
    flaky = _failing_provider({"a1"}, single_criterion_payload)
    first = DatasetRunner(
        Judge(provider=flaky, settings=hermetic_settings), checkpoint=checkpoint
    ).run(_pairs(3))
    assert first.failed == 1
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 3

    # Starting again without --resume must not silently mix runs.
    with pytest.raises(DatasetError, match="already exists"):
        DatasetRunner(Judge(provider=flaky, settings=hermetic_settings), checkpoint=checkpoint).run(
            _pairs(3)
        )

    healthy = FakeProvider(single_criterion_payload)
    second = DatasetRunner(
        Judge(provider=healthy, settings=hermetic_settings), checkpoint=checkpoint, resume=True
    ).run(_pairs(3))
    assert healthy.calls == 1  # only the failed row was re-run
    assert second.succeeded == 3 and second.resumed == 2
    assert read_checkpoint(checkpoint, "pairwise")["r1"].ok is True  # the retry appended

    keep = FakeProvider(single_criterion_payload)
    DatasetRunner(
        Judge(provider=keep, settings=hermetic_settings),
        checkpoint=checkpoint,
        resume=True,
        retry_failed=False,
    ).run(_pairs(4))
    assert keep.calls == 1  # just the new row r3


def test_resume_refuses_changed_records(tmp_path, hermetic_settings, single_criterion_payload):
    checkpoint = tmp_path / "c.jsonl"
    judge = Judge(provider=FakeProvider(single_criterion_payload), settings=hermetic_settings)
    DatasetRunner(judge, checkpoint=checkpoint).run(_pairs(2))
    changed = [_pairs(2)[0], _pairs(2)[1].model_copy(update={"output_b": "edited"})]
    with pytest.raises(DatasetError, match="changed since it was checkpointed"):
        DatasetRunner(judge, checkpoint=checkpoint, resume=True).run(changed)


def test_checkpoint_robustness(tmp_path, hermetic_settings, single_criterion_payload):
    checkpoint = tmp_path / "c.jsonl"
    judge = Judge(provider=FakeProvider(single_criterion_payload), settings=hermetic_settings)
    DatasetRunner(judge, checkpoint=checkpoint).run(_pairs(2))
    good = checkpoint.read_text(encoding="utf-8")

    checkpoint.write_text(good + '{"id": "r9", "kin', encoding="utf-8")  # crash mid-write
    assert set(read_checkpoint(checkpoint, "pairwise")) == {"r0", "r1"}

    checkpoint.write_text("garbage\n" + good, encoding="utf-8")
    with pytest.raises(DatasetError, match="corrupt at line 1"):
        read_checkpoint(checkpoint, "pairwise")

    checkpoint.write_text(good, encoding="utf-8")
    with pytest.raises(DatasetError, match="is for a pairwise dataset"):
        read_checkpoint(checkpoint, "pointwise")


def test_stop_event_interrupts_and_resume_finishes(
    tmp_path, hermetic_settings, single_criterion_payload
):
    checkpoint = tmp_path / "c.jsonl"
    stop = threading.Event()
    judge = Judge(provider=FakeProvider(single_criterion_payload), settings=hermetic_settings)
    first = DatasetRunner(
        judge,
        max_workers=1,
        checkpoint=checkpoint,
        stop_event=stop,
        progress=lambda done, total: stop.set() if done >= 1 else None,
    ).run(_pairs(10))
    assert first.interrupted and not first.complete
    assert 1 <= len(first.outcomes) < 10

    second = DatasetRunner(judge, checkpoint=checkpoint, resume=True).run(_pairs(10))
    assert second.complete and second.succeeded == 10


def test_keyboard_interrupt_drains_in_flight_rows(hermetic_settings, single_criterion_payload):
    def fn(messages):
        if slots(messages)[0] == "a1":
            raise KeyboardInterrupt
        return single_criterion_payload

    judge = Judge(provider=FnProvider(fn), settings=hermetic_settings)
    report = DatasetRunner(judge, max_workers=1).run(_pairs(6))
    ids = {o.id for o in report.outcomes}
    assert report.interrupted
    assert "r0" in ids and "r1" not in ids and "r5" not in ids


def test_runner_rejects_duplicate_ids(hermetic_settings, single_criterion_payload):
    judge = Judge(provider=FakeProvider(single_criterion_payload), settings=hermetic_settings)
    with pytest.raises(DatasetError, match="unique"):
        DatasetRunner(judge).run([PairwiseRecord(id="x", **PAIR)] * 2)


# --- writers and summary ------------------------------------------------------------------


def _mock_run(records):
    return DatasetRunner(Judge(provider=MockProvider(), settings=_settings())).run(records)


def test_write_outcomes_formats(tmp_path):
    report = _mock_run(load_sample("summarization"))
    write_outcomes(report.outcomes, tmp_path / "out.csv")
    with (tmp_path / "out.csv").open(encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header[:5] == ["id", "model_a", "model_b", "ok", "overall_winner"]
    assert "Helpfulness (A)" in header

    write_outcomes(report.outcomes, tmp_path / "out.jsonl")
    first = json.loads((tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["result"]["schema_version"] and first["id"] == "sum-1"

    reference = _mock_run(load_sample("general_qa"))
    write_outcomes(reference.outcomes, tmp_path / "ref.json")
    rows = json.loads((tmp_path / "ref.json").read_text(encoding="utf-8"))
    assert {"token_f1", "missing_information", "unsupported_claims"} <= set(rows[0])

    with pytest.raises(DatasetError, match="Unsupported output format"):
        write_outcomes(report.outcomes, tmp_path / "out.xlsx")


def test_summary_pairwise_and_graded():
    pairwise = summarize(_mock_run(load_sample("summarization")))
    assert pairwise.kind == "pairwise" and pairwise.succeeded == 5
    assert pairwise.wins_a + pairwise.wins_b + pairwise.ties == 5
    assert {m.model for m in pairwise.models} == {"system-alpha", "system-beta"}
    assert all(0 <= m.win_rate <= 1 for m in pairwise.models)
    assert set(pairwise.criteria_means["Helpfulness"]) == {"A", "B"}

    reference = summarize(_mock_run(load_sample("general_qa")))
    assert reference.mean_token_f1 is not None
    assert reference.rows_with_unsupported_claims is not None
    assert reference.models[0].mean_overall_score >= reference.models[-1].mean_overall_score

    pointwise = summarize(_mock_run(load_sample("instruction_following")))
    assert pointwise.mean_token_f1 is None and pointwise.rows_with_unsupported_claims is None


# --- samples ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SAMPLE_DATASETS))
def test_every_sample_is_valid_and_runs_offline(name):
    sample = get_sample(name)
    records, report = load_dataset(
        [json.loads(line) for line in sample_text(name).splitlines()], settings=_settings()
    )
    assert report.ok and report.kind == sample.kind
    run = DatasetRunner(Judge(provider=MockProvider(), settings=_settings()), sample.criteria).run(
        load_sample(name)
    )
    assert run.complete and run.failed == 0


def test_pairwise_samples_decouple_models_from_positions():
    for name in ("code_generation", "summarization", "safety"):
        slots_for_alpha = {r.model_a == "system-alpha" for r in load_sample(name)}
        assert slots_for_alpha == {True, False}, name


def test_unknown_sample():
    with pytest.raises(DatasetError, match="Unknown sample dataset"):
        get_sample("nope")
