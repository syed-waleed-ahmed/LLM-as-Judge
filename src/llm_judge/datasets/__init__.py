"""Datasets: load, validate, version, sample and evaluate datasets at scale.

Quick start::

    from llm_judge import Judge
    from llm_judge.datasets import DatasetRunner, load_dataset, summarize

    records, report = load_dataset("data.csv")   # kind inferred from the columns
    assert report.ok, report.issues
    run = DatasetRunner(Judge(), ["Helpfulness"], checkpoint="run.ckpt.jsonl").run(records)
    print(summarize(run))

Supported formats are CSV, JSON, JSONL and YAML (with the ``yaml`` extra).
There are three dataset kinds, inferred from the columns:

* ``pairwise``: ``task_description, output_a, output_b`` (+ ``model_a``, ``model_b``)
* ``pointwise``: ``task_description, output`` (+ ``model``)
* ``reference``: ``task_description, output, reference`` (+ ``model``)
"""

from __future__ import annotations

from .io import SUPPORTED_FORMATS, DatasetError, iter_rows
from .records import (
    KINDS,
    DatasetKind,
    PairwiseRecord,
    PointwiseRecord,
    Record,
    ReferenceRecord,
    dataset_fingerprint,
    infer_kind,
    kind_of,
    record_fingerprint,
    sample_records,
    split_records,
)
from .runner import DatasetRunner, RowOutcome, RunReport, read_checkpoint, write_outcomes
from .samples import SAMPLE_DATASETS, SampleDataset, get_sample, load_sample, sample_text
from .summary import DatasetSummary, ModelStats, summarize
from .validation import DatasetReport, RowIssue, load_dataset, load_rows, validate_dataset

__all__ = [
    "KINDS",
    "SAMPLE_DATASETS",
    "SUPPORTED_FORMATS",
    "DatasetError",
    "DatasetKind",
    "DatasetReport",
    "DatasetRunner",
    "DatasetSummary",
    "ModelStats",
    "PairwiseRecord",
    "PointwiseRecord",
    "Record",
    "ReferenceRecord",
    "RowIssue",
    "RowOutcome",
    "RunReport",
    "SampleDataset",
    "dataset_fingerprint",
    "get_sample",
    "infer_kind",
    "iter_rows",
    "kind_of",
    "load_dataset",
    "load_rows",
    "load_sample",
    "read_checkpoint",
    "record_fingerprint",
    "sample_records",
    "sample_text",
    "split_records",
    "summarize",
    "validate_dataset",
    "write_outcomes",
]
