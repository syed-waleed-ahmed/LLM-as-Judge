"""Typed dataset records, kind inference, content fingerprints, and sampling."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from ..exceptions import ValidationError

DatasetKind = Literal["pairwise", "pointwise", "reference"]
KINDS: tuple[DatasetKind, ...] = ("pairwise", "pointwise", "reference")

REQUIRED_FIELDS: dict[DatasetKind, tuple[str, ...]] = {
    "pairwise": ("task_description", "output_a", "output_b"),
    "pointwise": ("task_description", "output"),
    "reference": ("task_description", "output", "reference"),
}
OPTIONAL_FIELDS: dict[DatasetKind, tuple[str, ...]] = {
    "pairwise": ("id", "model_a", "model_b"),
    "pointwise": ("id", "model"),
    "reference": ("id", "model"),
}

#: Common column names in public datasets, mapped to canonical field names.
#: An alias is used only when the canonical column is absent.
FIELD_ALIASES: dict[str, str] = {
    "task": "task_description",
    "prompt": "task_description",
    "question": "task_description",
    "instruction": "task_description",
    "response": "output",
    "expected_output": "reference",
    "expected": "reference",
    "ground_truth": "reference",
    "gold": "reference",
}


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., min_length=1)
    task_description: str = Field(..., min_length=1)


class PairwiseRecord(_Record):
    """Two outputs to compare for one task."""

    output_a: str = Field(..., min_length=1)
    output_b: str = Field(..., min_length=1)
    model_a: str | None = None
    model_b: str | None = None


class PointwiseRecord(_Record):
    """One output to grade on its own."""

    output: str = Field(..., min_length=1)
    model: str | None = None


class ReferenceRecord(PointwiseRecord):
    """One output to grade against an expected answer."""

    reference: str = Field(..., min_length=1)


Record = PairwiseRecord | PointwiseRecord | ReferenceRecord
RECORD_TYPES: dict[DatasetKind, type[_Record]] = {
    "pairwise": PairwiseRecord,
    "pointwise": PointwiseRecord,
    "reference": ReferenceRecord,
}

_R = TypeVar("_R")


def kind_of(record: Record) -> DatasetKind:
    """The dataset kind a record belongs to."""
    if isinstance(record, PairwiseRecord):
        return "pairwise"
    if isinstance(record, ReferenceRecord):  # check before its parent class
        return "reference"
    return "pointwise"


def infer_kind(columns: Iterable[str]) -> DatasetKind | None:
    """Guess the dataset kind from (canonical) column names."""
    cols = set(columns)
    if {"output_a", "output_b"} <= cols:
        return "pairwise"
    if {"output", "reference"} <= cols:
        return "reference"
    if "output" in cols:
        return "pointwise"
    return None


#: Every canonical field name across all dataset kinds.
CANONICAL_FIELDS = frozenset(
    name for fields in (*REQUIRED_FIELDS.values(), *OPTIONAL_FIELDS.values()) for name in fields
)


def canonical_columns(columns: Iterable[object]) -> dict[str, str]:
    """Map each original column name (as given) to its canonical field name.

    Names are compared after trimming whitespace, so ``" output "`` is
    ``output``. An alias such as ``prompt`` applies only when the canonical
    column is absent and no earlier column already claimed that field. Any
    other column keeps its own name and is reported as unknown, so two columns
    never silently overwrite each other.
    """
    original = [str(c) for c in columns]
    present = {c.strip() for c in original}
    claimed: set[str] = set()
    mapping: dict[str, str] = {}
    for col in original:
        name = col.strip()
        target = name
        if name not in CANONICAL_FIELDS:
            alias = FIELD_ALIASES.get(name.lower())
            if alias and alias not in present:
                target = alias
        if target in CANONICAL_FIELDS and target in claimed:
            target = col  # a second source for the same field: keep it separate
        claimed.add(target)
        mapping[col] = target
    return mapping


def record_fingerprint(record: Record) -> str:
    """Stable content hash of a record (its id excluded), for duplicates and resume."""
    payload = json.dumps(record.model_dump(exclude={"id"}), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def dataset_fingerprint(records: Sequence[Record]) -> str:
    """Version identifier of a dataset: changes when any id, row or row order changes."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(f"{record.id}:{record_fingerprint(record)}\n".encode())
    return digest.hexdigest()[:16]


def sample_records(records: Sequence[_R], n: int, *, seed: int = 0) -> list[_R]:
    """A reproducible random subset of ``n`` records, kept in their original order."""
    if n < 0:
        raise ValidationError("Sample size must be non-negative.")
    if n >= len(records):
        return list(records)
    chosen = sorted(random.Random(seed).sample(range(len(records)), n))
    return [records[i] for i in chosen]


def split_records(
    records: Sequence[_R], fractions: Sequence[float] = (0.8, 0.2), *, seed: int = 0
) -> list[list[_R]]:
    """Reproducibly split records into parts of the given fractions (summing to 1).

    Each part keeps the original record order. Rounding remainders go to the
    last part.
    """
    if not fractions or any(f < 0 for f in fractions) or abs(sum(fractions) - 1) > 1e-9:
        raise ValidationError("Split fractions must be non-negative and sum to 1.")
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    parts: list[list[_R]] = []
    start = 0
    for i, fraction in enumerate(fractions):
        end = len(order) if i == len(fractions) - 1 else start + round(fraction * len(order))
        parts.append([records[j] for j in sorted(order[start:end])])
        start = end
    return parts
