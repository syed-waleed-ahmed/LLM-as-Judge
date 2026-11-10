"""Bounded, incremental readers for CSV, JSON, JSONL and YAML datasets.

Rows are parsed one at a time where the format allows it (CSV, JSONL). Every
read is bounded by a byte limit and a row limit, so a huge or hostile file
fails fast with a clear error instead of exhausting memory. Nothing in a
dataset is ever executed: YAML is read with ``safe_load``.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..exceptions import ValidationError

SUPPORTED_FORMATS = (".csv", ".jsonl", ".ndjson", ".json", ".yaml", ".yml")


class DatasetError(ValidationError):
    """A dataset file can't be read, or a resume checkpoint doesn't match it."""


def detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise DatasetError(
            f"Unsupported dataset format '{suffix}'. Use one of: {', '.join(SUPPORTED_FORMATS)}."
        )
    return suffix


def iter_rows(
    path: str | Path, *, max_bytes: int | None = None, max_rows: int | None = None
) -> Iterator[dict[str, Any]]:
    """Yield the rows of a dataset file as dictionaries.

    Raises :class:`DatasetError` for missing or oversized files, malformed
    content, or more than ``max_rows`` rows.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Dataset file not found: {path}")
    suffix = detect_format(path)
    size = path.stat().st_size
    if max_bytes is not None and size > max_bytes:
        raise DatasetError(
            f"{path.name} is {size:,} bytes; the limit is {max_bytes:,} (JUDGE_MAX_DATASET_BYTES)."
        )

    count = 0
    for row in _rows(path, suffix):
        count += 1
        if max_rows is not None and count > max_rows:
            raise DatasetError(
                f"{path.name} has more than {max_rows:,} rows (JUDGE_MAX_DATASET_ROWS)."
            )
        yield row


def _rows(path: Path, suffix: str) -> Iterator[dict[str, Any]]:
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return
            for row in reader:
                if None in row:  # more values than header columns
                    raise DatasetError(f"{path.name} line {reader.line_num}: too many values.")
                yield dict(row)
    elif suffix in (".jsonl", ".ndjson"):
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetError(
                        f"{path.name} line {line_no}: invalid JSON ({exc})."
                    ) from None
                if not isinstance(obj, dict):
                    raise DatasetError(f"{path.name} line {line_no}: expected a JSON object.")
                yield obj
    else:
        yield from _document_rows(path, suffix)


def _document_rows(path: Path, suffix: str) -> Iterator[dict[str, Any]]:
    """Rows from a whole-document format (JSON or YAML): a list, or {"records": [...]}."""
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path.name}: invalid JSON ({exc}).") from None
    else:
        try:
            import yaml
        except ImportError:  # pragma: no cover - depends on the environment
            raise DatasetError(
                "Reading YAML datasets needs PyYAML: pip install 'llm-as-judge[yaml]'."
            ) from None
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise DatasetError(f"{path.name}: invalid YAML ({exc}).") from None

    if isinstance(data, dict) and isinstance(data.get("records"), list):
        data = data["records"]
    if not isinstance(data, list):
        raise DatasetError(f"{path.name}: expected a list of records or {{'records': [...]}}.")
    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise DatasetError(f"{path.name}: record {i} is not an object.")
        yield item
