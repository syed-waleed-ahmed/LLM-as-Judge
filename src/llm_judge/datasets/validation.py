"""Dataset loading with a full validation report.

:func:`load_dataset` never stops at the first problem. It reads every row and
reports missing fields, duplicate ids, duplicate content, unknown columns and
oversized fields, and it returns the valid records alongside the report.
Callers decide whether to proceed (``report.ok``) or to skip invalid rows.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError

from ..config import Settings, get_settings
from .io import DatasetError, detect_format, iter_rows
from .records import (
    KINDS,
    OPTIONAL_FIELDS,
    RECORD_TYPES,
    REQUIRED_FIELDS,
    DatasetKind,
    PairwiseRecord,
    PointwiseRecord,
    Record,
    ReferenceRecord,
    canonical_columns,
    dataset_fingerprint,
    infer_kind,
    kind_of,
    record_fingerprint,
)

_MAX_ISSUES = 100
_PREVIEW_CHARS = 200


class RowIssue(BaseModel):
    """One problem found in one row (``row`` is 1-based)."""

    row: int
    id: str | None = None
    field: str | None = None
    severity: Literal["error", "warning"]
    message: str


class DatasetReport(BaseModel):
    """Everything :func:`load_dataset` learned about a dataset."""

    source: str
    format: str
    kind: DatasetKind | None
    rows: int
    valid_rows: int
    columns: list[str] = Field(default_factory=list)
    aliases: dict[str, str] = Field(default_factory=dict)
    unknown_columns: list[str] = Field(default_factory=list)
    duplicate_ids: list[str] = Field(default_factory=list)
    duplicate_content: list[list[str]] = Field(default_factory=list)
    issues: list[RowIssue] = Field(default_factory=list)
    issues_truncated: int = 0
    error_count: int = 0
    warning_count: int = 0
    #: Content-derived version id of the valid records (None if there are none).
    fingerprint: str | None = None
    preview: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every row is valid and the kind is known."""
        return self.kind is not None and self.valid_rows == self.rows and self.rows > 0


class _NonScalarError(ValueError):
    pass


def _stringify(value: object) -> str | None:
    """Text for a field value; nested lists or objects are rejected, not stringified."""
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple, set)):
        raise _NonScalarError(type(value).__name__)
    text = value if isinstance(value, str) else str(value)
    return text if text.strip() else None


def _preview(record: Record) -> dict[str, Any]:
    return {
        k: (v[:_PREVIEW_CHARS] + "..." if isinstance(v, str) and len(v) > _PREVIEW_CHARS else v)
        for k, v in record.model_dump(exclude_none=True).items()
    }


def load_dataset(
    source: str | Path | Sequence[Mapping[str, Any] | Record],
    *,
    kind: DatasetKind | None = None,
    settings: Settings | None = None,
    preview_rows: int = 5,
) -> tuple[list[Record], DatasetReport]:
    """Load and validate a dataset from a file path or in-memory rows.

    ``source`` may be a path (CSV, JSON, JSONL, YAML), a list of dicts, or a
    list of record objects. ``kind`` is inferred from the columns if omitted.
    Rows without an ``id`` get their 0-based row index. Raises
    :class:`DatasetError` only when the source as a whole can't be read.
    """
    settings = settings or get_settings()
    if kind is not None and kind not in KINDS:
        raise DatasetError(f"Unknown dataset kind '{kind}'. Use one of: {', '.join(KINDS)}.")

    if isinstance(source, (str, Path)):
        path = Path(source)
        fmt = detect_format(path).lstrip(".")
        raw: Iterable[Mapping[str, Any]] = iter_rows(
            path, max_bytes=settings.max_dataset_bytes, max_rows=settings.max_dataset_rows
        )
        label = str(path)
    else:
        if len(source) > settings.max_dataset_rows:
            raise DatasetError(f"More than {settings.max_dataset_rows:,} rows supplied.")
        raw = [
            r.model_dump(exclude_none=True)
            if isinstance(r, (PairwiseRecord, PointwiseRecord, ReferenceRecord))
            else r
            for r in source
        ]
        fmt, label = "python", "<in-memory>"

    rows = [{str(k): v for k, v in r.items()} for r in raw]
    columns: list[str] = []
    for row in rows:
        for col in row:
            if col not in columns:
                columns.append(col)
    mapping = canonical_columns(columns)
    canonical = set(mapping.values())
    resolved_kind = kind or infer_kind(canonical)

    report = DatasetReport(
        source=label,
        format=fmt,
        kind=resolved_kind,
        rows=len(rows),
        valid_rows=0,
        columns=columns,
        aliases={k: v for k, v in mapping.items() if k != v},
    )
    issues: list[RowIssue] = []
    if resolved_kind is None:
        issues.append(
            RowIssue(
                row=0,
                severity="error",
                message="Can't infer the dataset kind: expected output_a/output_b (pairwise), "
                "output (pointwise), or output + reference (reference) columns.",
            )
        )
        _finish(report, issues, [])
        return [], report

    known = set(REQUIRED_FIELDS[resolved_kind]) | set(OPTIONAL_FIELDS[resolved_kind])
    report.unknown_columns = [c for c in columns if mapping[c] not in known]
    model = RECORD_TYPES[resolved_kind]

    records: list[Record] = []
    source_row: dict[int, int] = {}  # id(record) -> original 0-based row index
    for index, row in enumerate(rows):
        try:
            fields = {mapping[k]: _stringify(v) for k, v in row.items() if mapping[k] in known}
        except _NonScalarError as exc:
            issues.append(
                RowIssue(
                    row=index + 1,
                    severity="error",
                    message=f"field values must be text or numbers, not {exc}",
                )
            )
            continue
        record_id = fields.get("id") or str(index)
        missing = [f for f in REQUIRED_FIELDS[resolved_kind] if not fields.get(f)]
        if missing:
            issues.append(
                RowIssue(
                    row=index + 1,
                    id=record_id,
                    field=", ".join(missing),
                    severity="error",
                    message=f"missing required field(s): {', '.join(missing)}",
                )
            )
            continue
        fields["id"] = record_id
        try:
            record = model.model_validate({k: v for k, v in fields.items() if v is not None})
        except PydanticValidationError as exc:  # pragma: no cover - guarded by checks above
            issues.append(RowIssue(row=index + 1, id=record_id, severity="error", message=str(exc)))
            continue
        for name in REQUIRED_FIELDS[resolved_kind]:
            value = fields.get(name) or ""
            if len(value) > settings.max_input_chars:
                issues.append(
                    RowIssue(
                        row=index + 1,
                        id=record_id,
                        field=name,
                        severity="warning",
                        message=f"{len(value):,} characters; will be truncated to "
                        f"{settings.max_input_chars:,} (JUDGE_MAX_INPUT_CHARS)",
                    )
                )
        records.append(record)  # type: ignore[arg-type]
        source_row[id(record)] = index

    # Duplicate ids make resume ambiguous, so they are errors; duplicate
    # content is only suspicious, so it's a warning.
    by_id: dict[str, list[int]] = defaultdict(list)
    for record in records:
        by_id[record.id].append(source_row[id(record)])
    duplicate_ids = sorted(rid for rid, idx in by_id.items() if len(idx) > 1)
    if duplicate_ids:
        for rid in duplicate_ids:
            issues.append(
                RowIssue(
                    row=by_id[rid][1] + 1,
                    id=rid,
                    field="id",
                    severity="error",
                    message=f"duplicate id '{rid}' (appears {len(by_id[rid])} times)",
                )
            )
        records = [r for r in records if r.id not in set(duplicate_ids)]
    report.duplicate_ids = duplicate_ids

    by_content: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_content[record_fingerprint(record)].append(record.id)
    report.duplicate_content = [ids for ids in by_content.values() if len(ids) > 1]
    for ids in report.duplicate_content:
        issues.append(
            RowIssue(
                row=0,
                id=ids[0],
                severity="warning",
                message=f"identical content in rows with ids {ids}",
            )
        )

    report.valid_rows = len(records)
    report.preview = [_preview(r) for r in records[:preview_rows]]
    _finish(report, issues, records)
    return records, report


def _finish(report: DatasetReport, issues: list[RowIssue], records: list[Record]) -> None:
    # Errors first so truncation never hides them behind warnings.
    issues.sort(key=lambda i: (i.severity != "error", i.row))
    report.error_count = sum(1 for i in issues if i.severity == "error")
    report.warning_count = len(issues) - report.error_count
    report.issues = issues[:_MAX_ISSUES]
    report.issues_truncated = max(0, len(issues) - _MAX_ISSUES)
    report.fingerprint = dataset_fingerprint(records) if records else None


def validate_dataset(
    source: str | Path | Sequence[Mapping[str, Any] | Record],
    *,
    kind: DatasetKind | None = None,
    settings: Settings | None = None,
) -> DatasetReport:
    """Validate a dataset and return only the report."""
    return load_dataset(source, kind=kind, settings=settings)[1]


def ensure_single_kind(records: Sequence[Record]) -> DatasetKind:
    """Return the shared kind of ``records``, or raise if they're mixed or empty."""
    if not records:
        raise DatasetError("The dataset has no valid records.")
    kinds = {kind_of(r) for r in records}
    if len(kinds) > 1:
        raise DatasetError(f"Records of mixed kinds can't run together: {sorted(kinds)}.")
    return kinds.pop()
