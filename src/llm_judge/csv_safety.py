"""Defences for CSV files that people open in spreadsheet software.

Result files contain text the library doesn't control: judge explanations and
candidate outputs. A cell that starts with ``=``, ``+``, ``-``, ``@`` (or a tab
or carriage return) can be executed as a formula by Excel, LibreOffice or Google
Sheets ("CSV injection"). Following OWASP's guidance, such text cells are
prefixed with a single quote. Numbers are left untouched.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")


def safe_cell(value: Any) -> Any:
    """Neutralise a text value that a spreadsheet could interpret as a formula."""
    if isinstance(value, str) and value.startswith(_TRIGGERS):
        return "'" + value
    return value


def safe_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Apply :func:`safe_cell` to every value of every row."""
    return [{key: safe_cell(value) for key, value in row.items()} for row in rows]
