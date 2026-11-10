"""The coverage badge published with the docs site."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "coverage_badge", ROOT / "scripts" / "coverage_badge.py"
)
assert _spec and _spec.loader
coverage_badge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(coverage_badge)


@pytest.mark.parametrize(
    ("percent", "message", "color"),
    [
        (98.7, "98%", "brightgreen"),  # floored, never rounded up
        (95.0, "95%", "brightgreen"),
        (90.2, "90%", "green"),
        (72.0, "72%", "yellow"),
        (40.0, "40%", "red"),
    ],
)
def test_badge_payload(percent, message, color):
    payload = coverage_badge.badge(percent)
    assert payload == {
        "schemaVersion": 1,
        "label": "coverage",
        "message": message,
        "color": color,
    }


def test_main_reads_a_pytest_cov_report(tmp_path, capsys):
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps({"totals": {"percent_covered": 97.4}}), encoding="utf-8")
    out = tmp_path / "site" / "badges" / "coverage.json"
    assert coverage_badge.main([str(report), str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["message"] == "97%"
    assert coverage_badge.main([]) == 2
