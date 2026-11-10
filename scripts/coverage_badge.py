"""Write a shields.io endpoint badge for test coverage.

Usage: python scripts/coverage_badge.py coverage.json site/badges/coverage.json

Reads the JSON report from ``pytest --cov --cov-report=json`` and writes the
endpoint format shields.io renders (https://shields.io/badges/endpoint-badge).
The docs workflow publishes it with the site, so the README badge always shows
the coverage of the latest ``main``; no third-party coverage service is needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def badge(percent: float) -> dict[str, object]:
    """The endpoint-badge payload for a coverage percentage."""
    if percent >= 95:
        color = "brightgreen"
    elif percent >= 85:
        color = "green"
    elif percent >= 70:
        color = "yellow"
    else:
        color = "red"
    return {
        "schemaVersion": 1,
        "label": "coverage",
        "message": f"{int(percent)}%",  # floor, so the badge never rounds up
        "color": color,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    report = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    percent = float(report["totals"]["percent_covered"])
    out = Path(argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(badge(percent)) + "\n", encoding="utf-8")
    print(f"coverage {percent:.2f}% -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
