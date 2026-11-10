"""Print the CHANGELOG.md section for one version, for use as GitHub Release notes.

Usage: python scripts/release_notes.py 2.0.0 [path/to/CHANGELOG.md]

A leading "v" is accepted ("v2.0.0"). Exits 1 if the version has no section,
so the release workflow fails rather than publishing empty notes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def section(changelog: str, version: str) -> str | None:
    """Return the body of the ``## [version]`` section, without its heading."""
    version = version.removeprefix("v")
    heading = re.compile(rf"^## \[{re.escape(version)}\][^\n]*\n", re.MULTILINE)
    match = heading.search(changelog)
    if not match:
        return None
    rest = changelog[match.end() :]
    # The section ends at the next version heading or at the link references.
    end = re.search(r"^(## \[|\[[^\]]+\]: )", rest, re.MULTILINE)
    body = rest[: end.start()] if end else rest
    return body.strip() + "\n"


def main(argv: list[str]) -> int:
    if not 1 <= len(argv) <= 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(argv[1] if len(argv) == 2 else "CHANGELOG.md")
    notes = section(path.read_text(encoding="utf-8"), argv[0])
    if notes is None:
        print(f"No section for {argv[0]} in {path}", file=sys.stderr)
        return 1
    sys.stdout.write(notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
