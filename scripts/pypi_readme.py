"""Point README.md's relative links and images at GitHub, for the PyPI page.

PyPI shows the README without the repository's files, so relative links and
images (screenshots, ROADMAP.md, LICENSE, ...) break there. The release workflow
runs this on its own checkout just before building, pinning every link to the
release tag; the README in the repository keeps its relative links.

Usage: python scripts/pypi_readme.py --repo OWNER/NAME --ref v2.4.0 [README.md]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")
# Every Markdown link or image target: "](target)". Matching the closing "](...)"
# rather than the whole "[text](target)" also catches links that wrap an image.
_TARGET = re.compile(r"\]\(([^)\s]+)\)")
_ABSOLUTE = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|#|//)")


def absolutize(markdown: str, repo: str, ref: str) -> str:
    """Return ``markdown`` with relative link targets made absolute GitHub URLs."""

    def replace(match: re.Match[str]) -> str:
        target = match.group(1)
        if _ABSOLUTE.match(target):
            return match.group(0)
        path, sep, fragment = target.partition("#")
        path = path.removeprefix("./")
        if path.lower().endswith(IMAGE_SUFFIXES):
            url = f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"
        else:
            url = f"https://github.com/{repo}/blob/{ref}/{path}"
        return f"]({url}{sep}{fragment})"

    return _TARGET.sub(replace, markdown)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="GitHub repository, OWNER/NAME")
    parser.add_argument("--ref", required=True, help="Tag or commit the links point at")
    parser.add_argument("readme", nargs="?", default="README.md")
    args = parser.parse_args(argv)
    path = Path(args.readme)
    text = path.read_text(encoding="utf-8")
    result = absolutize(text, args.repo, args.ref)
    path.write_text(result, encoding="utf-8", newline="\n")
    # Rewriting keeps every target in place, so the two lists line up one-to-one.
    pairs = zip(_TARGET.findall(text), _TARGET.findall(result), strict=True)
    changed = sum(1 for before, after in pairs if before != after)
    print(f"{path}: {changed} relative link(s) now point at {args.repo}@{args.ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
