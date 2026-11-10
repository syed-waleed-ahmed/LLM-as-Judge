"""MkDocs hook: make repository-relative links work on the documentation site.

Docs pages include repository files (README sections, ROADMAP.md, ...), so their
links are written relative to the repository, such as ``DEPLOYMENT.md`` or
``.env.example``. Those would 404 on the site. This hook rewrites each one in the
rendered HTML:

- a file that a docs page renders, or a page under ``docs/``, links to that page;
- any other repository file links to it on GitHub.

A link to a file that doesn't exist is logged as a warning, which fails
``mkdocs build --strict``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger("mkdocs.plugins.docs_links")

REPO = Path(__file__).resolve().parent.parent
GITHUB = "https://github.com/syed-waleed-ahmed/LLM-as-Judge"

# Repository files that a docs page renders in full, and that page's path.
RENDERED = {
    "ARCHITECTURE.md": "project/architecture/",
    "API_REFERENCE.md": "project/api-reference/",
    "BENCHMARKS.md": "project/benchmarks/",
    "CHANGELOG.md": "project/changelog/",
    "CONTRIBUTING.md": "project/contributing/",
    "DEPLOYMENT.md": "project/deployment/",
    "ROADMAP.md": "project/roadmap/",
    "SECURITY.md": "project/security/",
}

_HREF = re.compile(r'(<a\b[^>]*?\shref=")([^"]*)(")')
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def repo_path(href: str) -> str:
    """``../../ROADMAP.md`` -> ``ROADMAP.md`` (links are relative to the repository root)."""
    return "/".join(part for part in href.split("/") if part not in ("", ".", ".."))


def resolve(href: str, site_url: str, repo: Path = REPO) -> str | None:
    """The working URL for a repository-relative link, or None to leave it unchanged."""
    if not href or href.startswith(("#", "/")) or _SCHEME.match(href):
        return None
    path, _, fragment = href.partition("#")
    if path.endswith("/"):
        return None  # already a site URL produced by MkDocs
    target = repo_path(path)
    is_markdown = target.endswith(".md")
    if not is_markdown and not (repo / target).exists():
        return None  # a site-relative asset, not a repository file
    anchor = f"#{fragment}" if fragment else ""
    site = site_url.rstrip("/") + "/"
    if target in RENDERED:
        return site + RENDERED[target] + anchor
    if target == "README.md" and not fragment:
        return site
    if target.startswith("docs/") and is_markdown:
        page = target.removeprefix("docs/").removesuffix(".md")
        page = "" if page == "index" else page.removesuffix("/index") + "/"
        if (repo / target).exists():
            return site + page + anchor
    if not (repo / target).exists():
        log.warning("Broken repository link %r: %s does not exist", href, target)
        return None
    kind = "tree" if (repo / target).is_dir() else "blob"
    return f"{GITHUB}/{kind}/main/{target}{anchor}"


def on_page_content(html: str, page: Any, config: Any, files: Any) -> str:
    site_url = config.get("site_url") or "/"

    def rewrite(match: re.Match[str]) -> str:
        url = resolve(match.group(2), site_url)
        return match.group(0) if url is None else match.group(1) + url + match.group(3)

    return _HREF.sub(rewrite, html)
