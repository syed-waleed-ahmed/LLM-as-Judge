"""The MkDocs hook that makes repository-relative links work on the docs site."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
SITE = "https://example.org/docs/"
GITHUB = "https://github.com/syed-waleed-ahmed/LLM-as-Judge"


def _load_hook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("docs_links", ROOT / "scripts" / "docs_links.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hook = _load_hook()


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        ("DEPLOYMENT.md", SITE + "project/deployment/"),
        ("DEPLOYMENT.md#3-rest-api", SITE + "project/deployment/#3-rest-api"),
        ("../../ROADMAP.md", SITE + "project/roadmap/"),
        ("docs/design/phase-0-audit.md", SITE + "design/phase-0-audit/"),
        ("README.md", SITE),
        ("README.md#installation", f"{GITHUB}/blob/main/README.md#installation"),
        (".env.example", f"{GITHUB}/blob/main/.env.example"),
        ("LICENSE", f"{GITHUB}/blob/main/LICENSE"),
        ("scripts/e2e_dashboard.py", f"{GITHUB}/blob/main/scripts/e2e_dashboard.py"),
        ("src/llm_judge", f"{GITHUB}/tree/main/src/llm_judge"),
    ],
)
def test_repository_links_are_rewritten(href: str, expected: str) -> None:
    assert hook.resolve(href, SITE) == expected


@pytest.mark.parametrize(
    "href",
    [
        "https://example.com/page",
        "mailto:someone@example.org",
        "#section",
        "/absolute/path/",
        "../guide/providers/",  # already a site URL
        "assets/images/favicon.png",  # a site asset, not a repository file
        "",
    ],
)
def test_other_links_are_left_alone(href: str) -> None:
    assert hook.resolve(href, SITE) is None


def test_missing_repository_files_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="mkdocs.plugins.docs_links"):
        assert hook.resolve("NO_SUCH_FILE.md", SITE) is None
    assert "Broken repository link" in caplog.text


def test_on_page_content_rewrites_only_anchor_hrefs() -> None:
    html = (
        '<p><a href="DEPLOYMENT.md">guide</a> '
        '<a class="x" href="https://example.com">site</a> '
        '<img src="DEPLOYMENT.md"></p>'
    )
    config = SimpleNamespace(get=lambda key: SITE if key == "site_url" else None)
    out = hook.on_page_content(html, page=None, config=config, files=None)
    assert f'<a href="{SITE}project/deployment/">guide</a>' in out
    assert 'href="https://example.com"' in out
    assert '<img src="DEPLOYMENT.md">' in out


def test_every_rendered_file_exists_and_has_a_docs_page() -> None:
    for name, page in hook.RENDERED.items():
        assert (ROOT / name).is_file(), name
        source = ROOT / "docs" / (page.rstrip("/") + ".md")
        assert source.is_file(), f"{name} maps to {page}, but {source} is missing"
        assert name in source.read_text(encoding="utf-8"), f"{source} doesn't include {name}"
