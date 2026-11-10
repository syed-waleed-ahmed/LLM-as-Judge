"""The release workflow publishes CHANGELOG sections as GitHub Release notes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from llm_judge import __version__

ROOT = Path(__file__).resolve().parent.parent


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "release_notes", ROOT / "scripts" / "release_notes.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release_notes = _load_script()

SAMPLE = """# Changelog

Intro text.

## [Unreleased]

### Added
- Something new.

## [2.0.0] - 2026-11-10

Platform release.

### Fixed
- A bug.

## [1.0.0] - 2026-07-04

First release.

[Unreleased]: https://example.org/compare/v2.0.0...HEAD
[2.0.0]: https://example.org/releases/tag/v2.0.0
"""


def test_current_version_has_release_notes() -> None:
    """Bumping the version without a changelog section would fail the release."""
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    notes = release_notes.section(changelog, __version__)
    assert notes and notes.strip(), f"CHANGELOG.md has no section for {__version__}"


def test_section_stops_at_the_next_version() -> None:
    notes = release_notes.section(SAMPLE, "2.0.0")
    assert notes == "Platform release.\n\n### Fixed\n- A bug.\n"


def test_last_section_stops_at_link_references() -> None:
    assert release_notes.section(SAMPLE, "1.0.0") == "First release.\n"


def test_tag_prefix_and_missing_versions() -> None:
    assert release_notes.section(SAMPLE, "v2.0.0") == release_notes.section(SAMPLE, "2.0.0")
    assert release_notes.section(SAMPLE, "3.0.0") is None
    # "1.0.0" must not match a longer version such as "1.0.0-rc1" or "11.0.0".
    assert release_notes.section("## [11.0.0]\nx\n", "1.0.0") is None


def test_main_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(SAMPLE, encoding="utf-8")
    assert release_notes.main(["v2.0.0", str(changelog)]) == 0
    assert "Platform release." in capsys.readouterr().out
    assert release_notes.main(["9.9.9", str(changelog)]) == 1
    assert "No section for 9.9.9" in capsys.readouterr().err
    assert release_notes.main([]) == 2
