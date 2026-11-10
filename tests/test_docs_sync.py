"""Documentation must not drift from the code: every CLI command and REST route is documented."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from llm_judge.cli import _build_parser
from llm_judge.config import Settings

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (ROOT / "API_REFERENCE.md").read_text(encoding="utf-8")


def _commands() -> list[str]:
    parser = _build_parser()
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    seen: set[int] = set()
    names = []
    for name, sub in action.choices.items():
        if id(sub) not in seen:  # skip aliases
            seen.add(id(sub))
            names.append(name)
    return names


@pytest.mark.parametrize("command", _commands())
def test_every_cli_command_is_documented(command):
    assert f"`llm-judge {command}`" in REFERENCE


def test_every_rest_route_is_documented():
    pytest.importorskip("fastapi")
    from llm_judge.api import create_app

    spec = create_app(Settings(_env_file=None, provider="mock", db_path=":memory:")).openapi()
    for path, operations in spec["paths"].items():
        for method in operations:
            row = re.compile(rf"\|\s*{method.upper()}\s*\|\s*`{re.escape(path)}`\s*\|")
            assert row.search(REFERENCE), f"{method.upper()} {path} is not in API_REFERENCE.md"


@pytest.mark.parametrize(
    "name",
    [
        "README.md",
        "ARCHITECTURE.md",
        "API_REFERENCE.md",
        "BENCHMARKS.md",
        "CHANGELOG.md",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "DEPLOYMENT.md",
        "LICENSE",
        "ROADMAP.md",
        "SECURITY.md",
    ],
)
def test_project_documents_exist(name):
    assert (ROOT / name).stat().st_size > 200
