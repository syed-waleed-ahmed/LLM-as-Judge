"""Smoke tests for the Streamlit app, using Streamlit's headless AppTest harness."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_renders_without_errors():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "LLM-as-Judge Evaluation Framework"
    assert at.button[0].label == "Evaluate Outputs"


def test_app_validates_empty_inputs_before_calling_the_judge():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button[0].click().run()
    assert not at.exception
    assert any("fill in the task description" in e.value for e in at.error)
