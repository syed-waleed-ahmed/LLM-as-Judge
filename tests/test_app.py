"""Dashboard smoke and flow tests with Streamlit's headless AppTest harness.

Each test runs in a temporary directory with its own database and no .env file,
using the offline mock judge, so nothing touches the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = [
    "overview",
    "playground",
    "datasets",
    "benchmarks",
    "rag",
    "agents",
    "experiments",
    "reliability",
    "about",
]


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no developer .env, so no live judge
    monkeypatch.setenv("JUDGE_PROVIDER", "mock")
    monkeypatch.setenv("JUDGE_DB_PATH", str(tmp_path / "experiments.db"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def _open(at: AppTest, page: str) -> AppTest:
    at.switch_page(f"dashboard/views/{page}.py")
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(app, page):
    _open(app, page)
    assert app.title[0].value


def test_overview_explains_the_mock_judge(app):
    assert app.title[0].value == "LLM-as-Judge"
    assert any("offline mock judge" in i.value for i in app.info)
    assert app.radio(key="judge_mode").value == "mock"


def test_playground_compares_two_outputs(app):
    _open(app, "playground")
    app.text_area(key="pw:task").set_value("Explain what an API is to a non-technical reader")
    app.text_area(key="pw:a").set_value(
        "An API is like a waiter: it takes your request to the kitchen and brings back what "
        "you asked for, so you never need to know how the kitchen works."
    )
    app.text_area(key="pw:b").set_value("An interface.")
    app.button(key="pw:run").click().run()
    assert not app.exception
    assert any("preferred overall" in s.value or "tie" in s.value for s in app.subheader)
    assert app.session_state["evaluations"] == 1


def test_playground_applies_judge_instructions(app):
    _open(app, "playground")
    app.text_area(key="pw:task").set_value("Explain recursion")
    app.text_area(key="pw:a").set_value("A function that calls itself, with a base case.")
    app.text_area(key="pw:b").set_value("Stuff repeats.")
    app.text_area(key="pw:instructions").set_value("Penalise answers without a base case.")
    app.button(key="pw:run").click().run()
    assert not app.exception
    assert any("Judged with your instructions" in c.value for c in app.caption)


def test_playground_validates_inputs(app):
    _open(app, "playground")
    app.button(key="pw:run").click().run()
    assert any("Fill in the task and both outputs" in e.value for e in app.error)


def test_playground_reference_and_grading(app):
    _open(app, "playground")
    app.text_area(key="ref:task").set_value("When did Apollo 11 land?")
    app.text_area(key="ref:output").set_value("Apollo 11 landed in 1969.")
    app.text_area(key="ref:reference").set_value("Apollo 11 landed on 20 July 1969.")
    app.button(key="ref:run").click().run()
    assert not app.exception
    assert any("Reference grading" in s.value for s in app.subheader)

    app.text_area(key="pt:task").set_value("Explain DNS briefly")
    app.text_area(key="pt:output").set_value("DNS translates domain names into IP addresses.")
    app.button(key="pt:run").click().run()
    assert any("Pointwise grading" in s.value for s in app.subheader)


def test_benchmark_runs_are_recorded_and_comparable(app):
    _open(app, "benchmarks")
    app.button(key="bm:run").click().run()
    assert not app.exception
    assert any("mock heuristic" in w.value for w in app.warning)
    app.button(key="bm:run").click().run()  # a second run to compare against

    _open(app, "experiments")
    assert app.selectbox(key="ex:base").value and app.selectbox(key="ex:cand").value
    assert any("No significant change" in i.value for i in app.info)


def test_datasets_page_runs_a_sample(app):
    _open(app, "datasets")
    app.button(key="ds:run").click().run()
    assert not app.exception
    assert any(m.label == "Rows evaluated" for m in app.metric)


def test_demo_mode_is_labelled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JUDGE_PROVIDER", "mock")
    monkeypatch.setenv("JUDGE_DEMO_MODE", "true")
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert any("Public demo" in c.value for c in at.sidebar.caption)
    _open(at, "about")
    assert any("public demo" in m.value.lower() for m in at.markdown)
