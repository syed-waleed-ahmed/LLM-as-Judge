"""Phase 8 through every interface: CLI, REST API and dashboard (offline mock judge)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

import llm_judge.config as config_module
from llm_judge.cli import main
from llm_judge.config import Settings

TRACE = {
    "task": "What is the weather in Paris?",
    "tools": [{"name": "weather", "required_args": ["city"]}],
    "steps": [
        {"type": "tool_call", "tool": "weather", "arguments": {"city": "Paris"}},
        {"type": "tool_result", "tool": "weather", "content": "Sunny"},
    ],
    "final_answer": "It is sunny in Paris.",
}
CONTEXTS = ["The Aurora 3 e-bike has a range of up to 80 km.", "Office hours are 9 to 5."]
ANSWER = "The Aurora 3 has a range of up to 80 km. It ships with a free helmet."


@pytest.fixture
def mock_settings(monkeypatch, tmp_path):
    settings = Settings(_env_file=None, provider="mock", db_path=str(tmp_path / "exp.db"))
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    return settings


# --- CLI ------------------------------------------------------------------------------------


def test_cli_rag(mock_settings, capsys, tmp_path):
    args = ["rag", "--question", "What is the range of the Aurora 3?", "--answer", ANSWER]
    assert (
        main([*args, *sum((["--context", c] for c in CONTEXTS), []), "--reference", "80 km"]) == 0
    )
    out = capsys.readouterr().out
    assert "Faithfulness:      50%" in out and "unsupported: It ships with a free helmet." in out

    as_json = tmp_path / "contexts.json"
    as_json.write_text(json.dumps(CONTEXTS), encoding="utf-8")
    assert main([*args, "--contexts-file", str(as_json), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["faithfulness"] == 0.5

    as_text = tmp_path / "contexts.txt"
    as_text.write_text("\n\n".join(CONTEXTS), encoding="utf-8")
    assert main([*args, "--contexts-file", str(as_text)]) == 0
    assert "context 2: not relevant" in capsys.readouterr().out


def test_cli_rag_input_errors(mock_settings, tmp_path):
    with pytest.raises(SystemExit, match="--context"):
        main(["rag", "--question", "q", "--answer", "a"])
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="list of strings"):
        main(["rag", "--question", "q", "--answer", "a", "--contexts-file", str(bad)])


def test_cli_agent(mock_settings, capsys, tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({**TRACE, "expected_tools": ["weather", "geocode"]}), "utf-8")
    assert main(["agent", "--trace-file", str(trace)]) == 0
    out = capsys.readouterr().out
    assert "Steps 2, tool calls 1" in out and "Expected tool 'geocode' was never called" in out
    assert (
        main(["agent", "--trace-file", str(trace), "--criteria", "Task Completion", "--json"]) == 0
    )
    assert json.loads(capsys.readouterr().out)["metrics"]["tool_calls"] == 1

    trace.write_text(json.dumps({"task": ""}), encoding="utf-8")
    with pytest.raises(SystemExit, match="invalid trace"):
        main(["agent", "--trace-file", str(trace)])


def test_cli_consistency_and_agreement(mock_settings, capsys):
    base = ["consistency", "--task", "Explain DNS", "--samples", "3"]
    detailed = (
        "DNS translates human-readable domain names into the numeric IP addresses that "
        "computers use to find each other, a bit like a phone book for the internet."
    )
    assert main([*base, "--output-a", detailed, "--output-b", "It is a thing."]) == 0
    out = capsys.readouterr().out
    assert "modal verdict A in 100% of samples" in out and "Temperature is 0" in out
    assert main([*base, "--output", "DNS maps names.", "--temperature", "0.5", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "pointwise" and report["temperature"] == 0.5

    for _ in range(2):
        main(["benchmark", "summarization", "--experiment", "agree"])
    capsys.readouterr()
    main(["experiments", "runs"])
    ids = [line.split()[0] for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert main(["experiments", "agreement", ids[0], ids[1]]) == 0
    out = capsys.readouterr().out
    assert "Cohen's kappa +1.000" in out and "same judge" in out
    assert main(["experiments", "agreement", ids[0], ids[1], "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["cohen_kappa"] == 1.0


# --- REST API -------------------------------------------------------------------------------


@pytest.fixture
def client(mock_settings):
    pytest.importorskip("fastapi")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
    from llm_judge.api import create_app
    from llm_judge.experiments import ExperimentStore

    settings = mock_settings.model_copy(update={"api_rate_limit_per_minute": 0})
    return TestClient(create_app(settings, store=ExperimentStore(":memory:")))


def test_api_rag_agent_and_consistency(client):
    rag = client.post(
        "/api/v1/evaluate/rag",
        json={"question": "Range of the Aurora 3?", "contexts": CONTEXTS, "answer": ANSWER},
    )
    assert rag.status_code == 200 and rag.json()["faithfulness"] == 0.5

    agent = client.post("/api/v1/evaluate/agent", json={"trace": TRACE})
    assert agent.status_code == 200 and agent.json()["checks"] == []

    consistency = client.post(
        "/api/v1/analysis/consistency",
        json={"task_description": "t", "output": "an answer", "samples": 3},
    )
    assert consistency.json()["succeeded"] == 3
    too_many = client.post(
        "/api/v1/analysis/consistency",
        json={"task_description": "t", "output": "o", "samples": 11},
    )
    assert too_many.status_code == 422
    neither = client.post("/api/v1/analysis/consistency", json={"task_description": "t"})
    assert neither.json()["error"]["code"] == "invalid_input"


def test_api_judge_agreement(client):
    import time

    ids = []
    for _ in range(2):
        job = client.post("/api/v1/benchmarks/safety/jobs", json={"experiment": "a"}).json()
        for _ in range(500):
            info = client.get(f"/api/v1/jobs/{job['id']}").json()
            if info["status"] == "completed":
                break
            time.sleep(0.01)
        ids.append(info["run_id"])
    agreement = client.get("/api/v1/runs/agreement", params={"run_a": ids[0], "run_b": ids[1]})
    assert agreement.status_code == 200 and agreement.json()["cohen_kappa"] == 1.0


# --- dashboard ----------------------------------------------------------------------------------


@pytest.fixture
def app(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JUDGE_PROVIDER", "mock")
    monkeypatch.setenv("JUDGE_DB_PATH", str(tmp_path / "experiments.db"))
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
    at.run()
    return at


def _open(at, page: str):
    at.switch_page(f"dashboard/views/{page}.py")
    at.run()
    assert not at.exception, at.exception
    return at


def test_dashboard_rag_page(app):
    _open(app, "rag").button(key="rag:run").click().run()
    assert not app.exception
    labels = {m.label: m.value for m in app.metric}
    assert labels["Faithfulness"] == "50%"  # the example's helmet claim is unsupported
    assert labels["Context recall"] == "100%"


def test_dashboard_agent_page(app):
    _open(app, "agents").button(key="ag:run").click().run()
    assert not app.exception
    assert any("missing required argument" in e.value for e in app.error)

    app.text_area(key="ag:trace").set_value("{not json").run()
    app.button(key="ag:run").click().run()
    assert any("not valid" in e.value for e in app.error)


def test_dashboard_reliability_page(app):
    _open(app, "reliability").button(key="rel:run").click().run()
    assert not app.exception
    labels = {m.label: m.value for m in app.metric}
    assert labels["Agreement with modal verdict"] == "100%"
    assert any("Record at least two runs" in i.value for i in app.info)
